r""" Hypercorrelation Squeeze training (validation) code """
import argparse
import os

import torch.optim as optim
import torch.nn as nn
import torch

from model.hsnet import HypercorrSqueezeNetwork
from common.logger import Logger, AverageMeter
from common.evaluation import Evaluator
from common import utils
from data.dataset import FSSDataset

def train_mask_nshot(model, query_img, support_img, support_mask, shot):
    logit_mask_agg = 0
    predict_mask_agg = 0
    for s_idx in range(shot):
        # print(support_img.shape)
        logit_mask = model(query_img, support_img[:, s_idx], support_mask[:, s_idx])
        # print(logit_mask.shape)
        # if self.use_original_imgsize:
        #     org_qry_imsize = tuple([batch['org_query_imsize'][1].item(), batch['org_query_imsize'][0].item()])
        #     logit_mask = F.interpolate(logit_mask, org_qry_imsize, mode='bilinear', align_corners=True)

        logit_mask_agg += logit_mask.clone()
        predict_mask_agg += logit_mask.argmax(dim=1).clone()
        if shot == 1: return (logit_mask_agg, predict_mask_agg)

    # Average & quantize predictions given threshold (=0.5)
    bsz = predict_mask_agg.size(0)
    max_vote = predict_mask_agg.view(bsz, -1).max(dim=1)[0]
    # print("max_vote1", predict_mask_agg.view(bsz, -1).shape)
    max_vote = torch.stack([max_vote, torch.ones_like(max_vote).long()])
    # print("max_vote2", max_vote.shape)
    max_vote = max_vote.max(dim=0)[0].view(bsz, 1, 1)
    # print("max_vote3", max_vote.shape)
    pred_mask = predict_mask_agg.float() / max_vote
    pred_mask[pred_mask < 0.5] = 0
    pred_mask[pred_mask >= 0.5] = 1
    logit_mask = logit_mask_agg / shot

    return logit_mask, pred_mask

def train(epoch, model, dataloader, optimizer, training, shot):
    r""" Train HSNet """

    # Force randomness during training / freeze randomness during testing
    utils.fix_randseed(None) if training else utils.fix_randseed(0)
    model.module.train_mode() if training else model.module.eval()
    average_meter = AverageMeter(dataloader.dataset)

    for idx, batch in enumerate(dataloader):

        # 1. Hypercorrelation Squeeze Networks forward pass
        batch = utils.to_cuda(batch)
        # print(batch['support_masks'].shape)
        # logit_mask = model(batch['query_img'], batch['support_imgs'].squeeze(1), batch['support_masks'].squeeze(1), shot)
        # pred_mask = logit_mask.argmax(dim=1)
        query_img = batch['query_img']
        support_img = batch['support_imgs']
        support_mask = batch['support_masks']

        logit_mask, pred_mask = train_mask_nshot(model, query_img, support_img, support_mask, shot)

        # pred_mask = train_mask_nshot(model, query_img, support_img, support_mask, shot)

        # 2. Compute loss & update model parameters
        loss = model.module.compute_objective(logit_mask, batch['query_mask'])
        # loss = model.module.compute_objective(logit_mask, batch['query_mask'])
        if training:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # 3. Evaluate prediction
        area_inter, area_union = Evaluator.classify_prediction(pred_mask, batch)
        average_meter.update(area_inter, area_union, batch['class_id'], loss.detach().clone())
        average_meter.write_process(idx, len(dataloader), epoch, write_batch_idx=50)

    # Write evaluation results
    average_meter.write_result('Training' if training else 'Validation', epoch)
    avg_loss = utils.mean(average_meter.loss_buf)
    miou, fb_iou = average_meter.compute_iou()

    return avg_loss, miou, fb_iou


if __name__ == '__main__':

    # Arguments parsing
    parser = argparse.ArgumentParser(description='Hypercorrelation Squeeze Pytorch Implementation')
    parser.add_argument('--datapath', type=str, default='../Datasets_HSN')
    parser.add_argument('--benchmark', type=str, default='custom', choices=['pascal', 'coco', 'fss', 'custom'])
    parser.add_argument('--logpath', type=str, default='newD_oldW_trans_5shot')
    parser.add_argument('--bsz', type=int, default=2)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--niter', type=int, default=2000)
    parser.add_argument('--nworker', type=int, default=2)
    parser.add_argument('--nwshot', type=int, default=3)
    parser.add_argument('--fold', type=int, default=0, choices=[0, 1, 2, 3])
    parser.add_argument('--backbone', type=str, default='resnet101', choices=['vgg16', 'resnet50', 'resnet101', 'resnet101_custom'])
    parser.add_argument('--resume', type=bool, default=False)
    parser.add_argument('--colab', type=bool, default=False)
    args = parser.parse_args()
    Logger.initialize(args, training=True)

    # Model initialization
    model = HypercorrSqueezeNetwork(args.backbone, False, colab = args.colab)
    Logger.log_params(model)

    # Device setup
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    Logger.info('# available GPUs: %d' % torch.cuda.device_count())
    model = nn.DataParallel(model)
    model.to(device)

    # Helper classes (for training) initialization
    # optimizer = optim.Adam([{"params": model.parameters(), "lr": args.lr}])
    new_optimizer = optim.SGD([{"params": model.parameters(), "lr": args.lr, "momentum": 0.9}])
    lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(new_optimizer, 'max')
    
    
    ######################### Enter previous epochs #############################
    init_eps = 0
    
    if args.resume:
        # checkpoint = torch.load('logs/'+args.logpath+'.log/best_model.pt')
        # model.load_state_dict(checkpoint['model_state_dict'])
        # new_optimizer.load_state_dict(checkpoint['optimizer_state_dict'])   
        # lr_scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        # init_eps = checkpoint['epoch']
        model.load_state_dict(torch.load(os.path.join(args.logpath, "best_model.pt")))
    Evaluator.initialize()

    # Dataset initialization
    FSSDataset.initialize(img_size=400, datapath=args.datapath, use_original_imgsize=False)
    dataloader_trn = FSSDataset.build_dataloader(args.benchmark, args.bsz, args.nworker, args.fold, 'trn', args.nwshot)
    dataloader_val = FSSDataset.build_dataloader(args.benchmark, args.bsz, args.nworker, args.fold, 'val', args.nwshot)

    # Train HSNet
    best_val_miou = float('-inf')
    best_val_loss = float('inf')
    for epoch in range(init_eps, args.niter):

        trn_loss, trn_miou, trn_fb_iou = train(epoch, model, dataloader_trn, new_optimizer, training=True, shot=args.nwshot)
        with torch.no_grad():
            val_loss, val_miou, val_fb_iou = train(epoch, model, dataloader_val, new_optimizer, training=False, shot=args.nwshot)
        lr_scheduler.step(val_miou)
        # Save the best model
        if val_miou > best_val_miou:
            best_val_miou = val_miou
            Logger.save_model_miou(model, epoch, val_miou, new_optimizer, lr_scheduler)
        if epoch%25==0:
            Logger.save_model_event(model, epoch, val_miou)

        Logger.tbd_writer.add_scalars('data/loss', {'trn_loss': trn_loss, 'val_loss': val_loss}, epoch)
        Logger.tbd_writer.add_scalars('data/miou', {'trn_miou': trn_miou, 'val_miou': val_miou}, epoch)
        Logger.tbd_writer.add_scalars('data/fb_iou', {'trn_fb_iou': trn_fb_iou, 'val_fb_iou': val_fb_iou}, epoch)
        Logger.tbd_writer.add_scalar('data/lr', new_optimizer.param_groups[0]['lr'], epoch)
        Logger.tbd_writer.flush()
    Logger.tbd_writer.close()
    Logger.info('==================== Finished Training ====================')
