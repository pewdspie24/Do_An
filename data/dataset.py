r""" Dataloader builder for few-shot semantic segmentation dataset  """
from torchvision import transforms
from torch.utils.data import DataLoader

from data.pascal import DatasetPASCAL
from data.coco import DatasetCOCO
from data.fss import DatasetFSS
from data.custom import DatasetCustom
import data.transform as transform
class FSSDataset:

    @classmethod
    def initialize(cls, img_size, datapath, use_original_imgsize):

        cls.datasets = {
            'pascal': DatasetPASCAL,
            'coco': DatasetCOCO,
            'fss': DatasetFSS,
            'custom': DatasetCustom,
        }

        cls.img_mean = [0.485, 0.456, 0.406]
        mean =  [item * 255 for item in cls.img_mean]
        cls.img_std = [0.229, 0.224, 0.225]
        std = [item * 255 for item in cls.img_std]
        cls.datapath = datapath
        cls.use_original_imgsize = use_original_imgsize

        cls.transform = transforms.Compose([transforms.Resize(size=(img_size, img_size)),
                                            transforms.ToTensor(),
                                            transforms.Normalize(cls.img_mean, cls.img_std)])

        cls.transform_train = transform.Compose([
            transform.RandScale([0.9, 1.1]),
            transform.RandRotate([-10, 10], padding=mean, ignore_label=255),
            transform.RandomGaussianBlur(),
            transform.RandomHorizontalFlip(),
            transform.Crop([img_size, img_size], crop_type='rand', padding=mean, ignore_label=255),
            transform.ToTensor(),
            transform.Normalize(mean=cls.img_mean, std=cls.img_std)
        ])
        cls.transform_val = transform.Compose([
            transform.Resize(size=img_size),
            transform.ToTensor(),
            transform.Normalize(mean=cls.img_mean, std=cls.img_std)
        ])


    @classmethod
    def build_dataloader(cls, benchmark, bsz, nworker, fold, split, shot):
        # Changing shot for experiments
        # Force randomness during training for diverse episode combinations
        # Freeze randomness during testing for reproducibility
        shuffle = split == 'trn'
        nworker = nworker if split == 'trn' else 0
        if split == "trn":
            dataset = cls.datasets[benchmark](cls.datapath, fold=fold, transform=cls.transform, split=split, shot=shot, use_original_imgsize=cls.use_original_imgsize)
        else:
            dataset = cls.datasets[benchmark](cls.datapath, fold=fold, transform=cls.transform, split=split, shot=shot, use_original_imgsize=cls.use_original_imgsize)

        dataloader = DataLoader(dataset, batch_size=bsz, shuffle=shuffle, num_workers=nworker)
        return dataloader
