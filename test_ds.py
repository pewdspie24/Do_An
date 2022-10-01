from data.dataset import FSSDataset
import sys
import numpy
numpy.set_printoptions(threshold=sys.maxsize)

FSSDataset.initialize(img_size=400, datapath="../my_data", use_original_imgsize=False)
dataloader_trn = FSSDataset.build_dataloader('custom', 1, 8, 0, 'trn')
dataloader_val = FSSDataset.build_dataloader('custom', 1, 8, 0, 'val')