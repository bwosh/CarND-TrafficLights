import os
import torch

from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import SingleClassDataset
from get_coco_images import extract_class_annotations
from opts import get_args
from utils.result import ResultTracker

from models.dla_keras import get_dla34_centernet
from keras.optimizers import SGD
args = get_args()
import numpy as np

from keras import backend as K

# Network model preparation
model = get_dla34_centernet()
# TODO : model loading

print("========================================================")

# Datasets
annot_train, annot_val = extract_class_annotations(args.input, args.class_name)
train_dataset = SingleClassDataset(annot_train, args.input, 
                args.input_size, args.input_size, 
                (args.output_size, args.output_size), augment=True)
val_dataset = SingleClassDataset(annot_val, args.input, 
                args.input_size, args.input_size, 
                (args.output_size, args.output_size), augment=False)

# Data loaders
train_loader = DataLoader(train_dataset, shuffle=True, 
                    batch_size=args.batch_size, num_workers=args.num_workers)
val_loader = DataLoader(val_dataset, shuffle=False, 
                    batch_size=args.batch_size, num_workers=args.num_workers)

# Training
if args.val:
    args.epochs = 1

def wh_data_to_mask(output, mask, ind):
    # TODO replace this dummy implementation
    mask = np.ones((output.shape[0], 128,128,2), dtype=float)
    wh = np.ones((output.shape[0], 128,128,2), dtype=float)

    return mask, wh

def generator(loader):
    for batch_idx,batch in enumerate(loader): 
        input, heatmaps, widhtandheight, reg_mask, ind = batch

        input = input.numpy().transpose(0,2,3,1)
        heatmaps = heatmaps.numpy().transpose(0,2,3,1)
        widhtandheight = widhtandheight.numpy()
        reg_mask = reg_mask.numpy() 
        ind = ind.numpy()

        mask, wh = wh_data_to_mask(widhtandheight, reg_mask, ind)

        yield [input, mask], [heatmaps, wh]

def reg1_loss(gt, pred):
    # TODO make it reg1 loss, not MSE
    return K.mean(K.square(gt-pred))

optimizer = SGD(lr=0.1)
model.compile(optimizer=optimizer, loss=['mse',reg1_loss])

steps_per_epoch = len(train_loader) // args.batch_size
print(f"LEN={len(train_loader)},STEPS={steps_per_epoch}, BS={args.batch_size}")
model.fit_generator(generator(train_loader), steps_per_epoch = steps_per_epoch, epochs = 2)