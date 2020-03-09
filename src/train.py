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
from keras.callbacks.callbacks import LambdaCallback

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

def wh_data_to_mask(output, reg_mask, ind):
    mask = np.zeros((output.shape[0], 128,128,2), dtype=float)
    wh = np.zeros((output.shape[0], 128,128,2), dtype=float)

    for batch_index in range(output.shape[0]):
        for i in range(ind.shape[1]):
            if reg_mask[batch_index,i]>0:
                val = ind[batch_index,i]
                x = val % 128
                y = val // 128

                mask[batch_index,x,y,:] = 1
                wh[batch_index,x,y,0] = output[batch_index,i,0]
                wh[batch_index,x,y,1] = output[batch_index,i,1]

    return mask, wh

def generator(loader, epochs):
    for e in range(epochs):
        for batch_idx,batch in enumerate(loader): 
            input, heatmaps, widhtandheight, reg_mask, ind = batch

            input = input.numpy().transpose(0,2,3,1)
            heatmaps = heatmaps.numpy().transpose(0,2,3,1)
            widhtandheight = widhtandheight.numpy()
            reg_mask = reg_mask.numpy() 
            ind = ind.numpy()

            mask, wh = wh_data_to_mask(widhtandheight, reg_mask, ind)

            yield [input, mask], [heatmaps, wh]

def weighted_regl1_loss(gt, pred):
    x = K.zeros_like(gt)
    x = K.cast(K.greater(gt,x), 'float32')
    x = K.sum(x)
    eps=1e-4
    return args.wh_weight * K.mean(K.abs(gt-pred))/(x+eps)

def weighted_mse_loss(gt, pred):
    return K.mean(K.square(gt-pred)) * args.hm_weight

optimizer = SGD(lr=0.01)
model.compile(optimizer=optimizer, loss=[weighted_mse_loss,weighted_regl1_loss])

callbacks = [LambdaCallback(on_epoch_end=lambda e,l:print(l)) ]
model.fit_generator(generator(train_loader, args.epochs), steps_per_epoch = len(train_loader), epochs = args.epochs, callbacks = callbacks)

# TODO preview outputs
# TODO add validation loss
