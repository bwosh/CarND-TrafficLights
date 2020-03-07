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

# Network model preparation
model = get_dla34_centernet()
optimizer = SGD(lr=0.1)
model.compile(optimizer=optimizer, loss='mse')
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
best = 1000
# TODO loading best

if args.val:
    args.epochs = 1

for epoch in range(args.epochs):
    loader = tqdm(train_loader, leave=False)
    for batch_idx,batch in enumerate(loader):   
        input, heatmaps, widhtandheight, reg_mask, ind = batch

        input = input.numpy()
        heatmaps = heatmaps.numpy()
        widhtandheight = widhtandheight.numpy()
        reg_mask = reg_mask.numpy() 
        ind = ind.numpy() 

        print(input.shape)
        print(heatmaps.shape)
        print(widhtandheight.shape)
        print(reg_mask.shape)
        print(ind.shape)

        # TODO: more training here
        exit()
        #torch.Size([8, 3, 512, 512])
        #torch.Size([8, 1, 128, 128])
        #torch.Size([8, 13, 2])
        #torch.Size([8, 13])
        #torch.Size([8, 13])
