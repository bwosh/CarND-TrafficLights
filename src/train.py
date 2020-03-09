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
from keras.callbacks.callbacks import LambdaCallback, LearningRateScheduler

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
                y = val % 128
                x = val // 128

                mask[batch_index,x,y,:] = 1
                wh[batch_index,x,y,0] = output[batch_index,i,0]
                wh[batch_index,x,y,1] = output[batch_index,i,1]

    return mask, wh

# TODO remove previews later
import cv2
def save_image(path, array):
    print(f"Saving {path}: {array.shape}")
    img = array.copy()
    if img.shape[2]==3:
        img = np.clip(img * 255,0,255).astype('uint8')
        cv2.imwrite(path, img)
    else:
        img = np.mean(img, axis=2)
        img = np.clip(img * 255,0,255).astype('uint8')
        cv2.imwrite(path, img)

def save_batch_images(idx, input, masks, heatmaps, wh):
    print("IDX",idx)
    for n in range(input.shape[0]):
        filename_base=f"./temp/preview_batch{idx}_n{n}_"
        save_image(filename_base+"input.png", input[n])
        save_image(filename_base+"masks.png", masks[n])
        save_image(filename_base+"heatmaps.png", heatmaps[n])
        save_image(filename_base+"wh.png", wh[n])
    exit()

def generator(loader, epochs):
    idx = 0
    for e in range(epochs):
        for batch_idx,batch in enumerate(loader): 
            input, heatmaps, widhtandheight, reg_mask, ind = batch

            input = input.numpy().transpose(0,2,3,1)
            heatmaps = heatmaps.numpy().transpose(0,2,3,1)
            widhtandheight = widhtandheight.numpy()
            reg_mask = reg_mask.numpy() 
            ind = ind.numpy()

            mask, wh = wh_data_to_mask(widhtandheight, reg_mask, ind)

            # TODO remove: save_batch_images(idx,input, mask, heatmaps, wh) 

            idx+=1
            yield [input, mask], [heatmaps, wh]

def weighted_regl1_loss(gt, pred):
    x = K.zeros_like(gt)
    x = K.cast(K.greater(gt,x), 'float32')
    x = K.sum(x)
    eps=1e-4
    return args.wh_weight * K.sum(K.abs(gt-pred))/(x+eps)

def weighted_mse_loss(gt, pred):
    return K.mean(K.square(gt-pred)) * args.hm_weight

current_lr = args.lr
optimizer = SGD(lr=current_lr)
model.compile(optimizer=optimizer, loss=[weighted_mse_loss,weighted_regl1_loss])


best_loss = 1000
def on_epoch_end(epoch,logs):
    global best_loss
    global args
    print(f"\nLoss:{logs['loss']} Val loss:{logs['val_loss']}")
    loss = logs['val_loss']
    if loss<best_loss:
        model_json = model.to_json()
        with open(os.path.join(args.output, f"model_{loss}.json"), "w") as json_file:
            json_file.write(model_json)
        model.save_weights(os.path.join(args.output, f"model_{loss}.h5"))
        print("Saved model to disk:",loss)
        best_loss=loss

def schedule(epoch):
    global current_lr
    global args

    epochs = [int(n) for n in args.lr_epochs.split(',')]
    gammas = [float(n) for n in args.lr_gammas.split(',')]

    for i in range(len(epochs)):
        if epoch+1 == epochs[i]:
            current_lr *= gammas[i]
            print("Change LR to", current_lr)

    return current_lr

callbacks = [LambdaCallback(on_epoch_end=on_epoch_end),
             LearningRateScheduler(schedule, verbose=1) ]
model.fit_generator(generator(train_loader, args.epochs), 
                    validation_data=generator(val_loader, args.epochs),
                    steps_per_epoch = len(train_loader), 
                    validation_steps = len(val_loader),
                    epochs = args.epochs, 
                    callbacks = callbacks)

# TODO decoder
# TODO mAP calculation
# TODO light inference code
