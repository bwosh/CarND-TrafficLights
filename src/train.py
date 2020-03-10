# python train.py --input ../../data/coco --output ../models --batch_size=8 --restore ../models/model_0.11231545358896255.json --lr 5e-7 --val

import os
import torch

from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import SingleClassDataset
from get_coco_images import extract_class_annotations
from opts import get_args
from utils.result import ResultTracker
from keras.models import model_from_json

from models.dla_keras import get_dla34_centernet
from keras.optimizers import SGD
args = get_args()
import numpy as np

from keras import backend as K
from keras.callbacks.callbacks import LambdaCallback, LearningRateScheduler
import time
from tqdm import tqdm

import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.8
set_session(tf.Session(config=config))

# Network model preparation
if args.restore != "":
    print("#### Loading model", args.restore)
    with open(args.restore, 'r') as json_file:
        loaded_model_json = json_file.read()
    model = model_from_json(loaded_model_json)
    weights_path = args.restore.replace('.json','.h5')
    print("#### Loading weights", args.restore)
    model.load_weights(weights_path)
else:
    model = get_dla34_centernet()

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
    args.epochs = 0

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

best_loss = 1000
def on_epoch_end(epoch,logs):
    global best_loss
    global args
    print(f"Epoch   \nLoss    : {logs['loss']}\nVal loss: {logs['val_loss']}")
    loss = logs['val_hm_loss'] + logs['val_multiply_1_loss'] #val_hm_loss #val_multiply_1_loss
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

current_lr = args.lr
optimizer = SGD(lr=current_lr)
model.compile(optimizer=optimizer, loss=[weighted_mse_loss,weighted_regl1_loss])

if args.epochs > 0:
    callbacks = [LambdaCallback(on_epoch_end=on_epoch_end),
                LearningRateScheduler(schedule, verbose=1) ]
    model.fit_generator(generator(train_loader, args.epochs), 
                        validation_data=generator(val_loader, args.epochs),
                        steps_per_epoch = len(train_loader), 
                        validation_steps = len(val_loader),
                        epochs = args.epochs, 
                        callbacks = callbacks,
                        workers=1, use_multiprocessing=False)
else:
    print("Starting validation...")
    from utils.keras_helpers import save_image, calc_map

    losses = []
    times = []
    maps = []
    for bi,batch in enumerate(tqdm(generator(val_loader, 1), total = len(val_loader))):
        input, output = batch
        gt_hm, gt_wh = output
        time_a = time.time()
        hm, wh = model.predict([input[0],np.ones_like(input[1])])
        time_b = time.time()

        gt_wh = gt_wh.astype('float32')

        l_a = K.get_value(weighted_mse_loss(hm, gt_hm))
        l_b = K.get_value(weighted_regl1_loss(wh, gt_wh))
        total_loss = l_a + l_b
        losses.append(total_loss)

        for i in range(input[0].shape[0]):
            maps.append( calc_map(hm[i], wh[i], gt_hm[i], gt_wh[i]) )
            save_image(f"temp/{bi}_{i}_img.png", input[0][i], verbose=False)
            save_image(f"temp/{bi}_{i}_hm.png", hm[i], max_div=True, verbose=False)
            save_image(f"temp/{bi}_{i}_wh.png", wh[i], max_div=True, verbose=False)

        times.append(time_b-time_a)
    print("Mean losses:", np.mean(losses))
    print("Avg batch time losses:", np.mean(times))
    print("Mean mAP:", np.mean(maps))

# TODO decoder
# TODO mAP calculation
# TODO light inference code
