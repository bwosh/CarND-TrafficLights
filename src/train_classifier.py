import argparse
import cv2
import numpy as np
import os
from tqdm import tqdm
from torch.optim.lr_scheduler import StepLR
from sklearn.model_selection import train_test_split
import albumentations as a
import time
import torch
from torch.utils.data import Dataset, DataLoader

from threading import Lock

import tensorflow as tf

config = tf.ConfigProto(gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.5))
config.gpu_options.allow_growth = True
session = tf.Session(config=config)

best_model_path = "../models/best.h5"

epochs = 110
start_lr = 0.01
slr_step = 30

dummy_check = False
batch_size = 16


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--red_folders", required=True, type=str)
    parser.add_argument("--nored_folders", required=True, type=str)
    parser.add_argument("--input_size", type=int, default=32)

    args = parser.parse_args()
    return args

class RedNoRedDataset(Dataset):
    def __init__(self, size, red_forlder_paths=None, nored_folder_paths=None):
        self.size = size
        self.aug = None
        self.lock = Lock()
        red_files = []
        nored_files = []

        if red_forlder_paths is not None:
            for path in red_forlder_paths:
                red_files += [os.path.join(path,f) for f in list(os.listdir(path)) if not f.startswith('.')]

        if nored_folder_paths is not None:
            for path in nored_folder_paths:
                nored_files += [os.path.join(path,f) for f in list(os.listdir(path)) if not f.startswith('.')]

        self.files = red_files + nored_files
        self.labels = np.array([1] * len(red_files) + [0] * len(nored_files))

    def split(self, percent=0.8):
        train = RedNoRedDataset(self.size)
        test = RedNoRedDataset(self.size)

        files_train, files_test, labels_train, labels_test = train_test_split(self.files ,self.labels, test_size=1-percent, random_state=42)

        train.files = files_train
        train.labels = labels_train
        test.files = files_test
        test.labels = labels_test
        return train, test

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        self.lock.acquire()
        file_path = self.files[index]
        try:
            img = cv2.imread(self.files[index])
            if self.aug is not None:
                img = self.aug(image=img)['image']
            img = cv2.resize(img,(self.size, self.size))
            #print(img.shape, file_path)
            label = self.labels[index]
            img = img/255
            img = (img).transpose(2,0,1)
            img = torch.tensor(img, dtype=torch.float32)
        except:
            print("ERROR", index, file_path)
        finally:
            self.lock.release() 
        return img, label

if __name__=='__main__':

    args = get_args()

    red_folders = args.red_folders.split(',')
    nored_folders = args.nored_folders.split(',')
    dataset = RedNoRedDataset( args.input_size, red_folders, nored_folders)
    dataset_train, dataset_test = dataset.split()

    aug = a.Compose([
                a.HorizontalFlip(p=0.5),
                a.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.05, rotate_limit=15,p=0.5),
                a.RandomBrightness(0.3,p=0.5),
                a.RandomContrast(0.3,p=0.5),
                a.GaussNoise(p=0.5),
                a.Blur(blur_limit=5,p=0.2),
            ],p=0.95)

    dataset_train.aug = aug

    trainloader = DataLoader(dataset_train, batch_size=8, num_workers=0, shuffle=True)
    testloader = DataLoader(dataset_test, batch_size=8, num_workers=0, shuffle=True)


    from keras.layers import Conv2D, Flatten, Dense, MaxPool2D, BatchNormalization, Activation
    from keras.models import Sequential, model_from_json, load_model
    from keras.optimizers import SGD
    from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
    from keras import backend as K

    model = Sequential()
    model.add(Conv2D(32, kernel_size=7, strides=1, padding='same', input_shape=(32,32,3)))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPool2D(pool_size=2, strides=2, padding='same'))

    model.add(Conv2D(32, kernel_size=3, strides=1, padding='same'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))

    model.add(Conv2D(32, kernel_size=3, strides=1, padding='same'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))

    model.add(Flatten())
    model.add(Dense(4))
    model.add(Activation('relu'))
    model.add(Dense(1, activation='sigmoid'))

    print(model.summary())

    #K.set_learning_phase(0)
    #with open("model.json", 'r') as json_file:
    #    loaded_model_json = json_file.read()
    #model = model_from_json(loaded_model_json)
    #weights_path = "model.h5"
    #model.load_weights(weights_path)
    model = load_model('model1.h5')

    optimizer = SGD(lr=0.0001)
    model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['acc'])

    def generator(loader, epochs):
        for e in range(epochs):
            for batch_idx,batch in enumerate(loader): 
                input, clsid = batch

                input = input.numpy().transpose(0,2,3,1)
                clsid = clsid.numpy()
                yield input, clsid

    epochs = 1#10*20
    callbacks = [ModelCheckpoint("best1.h5", monitor='val_acc', verbose=1, 
                save_best_only=True, save_weights_only=False, mode='max', period=1),

                ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=30, verbose=0, 
                mode='auto', min_lr=0.000001)]
    
    t = tqdm(total=len(trainloader)*epochs)
    inputs = []
    labels = []
    for input, clsid in generator(trainloader,epochs):
        #print(input.shape, clsid.shape)
        for b in range(input.shape[0]):
            inputs.append(input[b])
            labels.append(clsid[b])
        t.update()
    inputs = np.array(inputs)
    labels = np.array(labels)

    model.fit(inputs, labels, epochs=3, batch_size=8)

    #model.fit_generator(generator(trainloader,epochs), 
    #                    validation_data=generator(testloader, epochs),
    #                    steps_per_epoch = len(trainloader), 
    #                    validation_steps = len(testloader),
    #                    epochs = epochs, 
    #                    callbacks = callbacks,
    #                    workers=0, use_multiprocessing=False, max_queue_size=1)
    #K.set_learning_phase(0)
    #x = model.evaluate_generator(generator(testloader, 1), 
    #    steps=len(testloader), 
    #    #callbacks=None, 
    #    max_queue_size=1, 
    #    workers=0, use_multiprocessing=False)
    #print("DONE.")
    model.save("model1.h5")

    #print('Best val acc:', best_val_acc)
    #print(f"Inference speed(ms) MEAN:{np.mean(inference_speeds)} STD:{np.std(inference_speeds)} MIN:{np.min(inference_speeds)} MAX:{np.max(inference_speeds)}")
    print('Finished Training')