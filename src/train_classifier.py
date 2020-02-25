import argparse
import cv2
import numpy as np
import os
import torch
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from torch.optim.lr_scheduler import StepLR
from sklearn.model_selection import train_test_split
import albumentations as a
import time
from models.cls_model import ClassifierNet

best_model_path = "../models/best_cls.pth"
best_val_acc = 0.9534497090606816 

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

args = get_args()

red_folders = args.red_folders.split(',')
nored_folders = args.nored_folders.split(',')

class RedNoRedDataset(Dataset):
    def __init__(self, size, red_forlder_paths=None, nored_folder_paths=None):
        self.size = size
        self.aug = None
        red_files = []
        nored_files = []

        if red_forlder_paths is not None:
            for path in red_folders:
                red_files += [path+f for f in list(os.listdir(path)) if not f.startswith('.')]

        if nored_folder_paths is not None:
            for path in nored_folder_paths:
                nored_files += [path+f for f in list(os.listdir(path)) if not f.startswith('.')]

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
        file_path = self.files[index]
        img = cv2.imread(self.files[index])
        if self.aug is not None:
            img = self.aug(image=img)['image']
        img = cv2.resize(img,(self.size, self.size))
        label = self.labels[index]
        img = img/255-0.5
        img = (img).transpose(2,0,1)
        img = torch.tensor(img, dtype=torch.float32)
        return img, label

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


net = ClassifierNet()
print(best_model_path)
if os.path.isfile(best_model_path):
    print("Loading model...")
    state_dict = torch.load(best_model_path)
    net.load_state_dict(state_dict)
net=net.cuda()

if dummy_check:
    print("CHECKING")
    dummy = torch.zeros((batch_size,3,32,32), dtype=torch.float32)
    dummy = dummy.cuda()
    print(net.forward(dummy).shape)
    exit()

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(net.parameters(), lr=start_lr)
scheduler = StepLR(optimizer, step_size=slr_step, gamma=0.1)
trainloader = DataLoader(dataset_train, batch_size=batch_size, num_workers=batch_size, shuffle=True)
testloader = DataLoader(dataset_test, batch_size=1, num_workers=batch_size, shuffle=True)

inference_speeds = []
loader = tqdm(range(epochs))
for epoch in loader:  # loop over the dataset multiple times
    net.train()
    losses = []
    accs = []
    for i, data in enumerate(trainloader, 0):
        # get the inputs; data is a list of [inputs, labels]
        inputs, labels = data
        inputs, labels = inputs.cuda(), labels.cuda()

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        outputs = outputs.detach().cpu().numpy()
        labels = labels.detach().cpu().numpy()
        acc = np.mean(np.argmax(outputs, axis=1)==labels)
        accs.append(acc)
        losses.append(float(loss))

        loader.set_description(f"E{epoch+1}/{epochs}  [TRAIN] L:{np.mean(losses):.5f} ACC:{np.mean(accs):.5f}")
    scheduler.step()

    net.eval()
    with torch.no_grad():
        losses = []
        accs = []
        for i, data in enumerate(testloader, 0):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data
            inputs, labels = inputs.cuda(), labels.cuda()

            time_a = time.time()
            outputs = net(inputs)
            time_b = time.time()
            inference_speeds.append(1000*(time_b-time_a))
            outputs = outputs.detach().cpu().numpy()
            labels = labels.detach().cpu().numpy()
            acc = np.mean(np.argmax(outputs, axis=1)==labels)
            accs.append(acc)
            losses.append(float(loss))

            loader.set_description(f"E:{epoch+1}/{epochs} [VAL] L:{np.mean(losses):.5f} ACC:{np.mean(accs):.5f}")
        val_acc = np.mean(accs)
        if val_acc>best_val_acc:
            best_val_acc = val_acc
            print('NEW BEST FOUND! val acc:', best_val_acc)
            torch.save(net.state_dict(), best_model_path)

print('Best val acc:', best_val_acc)
print(f"Inference speed(ms) MEAN:{np.mean(inference_speeds)} STD:{np.std(inference_speeds)} MIN:{np.min(inference_speeds)} MAX:{np.max(inference_speeds)}")
print('Finished Training')