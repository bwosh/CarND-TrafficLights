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
        return train, test

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        file_path = self.files[index]
        img = cv2.imread(self.files[index])
        img = cv2.resize(img,(self.size, self.size))
        label = self.labels[index]
        #cv2.imwrite(f"./temp/{index}_{label}.png", img)
        img = img/255-0.5
        #print(f"{np.min(img):0.2f} {np.max(img):0.2f}, {np.mean(img):0.2f}, {np.std(img):0.2f}")
        img = (img).transpose(2,0,1)
        img = torch.tensor(img, dtype=torch.float32)
        return img, label

dataset = RedNoRedDataset( args.input_size, red_folders, nored_folders)
t1,t2 = dataset.split()
print(len(t1))
print(len(t2))
exit()

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, padding = 1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, 3, padding = 1)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 128, 3, padding = 1)
        self.bn3 = nn.BatchNorm2d(128)
        #self.conv4 = nn.Conv2d(128, 256, 3, padding = 1)
        #self.bn4 = nn.BatchNorm2d(256)
        #self.conv5 = nn.Conv2d(256, 512, 3, padding = 1)
        #self.bn5 = nn.BatchNorm2d(512)
        self.avg_pool = nn.AdaptiveAvgPool2d((1,1))
        self.aaa = 128#512
        self.fc3 = nn.Linear(self.aaa, 2)

    def forward(self, x, verbose = False):
        if verbose:
            print("INPUT", x.shape)
        x = F.relu(self.bn1(self.conv1(x)))
        if verbose:
            print("CONV1", x.shape)
        x = F.relu(self.bn2(self.conv2(x)))

        if verbose:
            print("CONV2",x.shape)
        x = F.relu(self.bn3(self.conv3(x)))

        if verbose:
            print("CONV3",x.shape)
        #x = F.relu(self.bn4(self.conv4(x)))
        #if verbose:
        #    print("CONV4",x.shape)
        #x = F.relu(self.bn5(self.conv5(x)))
        #if verbose:
        #    print("CONV5",x.shape)

        x = self.avg_pool(x)
        #x = x.view(-1, 2)
        x = torch.flatten(x,1)
        if verbose:
            print("FLATTEN",x.shape)
        x = self.fc3(x)
        return x


dummy_check = False
batch_size = 16
epochs = 480
slr_step = 25
net = Net()
net=net.cuda()

if dummy_check:
    dummy = torch.zeros((batch_size,3,32,32), dtype=torch.float32).cuda()
    print(net.forward(dummy, True).shape)
    exit()

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
scheduler = StepLR(optimizer, step_size=slr_step, gamma=0.1)
trainloader = DataLoader(dataset, batch_size=batch_size, num_workers=4, shuffle=True)

net.train()
for epoch in range(epochs):  # loop over the dataset multiple times

    running_loss = 0.0
    losses = []
    accs = []
    loader = tqdm(trainloader)
    for i, data in enumerate(loader, 0):
        # get the inputs; data is a list of [inputs, labels]
        inputs, labels = data
        inputs, labels = inputs.cuda(), labels.cuda()

        # zero the parameter gradients
        optimizer.zero_grad()

        #print(inputs.shape)

        # forward + backward + optimize
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        outputs = outputs.detach().cpu().numpy()
        labels = labels.detach().cpu().numpy()
        #print(np.argmax(outputs, axis=1), labels)
        acc = np.mean(np.argmax(outputs, axis=1)==labels)
        accs.append(acc)
        losses.append(float(loss))

        loader.desc = f"E:{epoch+1}/{epochs} L:{np.mean(losses):.5f} ACC:{np.mean(accs):.5f}"
    scheduler.step()
print('Finished Training')