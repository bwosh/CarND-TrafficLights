import argparse
import cv2
import numpy as np
import os
import time
import torch

from models.dla import get_pose_net

def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--folder", required=True, type=str)
    parser.add_argument("--model", type=str, default="")

    parser.add_argument("--input_size", type=int, default=512)

    parser.add_argument("--K", type=int, default=8)
    parser.add_argument("--vis_thresh", type=float, default=0.1)

    args = parser.parse_args()
    return args

args = get_args()

# Load model
print("Loading model...")
model = get_pose_net(34, heads={'hm': 1, 'wh': 2}, head_conv=-1).cuda()
state_dict = torch.load(args.model)
model.load_state_dict(state_dict)
model.cuda()
model.eval()

# Do for all images
images = os.listdir(args.folder)

def get_results(im_path, model, args):
    img = cv2.imread(im_path)
    print(f"Original size: {img.shape[1]}x{img.shape[0]}")
    img = cv2.resize(img, (args.input_size, args.input_size), interpolation=cv2.INTER_AREA)
    img = torch.tensor(img.transpose(2,0,1), dtype=torch.float)/255
    img = img.cuda().unsqueeze(0)

    start_time = time.time()
    heatmap, widthheight = model.forward(img)[0]   
    stop_time = time.time() 

    ms = (stop_time-start_time)*1000
    return ms

execution_times = []

for image in images:
    path = os.path.join(args.folder, image)
    print(f"Processing {image}...", end=' ')

    ms = get_results(path, model, args)
    
    execution_times.append(ms)
    print(f"Done in {ms:.0f}ms")
print(f"Avg inference time: {np.mean(execution_times):0.2f}ms ({len(execution_times)} samples)")
