import argparse
import cv2
import numpy as np
import os
import time
import torch
import pickle
import skvideo.io

from tqdm import tqdm

from models.dla import get_pose_net
from utils.transforms import decode_results

def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--input_path", required=True, type=str)
    parser.add_argument("--output_path", required=True, type=str)
    parser.add_argument("--model", type=str, default="")

    parser.add_argument("--input_size", type=int, default=512)
    parser.add_argument("--save_crops_path", type=str, default=None)

    parser.add_argument("--K", type=int, default=8)
    parser.add_argument("--vis_thresh", type=float, default=0.3)

    args = parser.parse_args()
    return args

args = get_args()

# Load model
print("Creating network...")
model = get_pose_net(34, heads={'hm': 1, 'wh': 2}, head_conv=-1).cuda()
print("Loading model...")
state_dict = torch.load(args.model)
model.load_state_dict(state_dict)
print("Moving to CUDA...")
model.cuda()
model.eval()
print("Model loaded...")

def get_results(frame, model, args):
    # Prepare input
    img = frame[:,:,::-1].copy()
    result_img = img.copy()
    o_width, o_height = img.shape[1], img.shape[0]
    img = cv2.resize(img, (args.input_size, args.input_size), interpolation=cv2.INTER_AREA)
    img = torch.tensor(img.transpose(2,0,1), dtype=torch.float)/255
    img = img.cuda().unsqueeze(0)

    # Do inference
    start_time = time.time()
    output = model.forward(img)[0]   
    output_hm = output['hm']
    output_wh = output['wh']

    # Decode results
    dets = decode_results(output_hm, output_wh, args.K)[0]
    stop_time = time.time() 
    
    x_factor = 4 * o_width/args.input_size
    y_factor = 4 * o_height/args.input_size
    final_bboxes = []
    for det_idx in range(len(dets['bboxes'])):
        bbox= dets['bboxes'][det_idx]
        score = dets['scores'][det_idx]

        x1,y1,x2,y2 = bbox
        x1,y1,x2,y2 = x1*x_factor,y1*y_factor,x2*x_factor,y2*y_factor
        x1,y1,x2,y2 = int(x1),int(y1),int(x2),int(y2)

        thickness = max(max(result_img.shape)//150,3)

        if score> args.vis_thresh:
            final_bboxes.append((x1,y1,x2,y2, score))
            cv2.rectangle(result_img,(x1,y1),(x2,y2),(0,255,0), thickness=thickness)
            cv2.putText(result_img, f"{score:0.2f}", (x1+20+thickness,y1+20+thickness), cv2.FONT_HERSHEY_DUPLEX, 1, (0,255,255))

    ms = (stop_time-start_time)*1000
    return ms, final_bboxes, result_img

execution_times = []

print("Starting video processing...")
crop_id=0
writer = skvideo.io.FFmpegWriter(args.output_path)
with skvideo.io.FFmpegReader(args.input_path) as reader:
    (numframes, _, _, _) = reader.getShape()
    for frame_id, frame in enumerate(tqdm(reader, total=numframes)):
        ms, bboxes, result_img = get_results(frame, model, args)
        if args.save_crops_path is not None and len(bboxes) > 0:
            for x1,y1,x2,y2, score in bboxes:
                crop = frame[y1:y2,x1:x2,::-1].copy()
                cv2.imwrite(f"{args.save_crops_path}/{crop_id}.png",crop)
                crop_id += 1

        writer.writeFrame(result_img)
writer.close()