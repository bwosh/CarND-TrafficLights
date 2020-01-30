import os
import torch

from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import SingleClassDataset
from get_coco_images import extract_class_annotations
from loss import CenterNetLoss
from models.dla import get_pose_net
from opts import get_args
from utils.result import ResultTracker


args = get_args()

# Network model preparation
model = get_pose_net(34, heads={'hm': 1, 'wh': 2}, head_conv=-1).cuda()

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
optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
criterion = CenterNetLoss(args)

for epoch in range(args.epochs):
    print(f"*** TRAIN, epoch {epoch+1}/{args.epochs} ***")
    model.train()
    tracker = ResultTracker()
    for batch in tqdm(train_loader, leave=False):
        input, heatmaps, widhtandheight = batch
        input, heatmaps, widhtandheight = input.cuda(), heatmaps.cuda(), widhtandheight.cuda()

        optimizer.zero_grad()
        output = model(input)
        output_hm = output[0]['hm']
        output_wh = output[0]['wh']

        loss, loss_stats = criterion(heatmaps, output_hm, widhtandheight, output_wh)
        tracker.add_loss_stats(loss_stats)

        loss.backward()
        optimizer.step()
    tracker.print_avg_loss_stats()

    print(f"*** VALIDATION, epoch {epoch+1}/{args.epochs} ***")
    model.eval()
    tracker = ResultTracker()
    with torch.no_grad():
        for batch in tqdm(val_loader, leave=False):
            input, heatmaps, widhtandheight = batch
            input, heatmaps, widhtandheight = input.cuda(), heatmaps.cuda(), widhtandheight.cuda()

            output = model(input)
            output_hm = output[0]['hm']
            output_wh = output[0]['wh']

            loss, loss_stats = criterion(heatmaps, output_hm, widhtandheight, output_wh)
            tracker.add_loss_stats(loss_stats)
    tracker.print_avg_loss_stats()

    # TODO LOSS, trainer, validation(mAP iou mAP@class AP 50 75 s m l)
    # TODO readme: carla, pytorch->quantization->onnx->tf 1.4

torch.save(model.state_dict(),os.path.join(args.output, "final_model.pth"))