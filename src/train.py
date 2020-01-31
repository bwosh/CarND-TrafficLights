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
if (args.restore !=""):
    print(f"Loading model from {args.restore}")
    state_dict = torch.load(args.restore)
    model.load_state_dict(state_dict)

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

if args.val:
    args.epochs = 1

for epoch in range(args.epochs):
    if args.val:
        print("ONLY VALIDATION MODE SELECTED.")
    else:
        print(f"*** TRAIN, epoch {epoch+1}/{args.epochs} ***")
        model.train()
        tracker = ResultTracker(args)
        loader = tqdm(train_loader, leave=False)
        for batch in loader:
            input, heatmaps, widhtandheight, reg_mask, ind = batch
            input, heatmaps, widhtandheight, reg_mask, ind = input.cuda(), heatmaps.cuda(), widhtandheight.cuda(), reg_mask.cuda(), ind.cuda()

            optimizer.zero_grad()
            output = model(input)
            output_hm = output[0]['hm']
            output_wh = output[0]['wh']

            loss, loss_stats = criterion(heatmaps, output_hm, widhtandheight, output_wh, reg_mask, ind)
            tracker.add_loss_stats(loss_stats)
            tracker.save_IoU_mAP(ind, reg_mask, output_hm, widhtandheight, output_wh)
            loader.desc = tracker.get_running_loss_text()

            loss.backward()
            optimizer.step()
        tracker.print_avg_loss_stats()
        tracker.print_IoU_mAP_stats()

    print(f"*** VALIDATION, epoch {epoch+1}/{args.epochs} ***")
    model.eval()
    tracker = ResultTracker(args)
    with torch.no_grad():
        loader = tqdm(val_loader, leave=False)
        for batch in loader:
            input, heatmaps, widhtandheight, reg_mask, ind = batch
            input, heatmaps, widhtandheight, reg_mask, ind = input.cuda(), heatmaps.cuda(), widhtandheight.cuda(), reg_mask.cuda(), ind.cuda()

            output = model(input)
            output_hm = output[0]['hm']
            output_wh = output[0]['wh']

            loss, loss_stats = criterion(heatmaps, output_hm, widhtandheight, output_wh, reg_mask, ind)
            tracker.add_loss_stats(loss_stats)
            tracker.save_IoU_mAP(ind, reg_mask, output_hm, widhtandheight, output_wh)
            loader.desc = tracker.get_running_loss_text()
    tracker.print_avg_loss_stats()
    tracker.print_IoU_mAP_stats()
    print()
    if not args.val:
        torch.save(model.state_dict(),os.path.join(args.output, "last.pth"))

if not args.val:
    torch.save(model.state_dict(),os.path.join(args.output, "final_model.pth"))