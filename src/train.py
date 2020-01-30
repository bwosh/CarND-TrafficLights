import os
import torch

from models.dla import get_pose_net
from opts import get_args

args = get_args()

# Network model preparation
model = get_pose_net(34, heads={'hm': 1, 'wh': 2}, head_conv=-1)

# TODO remove dummy code
dummy = torch.zeros((1,3,args.input_size,args.input_size), dtype=torch.float)
result = model(dummy)

for eid, e in enumerate(result):
    for xid,x in enumerate(e):
        print(eid, xid, x, e[x].shape)

# Training
optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

for epoch in range(args.epochs):
    print(f"*** TRAIN, epoch {epoch+1}")
    model.train()

    print(f"*** VALIDATION, epoch {epoch+1}")
    model.eval()
    with torch.no_grad():
        pass

    # TODO LOSS, trainer, validation(mAP iou mAP@class AP 50 75 s m l)
    # TODO readme: carla, pytorch->quantization->onnx->tf 1.4

torch.save(model.state_dict(),os.path.join(args.output, "final_model.pth"))