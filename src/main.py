from dla import get_pose_net
import torch

lr = 5e-04
epochs = 140
lr_epochs = [90,120]
lr_gamma = 0.1
input_size = 256 #512
output_size = 64 #128
num_classes=1

# Network model preparation
heads = {'hm': num_classes, 
         'wh': 2 }
net = get_pose_net(34, heads=heads, head_conv=-1)
dummy = torch.zeros((1,3,input_size,input_size), dtype=torch.float)
result = net(dummy)

for eid, e in enumerate(result):
    for xid,x in enumerate(e):
        print(eid, xid, x, e[x].shape)

# Training
optimizer = torch.optim.Adam(net.parameters(), lr=lr)
# TODO dataset, LOSS, trainer, validation(mAP iou mAP@class AP 50 75 s m l)
# TODO readme: carla, pytorch->quantization->onnx->tf 1.4
torch.save(net.state_dict(),"x.pth")