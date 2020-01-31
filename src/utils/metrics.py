import cv2
import numpy as np
import torch
from utils.transforms import _gather_feat, _transpose_and_gather_feat, ctdet_decode

class Metrics():
    counter = 0 # TODO remove debug code

    def __init__(self, in_hm, out_hm, in_wh, out_wh, reg_mask, ind, debug=False):
        self.in_hm = in_hm
        self.out_hm = out_hm
        self.in_wh = in_wh
        self.out_wh = out_wh 
        self.reg_mask = reg_mask
        self.ind = ind
        self.debug = debug

        self.preprocess()

    def preprocess(self):
        #print("self.in_hm:", self.in_hm.shape, self.in_hm.dtype)
        #print("self.out_hm:", self.out_hm.shape)
        #print("self.in_wh:", self.in_wh.shape)
        #print("self.out_wh:", self.out_wh.shape)
        #print("self.reg_mask:", self.reg_mask.shape)
        #print("self.ind:", self.ind.shape)
        #pred = _transpose_and_gather_feat(self.out_wh, self.ind)
        #mask = self.reg_mask.unsqueeze(2).expand_as(pred).float()
        k = 10

        dets = ctdet_decode(self.out_hm, self.out_wh, K=k)
        dets = dets.detach().cpu().numpy()
        print(dets.shape) # batch x K x 6

        # TODO remove debug code
        if self.debug:
            gt = self.in_hm.detach().cpu().numpy()
            gt = gt * 255
            gt = gt.astype('uint8')

            img = torch.clamp(self.out_hm,0.,1.)
            img = img * 255
            img = img.detach().cpu().numpy()
            img = img.astype('uint8')

            for idx in range(len(img)):
                i = img[idx][0]
                g = gt[idx][0]
                print("---")
                for ki in range(k):
                    det = dets[idx,ki,:]
                    x1,y1,x2,y2,score,_ = det
                    x1,y1,x2,y2 = int(x1),int(y1),int(x2),int(y2)
                    print(score, end=' ')

                    cv2.rectangle(i, (x1,y1),(x2,y2),(255,255,255))

                cv2.imwrite(f"../temp/{Metrics.counter}.png",i)
                cv2.imwrite(f"../temp/{Metrics.counter}gt.png",g)
                Metrics.counter += 1

    def calculate_APs(self):
        return [0] # TODO

    def calculate_IoUs(self):
        return [0] # TODO