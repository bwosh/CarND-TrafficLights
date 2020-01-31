import cv2
import numpy as np
import torch
from utils.transforms import _gather_feat, _transpose_and_gather_feat, decode_results

class Metrics():
    counter = 0 # TODO remove debug code

    def __init__(self, in_ind, reg_mask, out_hm, in_wh, out_wh, opts):
        self.in_ind = in_ind
        self.reg_mask = reg_mask
        self.out_hm = out_hm
        self.in_wh = in_wh
        self.out_wh = out_wh 
        self.opts = opts
        self.preprocess()

    def get_gt_image(self, inds, reg_mask, whs):
        size = self.opts.output_size
        for i in range(len(inds)):
            mask = reg_mask[i]
            if mask>0:
                ind = inds[i]
                wh = whs[i]

                cx = ind % size
                cy = (ind - cx) // size
                width, height = wh
                width, height = int(width), int(height)

                print("mask ind cx cy w h", mask, ind, cx, cy, width, height)


    def get_pred_image(self, bboxes, scores, threshold):
        print("PRED")

    def preprocess(self):
        results = decode_results(self.out_hm, self.out_wh, self.opts.K)
        self.results = results

        for idx in range(len(self.results)):
            image_result = self.results[idx]

            gt_ind = self.in_ind[idx].detach().cpu().numpy()
            gt_regmask = self.reg_mask[idx].detach().cpu().numpy()
            gt_wh = self.in_wh[idx].detach().cpu().numpy()

            bboxes = image_result['bboxes']
            scores = image_result['scores']

            img_gt = self.get_gt_image(gt_ind, gt_regmask, gt_wh)
            img_pred = self.get_pred_image(bboxes, scores, self.opts.vis_thresh)

            # TODO gt maps vs pred maps

    def calculate_APs(self):
        return [0] # TODO

    def calculate_IoUs(self):
        return [0] # TODO