import cv2
import numpy as np
import torch
from utils.transforms import _gather_feat, _transpose_and_gather_feat, decode_results

class Metrics():
    counter = 0 # TODO remove debug code

    def __init__(self, in_hm, out_hm, in_wh, out_wh, opts):
        self.in_hm = in_hm
        self.out_hm = out_hm
        self.in_wh = in_wh
        self.out_wh = out_wh 
        self.opts = opts
        self.preprocess()

    def preprocess(self):
        results = decode_results(self.out_hm, self.out_wh, self.opts.K)
        self.results = results

        for idx in range(len(self.results)):
            image_result = self.results[idx]

            gt_hm = self.in_hm[idx].detach().cpu().numpy()[0]
            gt_wh = self.in_wh[idx].detach().cpu().numpy()

            pred_hm = self.out_hm[idx].detach().cpu().numpy()[0]
            pred_wh = self.out_wh[idx].detach().cpu().numpy()

            bboxes = image_result['bboxes']
            scores = image_result['scores']

            print("BATCH index=",idx, list(image_result.keys()))
            print("GT hm, wh", gt_hm.shape, gt_wh.shape)
            print("PRED bboxes, scores",bboxes.shape, scores.shape)

            # TODO gt maps vs pred maps

    def calculate_APs(self):
        return [0] # TODO

    def calculate_IoUs(self):
        return [0] # TODO