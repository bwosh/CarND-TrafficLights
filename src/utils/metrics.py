import cv2
import numpy as np
import torch
from utils.transforms import _gather_feat, _transpose_and_gather_feat, decode_results

class Metrics():
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
        result = np.zeros((size, size), dtype='uint8')

        for i in range(len(inds)):
            mask = reg_mask[i]
            if mask>0:
                ind = inds[i]
                wh = whs[i]

                cx = ind % size
                cy = (ind - cx) // size
                width, height = wh
                width, height = int(width), int(height)

                x1 = cx-width//2
                y1 = cy-height//2
                x2 = x1+width+1 # TODO +1/-1 ?
                y2 = y1+height+1

                result[y1:y2,x1:x2] = 1

        return result

    def get_pred_image(self, bboxes, scores, threshold):
        size = self.opts.output_size
        result = np.zeros((size, size), dtype='uint8')

        for i in range(len(bboxes)):
            if scores[i]>=threshold:
                x1,y1,x2,y2 = bboxes[i]
                x1,y1,x2,y2 = int(x1),int(y1),int(x2),int(y2)
                result[y1:y2,x1:x2] = 1
        return result

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

            intersection = np.sum(img_gt * img_pred)
            union = np.sum((img_gt+img_pred)>0)
            gt_all = np.sum(img_gt)
            pred_all = np.sum(img_pred)

            if union == 0:
                image_result['iou'] = 1
            else:
                image_result['iou'] = intersection/union

            if pred_all == 0:
                if gt_all == 0:
                    image_result['precision'] = 1
                else:
                    image_result['precision'] = 0
            else:
                image_result['precision'] = intersection / pred_all

    def calculate_APs(self):
        return [i['precision'] for i in self.results]

    def calculate_IoUs(self):
        return [i['iou'] for i in self.results]