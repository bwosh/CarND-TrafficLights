import numpy as np

from utils.metrics import Metrics

class ResultTracker:
    def __init__(self, opts):
        self.opts = opts
        self.reset()

    def reset(self):
        self.loss = {}
        self.aps = []
        self.ious = []

    def add_loss_stats(self, loss_stats):
        for k in loss_stats:
            if k in self.loss:
                self.loss[k].append(float(loss_stats[k]))
            else:
                self.loss[k] = [float(loss_stats[k])] 

    def save_IoU_mAP(self, in_ind, in_regmask, out_hm, in_wh, out_wh):
        metrics = Metrics(in_ind, in_regmask, out_hm, in_wh, out_wh, self.opts)

        self.aps += metrics.calculate_APs()
        self.ious += metrics.calculate_IoUs()
        
    def print_IoU_mAP_stats(self):
        print(f"mAP: {np.mean(self.aps):.5f}, mIoU: {np.mean(self.ious):.5f}")

    def print_avg_loss_stats(self):
        for k in self.loss:
            avg = np.mean(self.loss[k])
            print(f"{k}:{avg:.5f} ", end='')
        print()

    def get_running_loss_text(self):
        return f"{np.mean(self.loss['loss']):0.5f}"
    