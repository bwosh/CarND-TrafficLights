import numpy as np

class ResultTracker:
    def __init__(self):
        self.reset()

    def reset(self):
        self.loss = {}

    def add_loss_stats(self, loss_stats):
        for k in loss_stats:
            if k in self.loss:
                self.loss[k].append(float(loss_stats[k]))
            else:
                self.loss[k] = [float(loss_stats[k])] 

    def print_avg_loss_stats(self):
        for k in self.loss:
            avg = np.mean(self.loss[k])
            print(f"{k}:{avg:.5f} ", end='')
        print()

    def get_running_loss_text(self):
        return f"{np.mean(self.loss['loss']):0.5f}"
    