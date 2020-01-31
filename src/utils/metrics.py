class Metrics():
    def __init__(self, in_hm, out_hm, in_wh, out_wh, reg_mask, ind):
        self.in_hm = in_hm
        self.out_hm = out_hm
        self.in_wh = in_wh
        self.out_wh = out_wh 
        self.reg_mask = reg_mask
        self.ind = ind

    def calculate_APs(self):
        return [0] # TODO

    def calculate_IoUs(self):
        return [0] # TODO