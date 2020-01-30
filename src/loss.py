import torch

class CenterNetLoss(torch.nn.Module):
    def __init__(self, opt):
        super(CenterNetLoss, self).__init__()
        self.opt = opt
        self.crit = torch.nn.MSELoss()#FocalLoss()
        self.crit_reg = torch.nn.MSELoss()#RegL1Loss()

    def forward(self, in_hm, out_hm, in_wh, out_wh):
        hm_loss, wh_loss = 0, 0

        #TODO out_hm = sigmoid(out_hm)

        hm_loss = self.crit(in_hm, out_hm) 
        wh_loss = 0#self.crit_reg(out_wh, batch['reg_mask'], batch['ind'])

        loss = self.opt.hm_weight * hm_loss + self.opt.wh_weight * wh_loss
        
        loss_stats = {'loss': loss, 'hm_loss': hm_loss, 'wh_loss': wh_loss}
        return loss, loss_stats