import torch

class CenterNetLoss(torch.nn.Module):
    def __init__(self, opt):
        super(CenterNetLoss, self).__init__()
        self.opt = opt

        self.dummy = torch.nn.MSELoss()

    def forward(self, in_hm, out_hm, in_wh, out_wh):
        hm_loss, wh_loss = 0, 0

        # TODO dummy
        hm_loss = self.dummy(in_hm, out_hm) 
        wh_loss = self.dummy(in_wh, out_wh)

        loss = self.opt.hm_weight * hm_loss + self.opt.wh_weight * wh_loss
        
        loss_stats = {'loss': loss, 'hm_loss': hm_loss, 'wh_loss': wh_loss}
        return loss, loss_stats