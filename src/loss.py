import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.transforms import _gather_feat, _transpose_and_gather_feat

class RegL1Loss(nn.Module):
  def __init__(self):
    super(RegL1Loss, self).__init__()
  
  def forward(self, output, mask, ind, target):
    pred = _transpose_and_gather_feat(output, ind)
    mask = mask.unsqueeze(2).expand_as(pred).float()
    loss = F.l1_loss(pred * mask, target * mask, reduction='sum')
    loss = loss / (mask.sum() + 1e-4)
    return loss

class CenterNetLoss(torch.nn.Module):
    def __init__(self, opt):
        super(CenterNetLoss, self).__init__()
        self.opt = opt
        self.crit = nn.MSELoss()
        self.crit_reg = RegL1Loss()

    def forward(self, in_hm, out_hm, in_wh, out_wh, reg_mask, ind):
        hm_loss, wh_loss = 0, 0

        hm_loss = self.crit(in_hm, out_hm) 
        wh_loss = self.crit_reg(out_wh, reg_mask, ind, in_wh)

        loss = self.opt.hm_weight * hm_loss + self.opt.wh_weight * wh_loss
        
        loss_stats = {'loss': loss, 'hm_loss': hm_loss, 'wh_loss': wh_loss}
        return loss, loss_stats