import torch
import torch.nn as nn
import torch.nn.functional as F

def _gather_feat(feat, ind, mask=None):
    dim  = feat.size(2)
    ind  = ind.unsqueeze(2).expand(ind.size(0), ind.size(1), dim)
    feat = feat.gather(1, ind)
    if mask is not None:
        mask = mask.unsqueeze(2).expand_as(feat)
        feat = feat[mask]
        feat = feat.view(-1, dim)
    return feat

def _transpose_and_gather_feat(feat, ind):
    feat = feat.permute(0, 2, 3, 1).contiguous()
    feat = feat.view(feat.size(0), -1, feat.size(3))
    feat = _gather_feat(feat, ind)
    return feat

class RegL1Loss(nn.Module):
  def __init__(self):
    super(RegL1Loss, self).__init__()
  
  def forward(self, output, mask, ind, target):
    pred = _transpose_and_gather_feat(output, ind)
    mask = mask.unsqueeze(2).expand_as(pred).float()
    loss = F.l1_loss(pred * mask, target * mask, reduction='sum')
    loss = loss / (mask.sum() + 1e-4)
    return loss

def _neg_loss(pred, gt):
  ''' Modified focal loss. Exactly the same as CornerNet.
      Runs faster and costs a little bit more memory
    Arguments:
      pred (batch x c x h x w)
      gt_regr (batch x c x h x w)
  '''
  pos_inds = gt.eq(1).float()
  neg_inds = gt.lt(1).float()

  neg_weights = torch.pow(1 - gt, 4)

  loss = 0

  pos_loss = torch.log(pred) * torch.pow(1 - pred, 2) * pos_inds
  neg_loss = torch.log(1 - pred) * torch.pow(pred, 2) * neg_weights * neg_inds

  num_pos  = pos_inds.float().sum()
  pos_loss = pos_loss.sum()
  neg_loss = neg_loss.sum()

  if num_pos == 0:
    loss = loss - neg_loss
  else:
    loss = loss - (pos_loss + neg_loss) / num_pos
  return loss

class FocalLoss(nn.Module):
  '''nn.Module warpper for focal loss'''
  def __init__(self):
    super(FocalLoss, self).__init__()
    self.neg_loss = _neg_loss

  def forward(self, out, target):
    return self.neg_loss(out, target)

class CenterNetLoss(torch.nn.Module):
    def __init__(self, opt):
        super(CenterNetLoss, self).__init__()
        self.opt = opt
        self.crit = torch.nn.MSELoss()#FocalLoss()
        self.crit_reg = RegL1Loss()

    def forward(self, in_hm, out_hm, in_wh, out_wh, reg_mask, ind):
        hm_loss, wh_loss = 0, 0

        #TODO out_hm = sigmoid(out_hm)

        hm_loss = self.crit(in_hm, out_hm) 
        #print(out_wh.shape, reg_mask.shape, ind.shape, in_wh.shape)
        wh_loss = self.crit_reg(out_wh, reg_mask, ind, in_wh)

        loss = self.opt.hm_weight * hm_loss + self.opt.wh_weight * wh_loss
        
        loss_stats = {'loss': loss, 'hm_loss': hm_loss, 'wh_loss': wh_loss}
        return loss, loss_stats