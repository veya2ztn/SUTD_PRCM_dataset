import torch.nn as nn
import numpy as np
import torch

import torch.nn.functional as F
def mean_stratage(loss,reduction):
    if reduction is None:return loss.mean()
    if reduction == 'per_batch':
        mean_channel  = np.arange(1,len(loss.shape)).tolist()
        return loss.mean(mean_channel)

class MyLoss(nn.Module):
    def __init__(self,reduction = None,weight=None,mode=None):
        super().__init__()
        self.reduction = reduction # per_batch
        self.weight = weight
        self.mode   = mode

class FocalLoss(MyLoss):
    def __init__(self, gamma=2, alpha=0.25,with_logits=False,reduction = None):
        super().__init__()
        self._gamma = gamma
        self._alpha = alpha
        self.reduction = reduction
        self.with_logits =  with_logits
    def forward(self, y_true, y_pred):
        if self.with_logits:
            cross_entropy_loss  = torch.nn.BCEWithLogitsLoss()(y_pred,y_true)
        else:
            cross_entropy_loss  = torch.nn.BCELoss()(y_pred,y_true)
        p_t                 = ((y_true * y_pred) + ((1 - y_true) * (1 - y_pred)))
        modulating_factor   = torch.pow(1.0 - p_t, self._gamma) if self._gamma else 1.0
        alpha_weight_factor = (y_true * self._alpha + (1 - y_true) * (1 - self._alpha)) if self._alpha is not None else 1.0
        focal_cross_entropy_loss = (modulating_factor * alpha_weight_factor *cross_entropy_loss)
        return mean_stratage(focal_cross_entropy_loss,self.reduction)



class FocalLossWrapper:
    def __init__(self,gamma=2, alpha=0.25):
        self._gamma = gamma
        self._alpha = alpha
    def __call__(self,reduction = None):
        return FocalLoss(gamma=self._gamma, alpha=self._alpha,reduction = reduction)

class EnhanceLoss1:
    def _loss(self,x,target):
        loss = (x - target)**2
        loss = loss.exp()-1
        loss = loss.mean()
        return loss

class EnhanceLoss2:
    def _loss(self,x,target):
        loss = (x - target)**2
        loss = loss*target
        loss = loss.mean()
        return loss

class EnhanceLoss3:
    def _loss(self,x,target):
        loss = (x - target)**2
        loss = loss*target*target
        loss = loss.mean()
        return loss

class TandemLossL2(MyLoss):
    def __init__(self, gamma=0):
        super().__init__()
        self._gamma = gamma
    def forward(self, Feat_Real,Feat_Pred,Imag_Real,Imag_Pred):
        curve_loss = torch.nn.MSELoss()(Feat_Pred,Feat_Real)
        image_loss1 = torch.nn.MSELoss()(Imag_Pred,Imag_Real)
        image_loss2 = torch.nn.BCELoss()(Imag_Pred,Imag_Real)
        image_accu = 1-(torch.round(Imag_Real)==torch.round(Imag_Pred)).float().mean()
        #loss_image = 1/((Imag_Pred-0.5+1e-6)**2).mean()
        loss       = (1-self._gamma)*curve_loss+self._gamma/2*image_loss1+self._gamma/2*image_loss2
        accu       = image_accu

        return loss,accu

class BinaryBingo_logits(MyLoss):
    def forward(self, predict,target):
        predict = torch.sigmoid(predict)
        predict = torch.round(predict)
        target  = torch.round(target.float())
        loss_per_batch    = 1-(predict==target).float()
        return mean_stratage(loss_per_batch,self.reduction)
class BinaryPositive_logits(MyLoss):
    def forward(self, predict,target):
        predict           = torch.sigmoid(predict)
        mean_channel      = np.arange(1,len(target.shape)).tolist()
        predict           = torch.round(predict)
        target            = torch.round(target.float())
        bingo             = ((predict==target)*(target==1)).sum(mean_channel,keepdim=True).float()
        should            = target.sum(mean_channel,keepdim=True).float()+0.0001
        loss_per_batch    = 1-bingo/should
        return mean_stratage(loss_per_batch,self.reduction)
class BinaryNegative_logits(MyLoss):
    def forward(self, predict,target):
        predict = torch.sigmoid(predict)
        mean_channel      = np.arange(1,len(target.shape)).tolist()
        predict           = torch.round(predict)
        target            = torch.round(target.float())
        bingo             = ((predict==target)*(target==0)).sum(mean_channel,keepdim=True).float()
        should            = (1-target).sum(mean_channel,keepdim=True).float()+0.0001
        loss_per_batch    = 1-bingo/should
        return mean_stratage(loss_per_batch,self.reduction)
class BinaryBingo(MyLoss):
    def forward(self, predict,target):
        predict = torch.round(predict)
        target  = torch.round(target)
        loss_per_batch    = 1-(predict==target).float()
        return mean_stratage(loss_per_batch,self.reduction)
class BinaryPositive(MyLoss):
    def forward(self, predict,target):

        mean_channel  = np.arange(1,len(target.shape)).tolist()
        predict           = torch.round(predict)
        target            = torch.round(target)
        active            = (target==1)
        bingo             = ((predict==target)*active).sum(mean_channel,keepdim=True).float()
        should            = target.sum(mean_channel,keepdim=True).float()+0.01
        loss_per_batch    = 1-bingo/should
        return mean_stratage(loss_per_batch,self.reduction)
class BinaryNegative(MyLoss):
    def forward(self, predict,target):
        mean_channel      = np.arange(1,len(target.shape)).tolist()
        predict           = torch.round(predict)
        target            = torch.round(target)
        active            = (target==0)
        bingo             = ((predict==target)*active).sum(mean_channel,keepdim=True).float()
        should            = (1-target).sum(mean_channel,keepdim=True).float()+0.01
        loss_per_batch    = 1-bingo/should
        return mean_stratage(loss_per_batch,self.reduction)
class OnehotBingo(MyLoss):
    def forward(self, pred,real):
        pred = pred.max(-1)[-1] # the position
        loss = 1-(real==pred).float().mean()
        return mean_stratage(loss,self.reduction)
class OnehotPositive(MyLoss):
    def forward(self, pred,real):
        pred = pred.max(-1)[-1] # the position
        loss = 1-(real==pred)[real==1].float().mean()
        return mean_stratage(loss,self.reduction)
class OnehotNegative(MyLoss):
    def forward(self, pred,real):
        pred = pred.max(-1)[-1] # the position
        loss = 1-(real==pred)[real==0].float().mean()
        return mean_stratage(loss,self.reduction)


class ClassifierAccuAll(MyLoss):
    def forward(self,predict,target):
        assert len(predict.shape) >= 2
        if predict.shape[-1] >= 2 and len(target.shape)==1:
            # one-hot
            predict = predict.max(-1)[-1] # the position
            loss = 1-(target.long()==predict).float().mean()
            return mean_stratage(loss,self.reduction)
        assert len(target.shape) == len(predict.shape)
        assert target.shape[-1]  == predict.shape[-1]
        # both for multi-label and single-label
        if self.mode == "logits":predict = torch.sigmoid(predict)
        mean_channel      = np.arange(1,len(target.shape)).tolist()
        predict           = torch.round(predict)
        target            = torch.round(target)
        loss_per_batch    = 1-(predict==target).float()
        return mean_stratage(loss_per_batch,self.reduction)
class ClassifierAccuPos(MyLoss):
    def forward(self,predict,target):
        assert len(predict.shape) >= 2
        if predict.shape[-1] >= 2 and len(target.shape)==1:
            predict = predict.max(-1)[-1] # the position
            loss = 1-(target.long()==predict)[target==1].float().mean()
            return mean_stratage(loss,self.reduction)
        assert len(target.shape) == len(predict.shape)
        assert target.shape[-1]  == predict.shape[-1]
        if self.mode == "logits":predict = torch.sigmoid(predict)
        mean_channel  = np.arange(1,len(target.shape)).tolist()
        predict           = torch.round(predict)
        target            = torch.round(target)
        active            = (target==1)
        bingo             = ((predict==target)*active).sum(mean_channel,keepdim=True).float()
        should            = active.sum(mean_channel,keepdim=True).float()+0.01
        loss_per_batch    = 1-bingo/should
        return mean_stratage(loss_per_batch,self.reduction)
class ClassifierAccuNeg(MyLoss):
    def forward(self,predict,target):
        assert len(predict.shape) >= 2
        if predict.shape[-1] >= 2 and len(target.shape)==1:
            predict = predict.max(-1)[-1] # the position
            loss = 1-(target.long()==predict)[target==0].float().mean()
            return mean_stratage(loss,self.reduction)
        assert len(target.shape) == len(predict.shape)
        assert target.shape[-1]  == predict.shape[-1]
        if self.mode == "logits":predict = torch.sigmoid(predict)
        mean_channel  = np.arange(1,len(target.shape)).tolist()
        predict           = torch.round(predict)
        target            = torch.round(target)
        active            = (target==0)
        bingo             = ((predict==target)*active).sum(mean_channel,keepdim=True).float()
        should            = active.sum(mean_channel,keepdim=True).float()+0.01
        loss_per_batch    = 1-bingo/should
        return mean_stratage(loss_per_batch,self.reduction)


class MAELoss(MyLoss):
    def forward(self, x,target):
        loss = (x - target).abs()
        return mean_stratage(loss,self.reduction)
class MSELoss(MyLoss):
    def forward(self, x,target):
        loss = (x - target)**2
        return mean_stratage(loss,self.reduction)
class CrossEntropyLoss(MyLoss):
    def forward(self, x,target):
        if len(x.shape) != len(target.shape):
            x = x.squeeze()
        loss = torch.nn.CrossEntropyLoss(reduction='none')(x,target.long())
        return mean_stratage(loss,self.reduction)
class BCEWithLogitsLoss(MyLoss):
    def forward(self, x,target):
        loss = torch.nn.BCEWithLogitsLoss(reduction='none')(x,target.float())
        return mean_stratage(loss,self.reduction)
class BCEloss(MyLoss):
    def forward(self, x,target):
        loss = torch.nn.BCELoss(reduction='none')(x,target)
        return mean_stratage(loss,self.reduction)
class BalanceBCE(MyLoss):
    def __init__(self,weight_zero,reduction = None):
        super().__init__()
        self.weight = weight_zero
    def forward(self, x,target):
        if x.is_cuda and not self.weight.is_cuda:self.weight=self.weight.cuda()
        coef = target*(1-self.weight) + (1-target)*self.weight
        loss = F.binary_cross_entropy(x,target,reduction='none')*coef
        return mean_stratage(loss,self.reduction)
class BalancedBCEWrapper:
    def __init__(self,weight_zero):
        self.weight_zero= weight_zero
    def __call__(self,**kargs):
        return BalanceBCE(self.weight_zero,**kargs)
eps=1e-5

class SelfAttentionedLoss(MyLoss):
    def forward(self, x,target):
        loss = (x - target)**2
        loss = loss*target
        return mean_stratage(loss,self.reduction)
class SelfEnhanceLoss1(MyLoss):
    def forward(self, x,target):
        loss = (x - target)**2
        loss = loss*torch.pow(3,target)
        return mean_stratage(loss,self.reduction)
class SelfEnhanceLoss2(MyLoss):
    def forward(self, x,target):
        loss1 = ((x - target)**2).mean()
        loss2 = x.sum()-target.sum()
        return loss1+loss2
class SelfEnhanceLoss3(MyLoss):
    def forward(self, x,target):
        loss1 = ((x - target)**2)
        loss2 = (x/(x.sum()+eps)-target/(target.sum()+eps))
        return mean_stratage(loss1+loss2,self.reduction)
class SelfEnhanceLoss4(MyLoss):
    def forward(self, x,target):
        loss1 = (x - target)**2
        loss2 = x/(x.sum((1,2),keepdim=True)+eps)-target/(target.sum((1,2),keepdim=True)+eps)
        return mean_stratage(loss1+loss2,self.reduction)
class SelfEnhanceLoss4Tend(MyLoss):
    def forward(self, x,target):
        loss1 = ((x - target)**2)
        if x.is_cuda and not self.weight.is_cuda:self.weight=self.weight.cuda()
        x = x + self.weight
        target = target + self.weight
        loss2 = (x/(x.sum((1,2),keepdim=True)+eps)-target/(target.sum((1,2),keepdim=True)+eps))
        return mean_stratage(loss1+loss2,self.reduction)
class SelfEnhanceLoss5(MyLoss):
    def forward(self, x,target):
        loss1 = (x - target)**2
        loss2 = (x/(x.sum((1,2),keepdim=True)+eps)-target/(target.sum((1,2),keepdim=True)+eps))**2
        return mean_stratage(loss1+loss2,self.reduction)
class SelfEnhanceLoss6(MyLoss):
    def forward(self, x,target):
        loss1 = (x - target)**2
        loss2 = (x/(x.sum()+eps)-target/(target.sum()+eps))**2
        return mean_stratage(loss1+loss2,self.reduction)
class SelfEnhanceLoss3Tend(MyLoss):
    def forward(self, x,target):
        loss1 = ((x - target)**2)
        if x.is_cuda and not self.weight.is_cuda:self.weight=self.weight.cuda()
        x = x + self.weight
        target = target + self.weight
        loss2 = (x/(x.sum()+eps)-target/(target.sum()+eps))
        return mean_stratage(loss1+loss2,self.reduction)

class SEALoss4TWrapper:
    def __init__(self,weight):
        self.weight= weight
    def __call__(self,**kargs):
        return SelfEnhanceLoss4Tend(weight=self.weight,**kargs)
class SEALoss3TWrapper:
    def __init__(self,weight):
        self.weight= weight
    def __call__(self,**kargs):
        return SelfEnhanceLoss3Tend(weight=self.weight,**kargs)

from mltool.pytorch_ssim import SSIM
class SSIMError(MyLoss):
    def forward(self, x,target):
        loss = 1-SSIM(size_average=False)(x,target)
        return mean_stratage(loss,self.reduction)
class BinaryGood(MyLoss):
    def forward(self,image,target=None):
        loss_per_batch = (1/((image-0.5)**2+1e-5))
        return mean_stratage(loss_per_batch,self.reduction)
class ImageEntropy(MyLoss):
    def forward(self,image,target=None):
        mean_channel  = np.arange(1,len(image.shape)).tolist()
        loss_per_batch = (1/(image.var(mean_channel,keepdim=True)+1e-5))
        return mean_stratage(loss_per_batch,self.reduction)
class VaryingLoss(MyLoss):
    '''
    use to avoid mode colleps
    '''
    def forward(self, tensor,target=None):
        repeat=10
        loss = 0
        for _ in range(repeat):
            loss+=torch.nn.MSELoss()(tensor[torch.randperm(len(tensor))],tensor[torch.randperm(len(tensor))])
        loss = loss/repeat
        loss = (loss-0.5)**2
        return loss
class MeanLoss(MyLoss):
    def forward(self, tensor,target=None):
        return ((tensor.mean(0)-0.5)**2).mean()
class Kurtosis(MyLoss):
    def forward(self, data,target=None):
        mean = data.mean(0)
        var  = data.var(0)
        ku   = ((data - mean) ** 4).mean(0) / (var**2+0.01) #计算峰度
        return ku

class TandemLoss(MyLoss):
    def __init__(self, gamma=0):
        super().__init__()
        self._gamma = gamma
    def forward(self, Feat_Real,Feat_Pred,Imag_Real,Imag_Pred):
        curve_loss = torch.nn.MSELoss()(Feat_Pred,Feat_Real)
        image_loss = torch.nn.MSELoss()(Imag_Pred,Imag_Real)
        image_accu = 1-(torch.round(Imag_Real)==torch.round(Imag_Pred)).float().mean()
        loss       = (1-self._gamma)*curve_loss+self._gamma*image_loss
        accu       = image_accu
        return loss,accu

class TandemLossL1(MyLoss):
    def __init__(self, gamma=0):
        super().__init__()
        self._gamma = gamma
    def forward(self, Feat_Real,Feat_Pred,Imag_Real,Imag_Pred):
        curve_loss   = torch.nn.MSELoss()(Feat_Pred,Feat_Real)
        image_loss   = torch.nn.MSELoss()(Imag_Pred,Imag_Real)
        image_binary = BinaryGood()(Imag_Pred,Imag_Real)
        loss         = (1-self._gamma)*curve_loss+self._gamma*image_loss+self._gamma*0.1*image_binary
        accu         = image_binary

        return loss,accu
class TandemLossL2(MyLoss):
    def __init__(self, gamma=0):
        super().__init__()
        self._gamma = gamma
    def forward(self, Feat_Real,Feat_Pred,Imag_Real,Imag_Pred):
        curve_loss   = torch.nn.MSELoss()(Feat_Pred,Feat_Real)
        #image_mseloss= torch.nn.MSELoss()(Imag_Pred,Imag_Real)
        image_mseloss= 0
        image_binary = BinaryGood()(Imag_Pred,Imag_Real)
        image_entropy= ImageEntropy()(Imag_Pred,Imag_Real)
        image_loss   = image_mseloss + 2*image_binary + 0.01*image_entropy
        loss         = (1-self._gamma)*curve_loss+self._gamma*image_loss
        accu         = image_binary

        return loss,accu
from mltool.pytorch_ssim import ssim

class TandemLossSSIM(MyLoss):
    def __init__(self, gamma=0):
        super().__init__()
        self._gamma = gamma
    def forward(self, Feat_Real,Feat_Pred,Imag_Real,Imag_Pred):
        curve_loss = torch.nn.MSELoss()(Feat_Pred,Feat_Real)
        image_loss = 1-ssim(Imag_Pred,Imag_Real)
        image_binary = BinaryGood()(Imag_Pred,Imag_Real)
        loss       = (1-self._gamma)*curve_loss+self._gamma*image_loss+0.1*image_binary
        accu       = image_binary
        return loss,accu

class ComplexMSELoss(MyLoss):
    "assume [...,2] as complex input"
    def forward(self, x,target):
        loss = (x-target).norm(dim=-1)
        return mean_stratage(loss,self.reduction)
class NormMSELoss(MyLoss):
    "assume [...,2] as complex input"
    def forward(self, x,target):
        loss = (x.norm(dim=-1)-target.norm(dim=-1))**2
        return mean_stratage(loss,self.reduction)


loss_functions={}
loss_functions['MAError']        = MAELoss(reduction = 'per_batch')
loss_functions['MsaError']       = SelfAttentionedLoss(reduction = 'per_batch')
loss_functions['MSReduce']       = SelfEnhanceLoss5(reduction = 'per_batch')
loss_functions['MseaError']      = SelfEnhanceLoss1(reduction = 'per_batch')
loss_functions['MseaError']      = SelfEnhanceLoss1(reduction = 'per_batch')
loss_functions['CELoss']         = CrossEntropyLoss(reduction = 'per_batch')
loss_functions['BCELoss']        = BCEloss(reduction = 'per_batch')
loss_functions['MSError']        = MSELoss(reduction = 'per_batch')
loss_functions['BinaryImgError'] = BinaryBingo(reduction = 'per_batch')
loss_functions['BinarySSIM']     = SSIMError(reduction = 'per_batch')
loss_functions['BinaryGood']     = BinaryGood(reduction = 'per_batch')
loss_functions['BinaryEntropy']  = ImageEntropy(reduction = 'per_batch')
loss_functions['BinaryEntropy']  = ImageEntropy(reduction = 'per_batch')
loss_functions['ComplexMSE']     = ComplexMSELoss(reduction = 'per_batch')
loss_functions['NormMSE']        = NormMSELoss(reduction = 'per_batch')
loss_functions['FocalLoss']      = FocalLoss(reduction = 'per_batch')

loss_functions['OneHotError']    = OnehotBingo(reduction = 'per_batch')
loss_functions['OneHotP']        = OnehotPositive(reduction = 'per_batch')
loss_functions['OneHotN']        = OnehotNegative(reduction = 'per_batch')
loss_functions['BinaryP']        = BinaryPositive(reduction = 'per_batch')
loss_functions['BinaryN']        = BinaryNegative(reduction = 'per_batch')
loss_functions['BinaryA']        = BinaryBingo(reduction = 'per_batch')
loss_functions['BinaryPL']       = BinaryPositive_logits(reduction = 'per_batch')
loss_functions['BinaryNL']       = BinaryNegative_logits(reduction = 'per_batch')
loss_functions['BinaryAL']       = BinaryBingo_logits(reduction = 'per_batch')

loss_functions['ClassifierA']    = ClassifierAccuAll(reduction = 'per_batch',mode='logits')
loss_functions['ClassifierP']    = ClassifierAccuPos(reduction = 'per_batch',mode='logits')
loss_functions['ClassifierN']    = ClassifierAccuNeg(reduction = 'per_batch',mode='logits')

loss_functions['ClassifierA_nologits']    = ClassifierAccuAll(reduction = 'per_batch')
loss_functions['ClassifierP_nologits']    = ClassifierAccuPos(reduction = 'per_batch')
loss_functions['ClassifierN_nologits']    = ClassifierAccuNeg(reduction = 'per_batch')
