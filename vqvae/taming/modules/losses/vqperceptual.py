import torch
import torch.nn as nn
import torch.nn.functional as F
from taming.modules.losses.classification import ClassLoss

class DummyLoss(nn.Module):
    def __init__(self):
        super().__init__()


class VQWithClassification(nn.Module):
    def __init__(self, codebook_weight=1.0, pixel_weight=1.0, classification_weight=1.0):
        super().__init__()
        self.codebook_weight = codebook_weight
        self.pixel_weight = pixel_weight
        
        self.classification_weight = classification_weight
        self.classification_loss = ClassLoss()
        
        
    def forward(self, codebook_loss, inputs, reconstructions, c_hat, c_vec,
                global_step, last_layer=None, cond=None, split="train"):  
        # reconstruction error
        rec_loss = F.mse_loss(reconstructions.contiguous(), inputs.contiguous())
        rec_loss = torch.mean(rec_loss)
        
        #classification error
        cl_loss, log_dict_cl = self.classification_loss(c_hat, c_vec, split='train')
        
        #total loss
        total_loss = rec_loss + self.codebook_weight * codebook_loss.mean() + self.classification_weight * cl_loss
        #log_dict_total= {"{}/total_loss".format(split): total_loss.clone().detach().mean() }
        
        log_dict_ae = {"{}/total_loss".format(split): total_loss.clone().detach().mean(),
                "{}/quant_loss".format(split): codebook_loss.detach().mean(),
                "{}/rec_loss".format(split): rec_loss.clone().detach().mean(),
                "{}/cl_loss".format(split): cl_loss.clone().detach().mean(),
                }
        return [total_loss, rec_loss, cl_loss], log_dict_ae