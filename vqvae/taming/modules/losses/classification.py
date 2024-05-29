import torch.nn as nn
    
class ClassLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.loss = nn.CrossEntropyLoss()

    def forward(self, chat, c, split='train'):  
        loss_val = self.loss(chat, c)
        log = {"{}/clloss".format(split): loss_val.clone().detach().mean() }
        return loss_val, log
