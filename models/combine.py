import torch.nn as nn
import torch
__all__ = ['combine_model']

class CombineModel(nn.Module):
    def __init__(self,model_dae,model_clf,type = 'recon'):
        super(COBINE_MODEL,self).__init__()
        self.model_dae = model_dae
        self.model_clf = model_clf
        assert type in ['recon','denoi'], "Error! type should be in [recon, denoi]"
        self.type = type # another option is 'denoi'

    def forward(self,x):
        x_ = self.model_dae(x)
        if type =='recon':
            out = self.model_clf(x_)
        else:
            out = torch.clamp(x+x_,0,1)
            out = self.model_clf(out)
        return x_, out

def combine_model(**kwargs):

    return CombineModel(**kwargs)
