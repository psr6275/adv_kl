import torch.nn as nn
import torch
__all__ = ['combine_model']

class CombineModel(nn.Module):
    def __init__(self,model_dae,model_clf,dae_type = 'recon'):
        super(CombineModel,self).__init__()
        self.model_dae = model_dae
        self.model_clf = model_clf
        assert dae_type in ['recon','denoi'], "Error! type should be in [recon, denoi]"
        self.dae_type = dae_type # another option is 'denoi'

    def forward(self,x):
        x_ = self.model_dae(x)
        if self.dae_type =='recon':
            out = self.model_clf(x_)
        else:
            out = torch.clamp(x+x_,0,1)
            out = self.model_clf(out)
        return x_, out

def combine_model(**kwargs):
    model = CombineModel(**kwargs)
    return model
