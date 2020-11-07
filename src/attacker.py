import torch
import numpy as np
import torch.nn as nn

from data_loader import get_loader
from models import MISA
from config import get_config

class FGSMAttacker(object):
    """ FGSM back propogation
    Rebuild trained model and backpropagate gradients to input data.
    Modify data using FGSM gradient ascent and dump all data into pickle file.
    Evaluate new samples under different ascending step, yield and compare the results. 
    """
    def __init__(self, epsilon, config, device='cuda'):
        """Initialize an FGSM Attacker
        Args:
            epsilon (int): the step size of FGSM
            config (dict): the configuration dictionary of the model
        Return:
            None
        """
        self.epsilon = epsilon
        self.config = config
        self.device = device
    
    def _to_gpu(self, *args):
        return x.to(self.device) for x in args 
    
    def _add_grad_reqr(self, *args):
        for x in args:
            x.requires_grad = True
        return args

    def _load_data(self):
        """Load primitive data from disk 
        """
        data_loader = get_loader(self.config, shuffle=False)
        return data_loader
    
    def attack(self, model, ckpt_path='./checkpoints/best.std'):
        """Using FGSM method to attack input data
            ***core function for this class***
        Args:
            model (nn.Module): The targeted model with trained parameters
            ckpt_path (str): The relative path to store the checkpoint file.
        Returns:
            adv_data (dict[torch.Tensor]): A dictionary containing several tensors which are
            corresponding to each item in original inputs.
        """
        # load model and data
        model = MISA(self.config)
        model.load_state_dict(torch.load(ckpt_path))
        data_loader = self._load_data()

        if self.config.data == "ur_funny":
            criterion = nn.MSELoss()
        elif self.config.data.lower() in ["mosi", "mosei"]:
            criterion = nn.MSELoss() 

        # Attack data batch by batch
        for data in data_loader:
            model.zero_grad()
            t, v, a, y, l, bert_sent, bert_sent_type, bert_sent_mask = batch
            batch_size = t.size(0)
            t, v, a, y, l, bert_sent, bert_sent_type, bert_sent_mask = self._to_gpu(t, v, a, y, l, bert_sent, bert_sent_type, bert_sent_mask)
            t, v, a, y, l, bert_sent, bert_sent_type, bert_sent_mask = self._add_grad_reqr(t, v, a, y, l, bert_sent, bert_sent_type, bert_sent_mask)
        
            y_tilde = model(t, v, a, l, bert_sent, bert_sent_type, bert_sent_mask)

            if self.config.data == "ur_funny":
                y = y.squeeze
            
            cls_loss = criterion(y_tilde, y)
            ## line 119

        return adv_data
    
    def save_data(self, data):
        """Store adversarial data back to disk
        Args:
            data: Adversarial data in the readable format
        Returns:
            None
        """
        raise NotImplementedError("Save data not implemented!")
        ###################
        ### coding here ###
        ###################
        return adv_data
    
    def evaluate(self, data, model):
        """Evaluate model performance with manufactured data
        Args:
            data: The same as above
            model: The same as above
        """
        raise NotImplementedError("Evaluation not implemented!")
        ###################
        ### coding here ###
        ###################
        return None

if __name__ == '__main__':
    # Use test data
    config = get_config(mode='test')
    Attacker = FGSMAttacker(epsilon=1e-2, config)
    Attacker.attack()
