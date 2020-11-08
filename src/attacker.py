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

    def _load_data(self):
        """Load primitive data from disk 
        """
        data_loader = get_loader(self.config, shuffle=False)
        return data_loader
    
    def get_diff_loss(self):

        shared_t = self.model.utt_shared_t
        shared_v = self.model.utt_shared_v
        shared_a = self.model.utt_shared_a
        private_t = self.model.utt_private_t
        private_v = self.model.utt_private_v
        private_a = self.model.utt_private_a

        # Between private and shared
        loss = self.loss_diff(private_t, shared_t)
        loss += self.loss_diff(private_v, shared_v)
        loss += self.loss_diff(private_a, shared_a)

        # Across privates
        loss += self.loss_diff(private_a, private_t)
        loss += self.loss_diff(private_a, private_v)
        loss += self.loss_diff(private_t, private_v)

        return loss

    def get_domain_loss(self):

        if self.train_config.use_cmd_sim:
            return 0.0
        
        # Predicted domain labels
        domain_pred_t = self.model.domain_label_t
        domain_pred_v = self.model.domain_label_v
        domain_pred_a = self.model.domain_label_a

        # True domain labels
        domain_true_t = self._to_gpu(torch.LongTensor([0]*domain_pred_t.size(0)))
        domain_true_v = self._to_gpu(torch.LongTensor([1]*domain_pred_v.size(0)))
        domain_true_a = self._to_gpu(torch.LongTensor([2]*domain_pred_a.size(0)))

        # Stack up predictions and true labels
        domain_pred = torch.cat((domain_pred_t, domain_pred_v, domain_pred_a), dim=0)
        domain_true = torch.cat((domain_true_t, domain_true_v, domain_true_a), dim=0)

        return self.domain_loss_criterion(domain_pred, domain_true)
    
    def get_recon_loss(self, ):

        loss = self.loss_recon(self.model.utt_t_recon, self.model.utt_t_orig)
        loss += self.loss_recon(self.model.utt_v_recon, self.model.utt_v_orig)
        loss += self.loss_recon(self.model.utt_a_recon, self.model.utt_a_orig)
        loss = loss/3.0
        return loss

    def get_cmd_loss(self):

        if not self.train_config.use_cmd_sim:
            return 0.0

        # losses between shared states
        loss = self.loss_cmd(self.model.utt_shared_t, self.model.utt_shared_v, 5)
        loss += self.loss_cmd(self.model.utt_shared_t, self.model.utt_shared_a, 5)
        loss += self.loss_cmd(self.model.utt_shared_a, self.model.utt_shared_v, 5)
        loss = loss/3.0
        return loss

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
        self.model = model
        model.load_state_dict(torch.load(ckpt_path))
        data_loader = self._load_data()

        if self.config.data == "ur_funny":
            criterion = nn.MSELoss()
        elif self.config.data.lower() in ["mosi", "mosei"]:
            criterion = nn.MSELoss() 

        # Attack data batch by batch
        for batch in data_loader:
            model.zero_grad()
            t, v, a, y, l, bert_sent, bert_sent_type, bert_sent_mask = batch
            batch_size = t.size(0)
            t, v, a, y, l, bert_sent, bert_sent_type, bert_sent_mask = self._to_gpu(t, v, a, y, z, bert_sent, bert_sent_type, bert_sent_mask)
            self._add_grad_reqr(t, v, a)
        
            y_tilde = model(t, v, a, l, bert_sent, bert_sent_type, bert_sent_mask)

            if self.config.data == "ur_funny":
                y = y.squeezea
            
            cls_loss = criterion(y_tilde, y)
            diff_loss = self.get_diff_loss()
            domain_loss = self.get_domain_loss()
            recon_loss = self.get_recon_loss()
            cmd_loss = self.get_cmd_loss()
            # line 127

            if self.config.use_cmd_sim:
                similarity_loss = cmd_loss
            else:
                similarity_loss = domain_loss
            
            loss = cls_loss + \
                self.config.diff_weight * diff_loss + \
                self.config.sim_weight * similarity_loss + \
                self.config.recon_weight * recon_loss
            
            loss.backward()

            # perform the attack
            # TODO: Now we didn't store attacked embeddings back (maybe too large to store). Instead we continuously change the weights of embedding. Considering adding all of these.
            embed_weight = self.model.embed.weight
            embed_weight += self.epsilon * embed_weight.grad
            v_adv += self.epsilon * v.grad
            a_adv += self.epsilon * a.grad

        return v, a
    
    def save_data(self, data):
        """Store adversarial data back to disk
        Args:
            data: Adversarial data in the readable format
        Returns:
            None
        """
        raise NotImplementedError("Save data not implemented!")
        ###################q
        ### coding here ###
        ###################
    
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
