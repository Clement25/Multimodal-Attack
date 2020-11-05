import torch
import numpy as np

class FGSMAttacker(object):
    """ FGSM back propogation
    Rebuild trained model and backpropagate gradients to input data.
    Modify data using FGSM gradient ascent and dump all data into pickle file.
    Evaluate new samples under different ascending step, yield and compare the results. 
    """
    def __init__(self, epsilon):
        """Initialize an FGSM Attacker
        Args:
            epsilon (int): the step size of FGSM
        Return:
            None
        """
        self.epsilon = epsilon

    def _restore_model(self):
        """Restore the trained model
        Args: 
            None
        Returns:
            The model trained as original paper describes 
        """
        raise NotImplementedError("Model restoration not implemented!")
        ###################
        ### coding here ###
        ###################
        return model
    
    def attack(self, model):
        """Using FGSM method to attack input data
            ***core function for this class***
        Args:
            model (nn.Module): The targeted model with trained parameters
        Returns:
            adv_data (dict[torch.Tensor]): A dictionary containing several tensors which are
            corresponding to each item in original inputs.
        """
        
        raise NotImplementedError("attack function not implemented!")
        ###################
        ### coding here ###
        ###################
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