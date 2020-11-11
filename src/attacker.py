import torch
import numpy as np
import torch.nn as nn
import pickle

from data_loader import get_loader
from models import MISA
from config import get_config
from sklearn.metrics import classification_report, accuracy_score, f1_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_recall_fscore_support
from random import random

from utils import to_gpu, time_desc_decorator, DiffLoss, MSE, SIMSE, CMD

class BaseAttacker(object):
    """ Base attacker, the base class of any other attackers
    Define basic member variables. Methods regarding to the propogation of target model are included.
    Leave 3 function placeholders for the development of successors.
    """
    def __init__(self, config, device='cuda'):
        """Initialize an FGSM Attacker
        Args:
            epsilon (int): the step size of FGSM
            config (dict): configurations of the model and dataset
            attconfig (dict): configurations of the attacker
            device (str): place to store tensors and do computation
        Return:
            None
        """
        self.config = config
        self.device = device
    
        if self.config.data == "ur_funny":
            self.criterion = nn.CrossEntropyLoss(reduction="mean")
        elif self.config.data.lower() in ["mosi", "mosei"]:
            self.criterion = nn.MSELoss() 

    def _to_gpu(self, *args):
        return [x.to(self.device) for x in args]

    
    def _add_grad_reqr(self, *args):
        for x in args:
            x.requires_grad = True

    def _load_data(self):
        """Load primitive data from disk 
        """
        data_loader = get_loader(self.config, shuffle=False)
        return data_loader
    
    def get_diff_loss(self, model, loss_diff):

        shared_t = model.utt_shared_t
        shared_v = model.utt_shared_v
        shared_a = model.utt_shared_a
        private_t = model.utt_private_t
        private_v = model.utt_private_v
        private_a = model.utt_private_a

        # Between private and shared
        loss = loss_diff(private_t, shared_t)
        loss += loss_diff(private_v, shared_v)
        loss += loss_diff(private_a, shared_a)

        # Across privates
        loss += loss_diff(private_a, private_t)
        loss += loss_diff(private_a, private_v)
        loss += loss_diff(private_t, private_v)

        return loss

    def get_domain_loss(self, model, domain_loss_criterion):

        if self.config.use_cmd_sim:
            return 0.0
        
        # Predicted domain labels
        domain_pred_t = model.domain_label_t
        domain_pred_v = model.domain_label_v
        domain_pred_a = model.domain_label_a

        # True domain labels
        domain_true_t = _to_gpu(torch.LongTensor([0]*domain_pred_t.size(0)))
        domain_true_v = _to_gpu(torch.LongTensor([1]*domain_pred_v.size(0)))
        domain_true_a = _to_gpu(torch.LongTensor([2]*domain_pred_a.size(0)))

        # Stack up predictions and true labels
        domain_pred = torch.cat((domain_pred_t, domain_pred_v, domain_pred_a), dim=0)
        domain_true = torch.cat((domain_true_t, domain_true_v, domain_true_a), dim=0)

        return domain_loss_criterion(domain_pred, domain_true)
    
    def get_recon_loss(self, model, loss_recon):
        """Calculate reconstruction loss using given function
        Args:
            model (torch.nn.module): The multimodal fusion model.
            loss_recon (nn.Fuunctional): The loss function to compute the reconstruction loss. In original paper it's the MSE (mean square error) loss.
        Return:
            loss: Overall reconstruction loss of the model.
        """
        loss = loss_recon(model.utt_t_recon, model.utt_t_orig)
        loss += loss_recon(model.utt_v_recon, model.utt_v_orig)
        loss += loss_recon(model.utt_a_recon, model.utt_a_orig)
        loss = loss/3.0
        return loss

    def get_cmd_loss(self, model, loss_cmd):

        if not self.config.use_cmd_sim:
            return 0.0

        # losses between shared states
        loss = loss_cmd(model.utt_shared_t, model.utt_shared_v, 5)
        loss += loss_cmd(model.utt_shared_t, model.utt_shared_a, 5)
        loss += loss_cmd(model.utt_shared_a, model.utt_shared_v, 5)
        loss = loss/3.0
        return loss
    
    def multiclass_acc(self, preds, truths):
        """
        Compute the multiclass accuracy w.r.t. groundtruth
        :param preds: Float array representing the predictions, dimension (N,)
        :param truths: Float/int array representing the groundtruth classes, dimension (N,)
        :return: Classification accuracy
        """
        return np.sum(np.round(preds) == np.round(truths)) / float(len(truths))
    
    def _cuda_to_numpy(self, x):
        return x.detach().cpu().numpy()

    def calc_metrics(self, y_true, y_pred):
        if self.config.data == "ur_funny":
            test_preds = np.argmax(y_pred, 1)
            test_truth = y_true

            if self.config.print:
                print("Confusion Matrix (pos/neg):")
                print(confusion_matrix(test_truth, test_preds))
                print("Classification Report (pos/neg):")
                print(classification_report(test_truth, test_preds, digits=5))
                print("Accuracy: (pos/neg)", accuracy_score(test_truth, test_preds))

            return accuracy_score(test_truth, test_preds)
        
        else:
            test_preds = y_pred
            test_truth = y_true

            non_zeros = np.array([i for i, e in enumerate(test_truth) if e != 0])

            test_preds_a7 = np.clip(test_preds, a_min=-3., a_max=3.)
            test_truth_a7 = np.clip(test_truth, a_min=-3., a_max=3.)
            test_preds_a5 = np.clip(test_preds, a_min=-2., a_max=2.)
            test_truth_a5 = np.clip(test_truth, a_min=-2., a_max=2.)

            mae = np.mean(np.absolute(test_preds - test_truth))
            corr = np.corrcoef(test_preds, test_truth)[0][1]
            mult_a7 = self.multiclass_acc(test_preds_a7, test_truth_a7)
            mult_a5 = self.multiclass_acc(test_preds_a5, test_truth_a5)
            
            f_score = f1_score((test_preds[non_zeros] > 0), (test_truth[non_zeros] > 0), average='weighted')

            # pos - neg
            binary_truth = (test_truth[non_zeros] > 0)
            binary_preds = (test_preds[non_zeros] > 0)

            if self.config.print:
                print("mae:", mae)
                print("corr:", corr)
                print("mult_acc:", mult_a7)
                print("Classification Report (pos/neg) :")
                print(classification_report(binary_truth, binary_preds, digits=5))
                print("Accuracy  (pos/neg)", accuracy_score(binary_truth,binary_preds))
            
            # non-neg-neg
            binary_truth = (test_truth >= 0)
            binary_preds = (test_preds >= 0)

            if self.config.print:
                print("Classification Report (non-neg/neg) :")
                print(classification_report(binary_truth, binary_preds, digits=5))
                print("Accuracy (non-neg/neg)", accuracy_score(binary_truth, binary_preds))
            
            return accuracy_score(binary_truth, binary_preds)

    def _restore_model(self):
        """Restore previously trained model
        """
        data_loader = self._load_data()
        model = MISA(self.config)
        model.load_state_dict(torch.load(self.config.ckpt_path))
        return model, data_loader

    def disp_attinfo(self):
        raise NotImplementedError("Attack information displaying function hasn't yet implemented")

    def attack(self, model, data_loader):
        """Using FGSM method to attack input data
            ***core function in this class***
        Args:
            model (nn.Module): The targeted model with trained parameters
            ckpt_path (str): The relative path to store the checkpoint file.
        Returns:
            (Optional)
        """
        raise NotImplementedError("Main attack method hasn't been implemented yet")
        

    def save_data(self, data):
        """Store adversarial data back to disk
        Args:
            data: Adversarial data in readable format
        Returns:
            None
        """
        pass

    def eval_adv(self, model, *data):
        """Evaluate model performance with manufactured data
        Args:
            data: The same as above
            model: The same as above
        """
        # Note: bert_xxx are dummy placeholder for compliance
        t, v, a, y_true, l, bert_sent, bert_sent_type, bert_sent_mask = data
        model.eval()

        with torch.no_grad():
            model.zero_grad()
            
            y_tilde = model(t, v, a, l, bert_sent, bert_sent_type, bert_sent_mask)

            if self.config.data == "ur_funny":
                y = y.squeeze()
            
            cls_loss = self.criterion(y_tilde, y_true)
            # Count the number of samples for calculating the average
            num_samples = t.size(0)

        # Return to training mode for the next iteration
        model.train()

        return y_true, y_tilde, cls_loss, num_samples

    def attack_and_save(self):
        """The main function that packs up the whole attack process 
        (Currently not support "save" operation) 
        """
        model, data_loader = self._restore_model()
        eval_loss, accuracy = self.attack(model, data_loader)

class FGSMAttacker(BaseAttacker):
    """ FGSM back propogation
    Rebuild trained model and backpropagate gradients to input data.
    Modify data using FGSM gradient ascent and dump all data into pickle file.
    Evaluate new samples under different ascending step, yield and compare the results. 
    """
    def __init__(self, config):
        super(FGSMAttacker, self).__init__(config=config)
        if config.modals == "all":
            attack_t = attack_v = attack_a = True
        elif 'a' in config.modals:
            attack_a = True
        elif 'v' in config.modals:
            attack_v = True
        elif 't' in config.modals:
            attack_t = True

        self.config.attack_a = attack_a
        self.config.attack_v = attack_v
        self.config.attack_t = attack_t

    def disp_attinfo(self, epsilon):
        print('''\
        ##########################################
                FGSM attack (epsilon={})
        ##########################################
        '''.format(epsilon))

    def attack(self, model, data_loader):
        """Using FGSM method to attack input data
            ***core function in this class***
        Args:
            model (nn.Module): The targeted model with trained parameters
            ckpt_path (str): The relative path to store the checkpoint file.
        Returns:
            adv_data (dict[torch.Tensor]): A dictionary containing several tensors which are
            corresponding to each item in original inputs.
        """
        epsilon = self.config.epsilon
        self.disp_attinfo(epsilon)

        # load model and data
        all_loss = []
        all_count = []
        all_y_true = []
        all_y_pred = []

        domain_loss_criterion = nn.CrossEntropyLoss(reduction="mean")
        loss_diff = DiffLoss()
        loss_recon = MSE()
        loss_cmd = CMD()

        # Attack data batch by batch
        for batch in data_loader:
            model.zero_grad()
            t, v, a, y, l, bert_sent, bert_sent_type, bert_sent_mask = batch
            batch_size = t.size(0)
            # t, v, a, y, l, bert_sent, bert_sent_type, bert_sent_mask = self._to_gpu(t, v, a, y, l, bert_sent, bert_sent_type, bert_sent_mask)
            t, v, a, y, bert_sent, bert_sent_type, bert_sent_mask = self._to_gpu(t, v, a, y, bert_sent, bert_sent_type, bert_sent_mask)
            self._to_gpu(model)
            # only vision and accoustic need gradient
            self._add_grad_reqr(v, a)
            
            y_tilde = model(t, v, a, l, bert_sent, bert_sent_type, bert_sent_mask)

            if self.config.data == "ur_funny":
                y = y.squeezea
            
            cls_loss = self.criterion(y_tilde, y)
            diff_loss = self.get_diff_loss(model, loss_diff)
            domain_loss = self.get_domain_loss(model, loss_diff)
            recon_loss = self.get_recon_loss(model, loss_recon)
            cmd_loss = self.get_cmd_loss(model, loss_cmd)

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
            if self.config.attack_t:
                with torch.no_grad():
                    embed_weight = model.embed.weight
                    embed_weight += epsilon * embed_weight.grad

            if self.config.attack_v:
                v_adv = v + epsilon * v.grad
            
            if self.config.attack_t:
                a_adv = a + epsilon * a.grad

            # Evaluate at once
            y_true, y_pred, loss, num_samples = self.eval_adv(model, t, v_adv, a_adv, y, l, None, None, None)

            all_loss.append(loss.item())
            all_count.append(num_samples)

            all_y_true.append(self._cuda_to_numpy(y_true))
            all_y_pred.append(self._cuda_to_numpy(y_pred))

            # Restore the original embedding layer
            if self.config.attack_t:
                with torch.no_grad():
                    embed_weight -= epsilon * embed_weight.grad
        
        y_true = np.concatenate(all_y_true, axis=0).squeeze()
        y_pred = np.concatenate(all_y_pred, axis=0).squeeze()

        # Show the report of evalutaion
        accuracy = self.calc_metrics(y_true, y_pred)
        eval_loss = np.mean(all_loss)
        #TODO: Are these returns really needed?
        return eval_loss, accuracy
    

if __name__ == '__main__':
    # Use test data
    config = get_config(mode='test')

    # set random seed
    random_name = str(random())
    random_seed = 1234
    torch.manual_seed(random_seed)

    np.random.seed(random_seed)

    # set deterministic
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    # config = add_attspecconfig(config)
    Attacker = FGSMAttacker(config=config)
    Attacker.attack_and_save()