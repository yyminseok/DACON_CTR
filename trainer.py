import abc
import torch
from model import FM
import numpy as np
from torch import optim
from sklearn.metrics import log_loss, roc_curve, auc

def pr_auc(true_labels: np.ndarray, pred_probs: np.ndarray):
    fpr, tpr, thresholds = roc_curve(true_labels, pred_probs)
    pr_auc = auc(fpr, tpr)
    return pr_auc


def logloss(true_labels: np.ndarray, pred_probs: np.ndarray):
    loss = log_loss(true_labels, pred_probs)
    return loss


def normalized_entropy(true_labels: np.ndarray, pred_probs: np.ndarray):
    p = np.mean(pred_probs)
    logloss = log_loss(true_labels, pred_probs)
    deno = - (p * np.log(p) + (1 - p) * np.log(1 - p))
    return logloss / deno




class Trainer(abc.ABC):
    def __init__(self, loss_fn, optimizer, device, hypara_dict, save_dir='./params'):
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.device = device
        self.save_dir = save_dir
        self.hypara_dict = hypara_dict

    def fit(self, data_loader, valid_X, valid_y, epochs, model_params_file='tmp_params.pth'):
        model = self._build_model(self.hypara_dict)
        model = model.to(self.device)
        if self.optimizer == 'RMSprop':
            optimizer = optim.RMSprop(model.parameters(), lr=0.001)
        if self.optimizer == 'Adam':
            optimizer = optim.Adam(model.parameters(), lr=0.001)
        if self.optimizer == 'SGD':
            optimizer = optim.SGD(model.parameters(), lr=0.001)

        for epoch in range(epochs):
            model.train()
            max_auc = 0.0
            for inputs, labels in data_loader:
                inputs = inputs.to(self.device)
                labels = labels.to(self.device).float()

                optimizer.zero_grad()
                logits = model(inputs)
                loss = self.loss_fn(logits, labels)
                loss.backward()  # back propagetion
                optimizer.step()
            _, valid_auc = self._validation_model(model, valid_X, valid_y)
            if valid_auc > max_auc:
                self._save_model_params(model, model_params_file)
                max_auc = valid_auc

    @abc.abstractmethod
    def _build_model(self, hypara_dict):
        pass

    def _validation_model(self, model, valid_X, valid_y):
        model.eval()
        inputs = valid_X.to(self.device)
        with torch.no_grad():
            pred_probs = torch.sigmoid(model(inputs))
        pred_probs = pred_probs.to('cpu').detach().numpy()
        loss = logloss(valid_y, pred_probs)
        auc = pr_auc(valid_y, pred_probs)
        return loss, auc
    
    def _save_model_params(self, model, model_params_file):
        torch.save(model.state_dict(), f'{self.save_dir}/{model_params_file}')
    
    def predict(self, eval_X, model_params_file='tmp_params.pth'):
        model = self._build_model(self.hypara_dict)
        model.load_state_dict(torch.load(f'{self.save_dir}/{model_params_file}'))
        model.eval()
        inputs = eval_X.to(self.device)
        with torch.no_grad():
            pred_probs = torch.sigmoid(model(inputs))
        return pred_probs


class FMTrainer(Trainer):
    def _build_model(self, hypara_dict):
        embed_rank = hypara_dict['embed_rank']
        field_dims = hypara_dict['field_dims']
        model = FM(field_dims, embed_rank)
        return model
