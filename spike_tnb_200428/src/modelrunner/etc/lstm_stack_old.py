from sklearn import metrics
import torch
from torch import nn
from torch import optim

from src.modelrunner.modelrunner import ModelRunner
from src.net.lstm_stack import LSTMStack
from src.loss.focal_loss import BinaryFocalLoss
from src import utils


class LSTMStackRunner(ModelRunner):
    """docstring for LSTMStackRunner."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.batch_size = self.config.lstm['batch_size']

    def _init_model(self):
        model = LSTMStack(
            n_input=self.config.dim_input,
            n_layer_list=self.config.lstm_stack['n_layer_list'],
            n_hidden_list=self.config.lstm_stack['n_hidden_list'],
            n_output=self.config.dim_input,
        )

        return model

    def _init_criterion(self):
        return BinaryFocalLoss(gamma=self.config.lstm_stack['gamma'])

    def _init_optimizer(self):
        return optim.Adam(self.model.parameters())

    def _process_batch(self, data, is_train=True):
        data = data.to(self.device)
        len(data)
        if is_train:
            self.optimizer.zero_grad()
            output, _ = self.model(data)
            output = output[:, :-1]
            loss = self.criterion(output, data[:, 1:].long())
            loss.backward()
            self.optimizer.step()
        else:
            with torch.no_grad():
                output, _ = self.model(data)
                output = output[:, :-1]
            loss = self.criterion(output, data[:, 1:].long())

        output = output.to('cpu')
        loss_value = loss.item()
        accuracy_value = metrics.accuracy_score(
            data[:, 1:].to('cpu').flatten(),
            torch.round(output).to('cpu').detach().flatten(),
            normalize=True)
        return loss_value, accuracy_value, output

    def _autoregression(self, data, quantization_mode):
        quantization = {
            'threshold': torch.round,
            'sample': utils.sample,
        }[quantization_mode]

        length = data.shape[1]
        data = data[:, 0:1, :].to(self.device)

        hidden = None
        for i in range(length):
            with torch.no_grad():
                output, hidden = self.model(data, hidden)
                data = quantization(output).to(self.device)
            yield data

    def _generate_from_data(self, data, quantization_mode):
        quantization = {
            'threshold': torch.round,
            'sample': utils.sample,
        }[quantization_mode]

        data = data.to(self.device)
        with torch.no_grad():
            output, _ = self.model(data)
            output = quantization(output).to(self.device)
        return output
