import torch
from torch import nn
from torch import optim
import sys


from src.modelrunner.modelrunner import ModelRunner
from src.net.lstm_stack import LSTMStack
from src.loss.focal_loss import BinaryFocalLoss
#  modified and commented again by Honji 21/06/10
# from modelrunner import ModelRunner
# from net.lstm_stack import LSTMStack
# from loss.focal_loss import BinaryFocalLoss

import utils


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
            output0 = output
            
            ##  2021/6/3 -- added for weighted label method ###############
            print('data: ')
            print(data.dtype)
            print('output: ')
            print(output.dtype)
  #          data0 = (data>0.0001) # .quantize
#            data0 = (data>0.5) # .quantize
#            output0 = (output>0.0001).float() # .quantize
#            output0 = (output>0.5).float() # .quantize
#            print('data0: ')
#            print(data0[:,1:5].dtype)
#            print('output0: ')
#            print(output0.dtype)
#            loss = self.criterion(output0, data0[:, 1:].long())
            ## ###########################################################
            loss = self.criterion(output, data[:, 1:].long()) # orignnal
            
            print(loss) 
           # print('loss: ') nakajima 20210614xc
            print(loss.dtype) 
           # print('loss.dtype: ') nakajima 20210614
           
            print(output)
            loss.backward()
#            print(loss)
            self.optimizer.step()
            
        else:
            with torch.no_grad():
                output, _ = self.model(data)
                output = output[:, :-1]
                
             ################data
            print('data: ')
            print(data[:,1:5])
            print('output: ')
            print(output)
            
            ## 2021/6/3 -- added for weighted label method #
            #data0 = (data>0.0001) # .quantize
            #output0 = (output>0.0001).float() # .quantize
            data0 = (data>0.5) # .quantize
            output0 = (output>0.5).float() # .quantize
            
            print('data0: ')
            print(data0[:,1:5].dtype)
            print('output0: ')
            print(output0.dtype)
            loss = self.criterion(output0, data0[:, 1:].long())
            ## #############################################################
            
            # loss = self.criterion(output, data[:, 1:].long()) # orignnal
            
            print(loss)

        output = output.to('cpu')
        thresh = 0.5
        loss = (loss > thresh) * 1
        print(loss)
        loss_value = loss.item()
        return loss_value, output

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
                # print(str(i))
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

print('finish')