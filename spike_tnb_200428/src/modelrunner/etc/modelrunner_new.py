from pathlib import Path
import traceback
from logging import getLogger

from sklearn import metrics
import numpy as np
import torch
from torch import nn
from torch import optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

##  added on 21/6/8 by Nakajima & Honji
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dataset.split import SplitDataset
## 

MODEL_FILENAME = 'spike.pt'
MODEL_BEST_FILENAME = 'spike_best.pt'
LOSS_IMG_FILENAME = 'loss.pdf'
ACCURACY_IMG_FILENAME = 'accuracy.pdf'
OUTPUT_DIRNAME = 'output'
OUTPUT_BEST_DIRNAME = 'output_best'
GENERATED_FILENAME = 'generated.npy'
GROUND_TRUTH_FILENAME = 'groundtruth.npy'


class ModelRunner(object):
    """docstring for ModelRunner."""

    def __init__(self, mode, experiment_name, config):
        super().__init__()

        assert mode in ['train', 'eval', 'generate']

        self.mode = mode
        self.experiment_name = experiment_name
        self.config = config
        self.dir_result = Path(self.config.dir_result)
        self.logger = getLogger(__name__)

        self.device = torch.device(
            f'cuda:{self.config.cuda[0]}'
            if torch.cuda.is_available() else 'cpu')
        self.model = self._init_model()
        self.modelpath = self.dir_result / MODEL_FILENAME
        self.modelbestpath = self.dir_result / MODEL_BEST_FILENAME
        if self.modelpath.exists():
            self.model.load_state_dict(torch.load(self.modelpath))
        self.criterion = self._init_criterion()
        self.optimizer = self._init_optimizer()

        if len(self.config.cuda) == 1:
            self.model = self.model.to(self.device)
        else:
            self.model = nn.DataParallel(
                self.model, device_ids=self.config.cuda)
            self.model = self.model.to(self.device)

    def run(self, *args, **kwargs):
        if self.mode == 'train':
            self._train(*args, **kwargs)
        elif self.mode == 'eval':
            self._eval(*args, **kwargs)
            if self.modelbestpath.exists():
                self.model.load_state_dict(torch.load(self.modelbestpath))
                self._eval(*args, **kwargs)
        elif self.mode == 'generate':
            self._generate(*args, **kwargs)

    def _train(self, *args, **kwargs):
        dataloader_train, dataloader_valid, dataloader_test = self._load_data()

        writer = SummaryWriter(comment=self.experiment_name)
        results_train = {
            'loss': [],
            'accuracy': {
                'all': [],
                'spike': [],
                'nonspike': [],
            },
        }
        results_valid = {
            'loss': [],
            'accuracy': {
                'all': [],
                'spike': [],
                'nonspike': [],
            },
        }
        best_valid = {
            'accuracy': float('-inf'),
            'model': None,
        }

        try:
            for epoch in range(self.config.n_epoch):
                self.model.train()
                sample_number_sum = {
                    'all': 0,
                    'spike': 0,
                    'nonspike': 0,
                }
                loss = 0
                correct_count_sum = {
                    'all': 0,
                    'spike': 0,
                    'nonspike': 0,
                }
                for data in dataloader_train:
                    loss_value, output = self._process_batch(data)
                    loss += loss_value * data.numel()
                    for key, value in self._get_sample_number(data).items():
                        sample_number_sum[key] += value
                    for key, value in self._calculate_correct_sample_number(data[:, 1:], output).items():
                        correct_count_sum[key] += value
                loss_train = loss / sample_number_sum['all']
                accuracy_train = {
                    key: correct_count / sample_number
                    for (key, correct_count), sample_number
                    in zip(correct_count_sum.items(), sample_number_sum.values())}

                self.model.eval()
                sample_number_sum = {
                    'all': 0,
                    'spike': 0,
                    'nonspike': 0,
                }
                loss = 0
                correct_count_sum = {
                    'all': 0,
                    'spike': 0,
                    'nonspike': 0,
                }
                for data in dataloader_valid:
                    loss_value, output = self._process_batch(data, is_train=False)
                    loss += loss_value * data.numel()
                    for key, value in self._get_sample_number(data).items():
                        sample_number_sum[key] += value
                    for key, value in self._calculate_correct_sample_number(data[:, 1:], output).items():
                        correct_count_sum[key] += value
                loss_valid = loss / sample_number_sum['all']
                accuracy_valid = {
                    key: correct_count / sample_number
                    for (key, correct_count), sample_number
                    in zip(correct_count_sum.items(), sample_number_sum.values())}

                results_train['loss'].append(loss_train)
                for key, value in accuracy_train.items():
                    results_train['accuracy'][key].append(value)
                results_valid['loss'].append(loss_valid)
                for key, value in accuracy_valid.items():
                    results_valid['accuracy'][key].append(value)

                writer.add_scalars(
                    f'loss/{self.experiment_name}',
                    {'train': loss_train,
                     'valid': loss_valid},
                    global_step=epoch)
                writer.add_scalars(
                    f'accuracy/{self.experiment_name}',
                    {'train': accuracy_train['all'],
                     'valid': accuracy_valid['all']},
                    global_step=epoch)

                self.logger.info(
                    f'train: {epoch:4}, loss: {loss_train:.6}, accu: {accuracy_train}')
                self.logger.info(
                    f'valid: {epoch:4}, loss: {loss_valid:.6}, accu: {accuracy_valid}')
                    

                if best_valid['accuracy'] < accuracy_valid['all']:
                    best_valid['accuracy'] = accuracy_valid['all']
                    if len(self.config.cuda) == 1:
                        best_valid['model'] = self.model.state_dict()
                    else:
                        best_valid['model'] = self.model.module.state_dict()

                if epoch % 100 == 99:
                    self._savemodel(best_valid['model'])
                    self._savefig(results_train, results_valid)
        except KeyboardInterrupt as e:
            self.logger.error(traceback.format_exc())
            self._savemodel(best_valid['model'])
            self._savefig(results_train, results_valid)
            raise

        self._savemodel(best_valid['model'])
        self._savefig(results_train, results_valid)

        self.model.eval()
        sample_number_sum = {
            'all': 0,
            'spike': 0,
            'nonspike': 0,
        }
        correct_count_sum = {
            'all': 0,
            'spike': 0,
            'nonspike': 0,
        }
        for data in dataloader_test:
            _, output = self._process_batch(data, is_train=False)
            for key, value in self._get_sample_number(data).items():
                sample_number_sum[key] += value
            for key, value in self._calculate_correct_sample_number(data[:, 1:], output).items():
                correct_count_sum[key] += value
        accuracy_test = {
            key: correct_count / sample_number
            for (key, correct_count), sample_number
            in zip(correct_count_sum.items(), sample_number_sum.values())}
        self.logger.info(f'test accuracy: {accuracy_test}')

    def _get_sample_number(self, data):
        sample_number_all = data.numel()
        sample_number_spike = data[data == 1].numel()
        sample_number_nonspike = sample_number_all - sample_number_spike
        return {
            'all': sample_number_all,
            'spike': sample_number_spike,
            'nonspike': sample_number_nonspike,
        }

    def _calculate_correct_sample_number(self, data, output):
        data = data.flatten()
        output = output.detach().round().flatten()
        spike_data, spike_output = data[data == 1], output[data == 1]

        correct_count = metrics.accuracy_score(
            data, output, normalize=False)
        correct_count_spike = metrics.accuracy_score(
            spike_data, spike_output, normalize=False)
        correct_count_nonspike = correct_count - correct_count_spike
        return {
            'all': correct_count,
            'spike': correct_count_spike,
            'nonspike': correct_count_nonspike,
        }

    def _savefig(self, results_train, results_valid):
        plt.plot(results_train['loss'], label='train')
        plt.plot(results_valid['loss'], label='valid')
        plt.legend()
        plt.savefig(self.dir_result / LOSS_IMG_FILENAME)
        plt.close()

        _, axes_list = plt.subplots(nrows=3, ncols=1, figsize=(8, 6))
        for axes, accuracy_train, accuracy_valid in zip(axes_list, results_train['accuracy'].values(), results_valid['accuracy'].values()):
            axes.plot(accuracy_train, label='train')
            axes.plot(accuracy_valid, label='valid')
            axes.legend()
        plt.savefig(self.dir_result / ACCURACY_IMG_FILENAME)
        plt.close()

    def _savemodel(self, best_valid_model):
        torch.save(best_valid_model, self.modelbestpath)
        if len(self.config.cuda) == 1:
            torch.save(self.model.state_dict(), self.modelpath)
        else:
            torch.save(self.model.module.state_dict(), self.modelpath)

    def _process_batch(self, data, target, is_train=True):
        raise NotImplementedError()

    def _eval(self, *args, **kwargs):
        _, _, dataloader_test = self._load_data()

        self.model.eval()
        sample_number_sum = {
            'all': 0,
            'spike': 0,
            'nonspike': 0,
        }
        correct_count_sum = {
            'all': 0,
            'spike': 0,
            'nonspike': 0,
        }
        for data in dataloader_test:
            _, output = self._process_batch(data, is_train=False)
            for key, value in self._get_sample_number(data).items():
                sample_number_sum[key] += value
            for key, value in self._calculate_correct_sample_number(data[:, 1:], output).items():
                correct_count_sum[key] += value
        accuracy_eval = {
            key: correct_count / sample_number
            for (key, correct_count), sample_number
            in zip(correct_count_sum.items(), sample_number_sum.values())}
        self.logger.info(f'eval accuracy: {accuracy_eval}')

    def _generate(self, *args, generate_mode, quantization_mode, **kwargs):
        _, _, test_data = self._load_data(raw=True)
        test_data = test_data[None, :, :] # add batch dimension

        self.model.eval()
        if generate_mode == 'autoregression':
            outputs = self._autoregression(test_data, quantization_mode)
            result = []
            for output in outputs:
                result.append(output.to('cpu').numpy()[:, 0, :])
            result = np.array(result).transpose(1, 0, 2)
        elif generate_mode == 'fromdata':
            result = self._generate_from_data(test_data, quantization_mode)
            result = result.to('cpu').numpy()

        test_data = test_data[0].numpy()
        result = result[0]
        np.save(
            self.dir_result
            / f'{generate_mode}_{quantization_mode}_{GROUND_TRUTH_FILENAME}',
            test_data.astype(np.bool)
        )
        np.save(
            self.dir_result
            / f'{generate_mode}_{quantization_mode}_{GENERATED_FILENAME}',
            result.astype(np.bool)
        )

    def _autoregression(self, data, quantization_mode):
        raise NotImplementedError()

    def _generate_from_data(self, data, quantization_mode):
        raise NotImplementedError()

    def _init_model(self):
        raise NotImplementedError()

    def _init_criterion(self):
        raise NotImplementedError()

    def _init_optimizer(self):
        raise NotImplementedError()

    def _load_data(self, raw=False):
        data = np.load(self.config.dir_spike)[self.config.use_segment]
        segment_size, dim = data.shape
        assert dim == self.config.dim_input

        train_size = int(segment_size * self.config.train_rate)
        valid_size = int(segment_size * self.config.valid_rate)

        train_data = torch.as_tensor(data[:train_size, :], dtype=torch.float32)
        valid_data = torch.as_tensor(data[train_size:train_size+valid_size, :], dtype=torch.float32)
        test_data = torch.as_tensor(data[train_size+valid_size:, :], dtype=torch.float32)

        if raw:
            return train_data, valid_data, test_data

        data_train = SplitDataset(train_data, split_size=self.config.split_size)
        data_valid = SplitDataset(valid_data, split_size=self.config.split_size)
        data_test = SplitDataset(test_data, split_size=self.config.split_size)

        dataloader_train = DataLoader(data_train, batch_size=self.config.batch_size)
        dataloader_valid = DataLoader(data_valid, batch_size=self.config.batch_size)
        dataloader_test = DataLoader(data_test, batch_size=self.config.batch_size)

        return dataloader_train, dataloader_valid, dataloader_test
