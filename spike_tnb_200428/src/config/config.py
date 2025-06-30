import configparser


class Config(object):

    def read(self, path):
        self.config = configparser.ConfigParser()
        self.config.read(path)

        self.logger = self._get('DEFAULT', 'logger')
        self.cuda = self._getlist('DEFAULT', 'cuda', int)
   #     self.cuda = self._getlist('DEFAULT', 'cuda', lambda x: int(x.split('#')[0].strip()))


        self.use_segment = self._getint('DEFAULT', 'use_segment')
        self.n_epoch = self._getint('DEFAULT', 'n_epoch')
        self.split_size = self._getint('DEFAULT', 'split_size')
        self.batch_size = self._getint('DEFAULT', 'batch_size')
        self.dim_input = self._getint('DEFAULT', 'dim_input')
        self.train_rate = self._getfloat('DEFAULT', 'train_rate')
        self.valid_rate = self._getfloat('DEFAULT', 'valid_rate')

        self.dir_base = self._get('path', 'base')
        self.dir_spike = self._get('path', 'spike')
        self.dir_result = self._get('path', 'result')

        self.lstm = {
            'n_lstm_hidden': self._getint('lstm', 'n_lstm_hidden'),
            'batch_size': self._getint('lstm', 'batch_size'),
        }

        self.lstm_stack = {
            'gamma': self._getfloat('lstm_stack', 'gamma'),
            'n_hidden_list': self._getlist('lstm_stack', 'n_hidden_list', int),
            'n_layer_list': self._getlist('lstm_stack', 'n_layer_list', int),
            'batch_size': self._getint('lstm', 'batch_size'),
        }

        return self

    def _getlist(self, section, option, func):
        return list(map(func, self.config.get(section, option).split(',')))
    
   # def _getlist(self, section, option, func):
   # return list(map(
   #     func,
   #     [value.split('#')[0].strip() for value in self.config.get(section, option).split(',')]))


    def _get(self, section, option):
        return self.config.get(section, option)

    def _getint(self, section, option):
        return self.config.getint(section, option)
    
    # def _getint(self, section, option):
    # value = self.config.get(section, option).split('#')[0].strip()
    # return int(value)

    def _getfloat(self, section, option):
        return self.config.getfloat(section, option)

    def _getboolean(self, section, option):
        return self.config.getboolean(section, option)
