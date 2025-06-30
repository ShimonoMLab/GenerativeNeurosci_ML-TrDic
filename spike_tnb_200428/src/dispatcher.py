#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 14 22:35:58 2021

@author: tanabe
"""
import os
import shutil

from src import log
from src.config.config import Config
from src.modelrunner.lstm import LSTMRunner
from src.modelrunner.lstm_focal import LSTMFocalRunner
from src.modelrunner.lstm_stack import LSTMStackRunner

import torch
#torch.cuda.empty_cache()

def dispatch(rootdir, CONFIG_FILENAME, CONFIG_PATH, MODEL, MODE, GENERATE_MODE, QUANTIZATION_MODE):

# MODEL = 'lstm_stack'
# # MODEL = 'lstm_focal' # choose {lstm, lstm_focal}  -- the second one is focal loss.
# MODE = 'eval' # choose {train,eval,generate}
# GENERATE_MODE = 'fromdata' # chose {'autoregression', 'fromdata'}
# QUANTIZATION_MODE = 'threshold' # choses {'threshold', 'sample'}

    modelrunners = {
        'lstm': LSTMRunner,
        'lstm_focal': LSTMFocalRunner,
        'lstm_stack': LSTMStackRunner,
    }

    assert MODEL in modelrunners, f'can not use the model {MODEL}'
    experiment_name = MODEL

    config = Config().read(os.path.join(rootdir, CONFIG_PATH))
    config.dir_result = os.path.join(config.dir_result, MODEL)

    os.makedirs(config.dir_result, exist_ok=True)

    shutil.copy(os.path.join(rootdir, CONFIG_PATH),
                os.path.join(config.dir_result, CONFIG_FILENAME))
    log.setlogger(config.dir_result, config.logger)

    modelrunner = modelrunners[MODEL](MODE, experiment_name, config)
    modelrunner.run(
        generate_mode=GENERATE_MODE,
        quantization_mode=QUANTIZATION_MODE,
    )

