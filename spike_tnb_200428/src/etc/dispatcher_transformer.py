# -*- coding: utf-8 -*-
"""
Created on Wed Jul 14 22:35:58 2021

@author: tanabe
"""
import os
import shutil

from src import log
from src.config.config import Config
from src.modelrunner.transformer import TransformerRunner  # モデルランナーのインポートを変更
from src.modelrunner.transformer_focal import TransformerFocalRunner  # モデルランナーのインポートを変更
from src.modelrunner.transformer_stack import TransformerStackRunner  # モデルランナーのインポートを変更

import torch

def dispatch(rootdir, CONFIG_FILENAME, CONFIG_PATH, MODEL, MODE, GENERATE_MODE, QUANTIZATION_MODE):

    modelrunners = {
        'transformer': TransformerRunner,  # キーを変更
        'transformer_focal': TransformerFocalRunner,  # キーを変更
        'transformer_stack': TransformerStackRunner,  # キーを変更
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
