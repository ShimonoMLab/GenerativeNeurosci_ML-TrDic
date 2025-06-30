from logging import (getLogger, StreamHandler, FileHandler, Formatter,
                     INFO, DEBUG)

def setlogger(path, name='log'):
    logger = getLogger(__name__.split('.')[0])
    logger.setLevel(DEBUG)

    stream_handler_format = Formatter('%(message)s')
    file_handler_format = Formatter(
        '%(asctime)s - %(name)s - %(levelname)s\t: %(message)s')

    stream_handler = StreamHandler()
    stream_handler.setLevel(INFO)
    stream_handler.setFormatter(stream_handler_format)

    file_handler = FileHandler(f'{path}/{name}.log')
    file_handler.setLevel(DEBUG)
    file_handler.setFormatter(file_handler_format)

    logger.addHandler(stream_handler)
    logger.addHandler(file_handler)
    logger.propagate = False
