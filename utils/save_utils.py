import logging
import os


def construct_logging_name(args):
    name = 'train_log'
    if hasattr(args, 'attack'):
        name = name + f'-attack.log'
    else:
        name = name + '.log'

    return name


def construct_logger(args):
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)
    formatter = logging.Formatter('[%(asctime)s] %(message)s', datefmt='%Y-%m-%d %H:%M:%S')

    file_handler = logging.FileHandler(os.path.join(args.save_path, construct_logging_name(args)))
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    screen_handler = logging.StreamHandler()
    screen_handler.setLevel(logging.INFO)
    screen_handler.setFormatter(formatter)
    logger.addHandler(screen_handler)

    return logger


def construct_weight_path_wocheck(args):
    path = 'weights'
    weight_path = os.path.join(args.save_path, path)
    weight_path = os.path.join(weight_path, args.target_model)
    return weight_path


def construct_weight_path(args):
    path = 'weights'
    weight_path = os.path.join(args.save_path, path)
    if not os.path.exists(weight_path):
        os.makedirs(weight_path)
    return weight_path


def check_repeat_running(args):
    path = 'weights'
    return os.path.exists(os.path.join(args.save_path, path))


def construct_target_model_path(target_model_path, target_model):
    path = 'weights'
    weight_path = os.path.join(target_model_path, path)
    return os.path.join(weight_path, target_model)
