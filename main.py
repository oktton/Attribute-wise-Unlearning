import json
import os

import torch.backends.cudnn as cudnn

from methods.adv import adv
from methods.u2u import u2u
from methods.d2d_retrain import d2d, retrain
from methods.original import original
from utils.save_utils import construct_logger, check_repeat_running, construct_target_model_path


def initialize_settings(path):
    class Args:
        def __init__(self, dictionary):
            for k, v in dictionary.items():
                setattr(self, k, v)

        def add_attribute(self, k, v):
            setattr(self, k, v)

        def record_config(self, logger):
            for attr in self.__dict__:
                logger.info(f'{attr}: {getattr(self, attr)}')

    with open(path, 'r') as f:
        configuration = json.load(f)
    return Args(configuration)


def in_training_methods(args):
    if check_repeat_running(args):
        args.logger.info('This setting exists')
    else:
        if args.method == 'original':
            original(args)
        elif args.method == 'adv':
            adv(args)
        elif args.method == 'retrain':
            retrain(args)


def post_training_methods(args):
    if check_repeat_running(args):
        args.logger.info('This setting exists')
    else:
        target_model_path = f'./exp_results/{args.model}/{args.dataset}/original/'
        target_model_path = construct_target_model_path(target_model_path, args.target_model)
        if not os.path.exists(target_model_path):
            args.logger.info(f'Target model path {target_model_path} does not exist')
        else:
            args.add_attribute("target_model_path", target_model_path)
            if args.method == 'd2d':
                d2d(args)
            elif args.method == 'u2u':
                u2u(args)   


if __name__ == '__main__':
    config_path = 'configs/exp_config.json'
    args = initialize_settings(config_path)
    cudnn.benchmark = True

    save_path = f'./exp_results/{args.model}/{args.dataset}/{args.method}/'
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    args.add_attribute("save_path", save_path)

    logger = construct_logger(args)
    args.record_config(logger)
    args.add_attribute("logger", logger)

    if args.method in ['original', 'adv', 'retrain']:
        in_training_methods(args)
    else:
        post_training_methods(args)
