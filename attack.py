import json
import os
import random
import numpy as np
import torch
import xgboost as xgb
from sklearn.model_selection import cross_validate
from sklearn.neural_network import MLPClassifier
from data.data_utils import get_label_dict
from utils.save_utils import construct_logger, construct_weight_path_wocheck


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


def get_set_attackers():
    return {
        "MLP": MLPClassifier(hidden_layer_sizes=[100, ], alpha=1, max_iter=500, learning_rate_init=0.01),

        "XGBoost": xgb.XGBClassifier(max_depth=4, tree_method="hist", n_estimators=50, reg_alpha=0, reg_lambda=1,
                                        learning_rate=0.5),

    }



def eval_attacker_multi_cross(X, y, class_num, args):
    scoring = ['accuracy', 'precision', 'recall', 'roc_auc']
    n_times = 3

    attacker_list = get_set_attackers()
    for name, clf in attacker_list.items():
        accumulate_acc, accumulate_precision, accumulate_recall, accumulate_auc = 0, 0, 0, 0
        for i in range(n_times):
            cv_results = cross_validate(clf, X, y, scoring=scoring, cv=5)

            accumulate_acc += cv_results['test_' + scoring[0]].mean()
            accumulate_precision += cv_results['test_' + scoring[1]].mean()
            accumulate_recall += cv_results['test_' + scoring[2]].mean()
            accumulate_auc += cv_results['test_' + scoring[3]].mean()

            args.logger.info(f"model {name} time {i} acc = {cv_results['test_' + scoring[0]].mean()}")
            args.logger.info(f"model {name} time {i} precision = {cv_results['test_' + scoring[1]].mean()}")
            args.logger.info(f"model {name} time {i} recall = {cv_results['test_' + scoring[2]].mean()}")
            args.logger.info(f"model {name} time {i} auc = {cv_results['test_' + scoring[3]].mean()}")
        args.logger.info(
            "accuracy = {:.4f}, precision = {:.4f}, recall = {:.4f}, auc = {:.4f}".format(
                accumulate_acc / n_times, 
                accumulate_precision / n_times, 
                accumulate_recall / n_times, 
                accumulate_auc / n_times
            )
        )

def main_attack_mutlti(user_mat, label_dict, args):
    args.logger.info('constructing dataset...')

    X, y = [], []
    for label in label_dict.keys():
        X.append(user_mat[label_dict[label]])
        y.extend([label] * len(label_dict[label]))
        args.logger.info(f'label {label} length = {len(label_dict[label])}')

    embeddings = np.array(np.concatenate(X, axis=0))
    labels = np.array(y)
    class_num = len(set(label_dict.keys()))
    eval_attacker_multi_cross(embeddings, labels, class_num, args)


def sample_attack_sample(label_dict, num_user, seed, logger):
    random.seed(seed)
    label_number_dict = {}

    labels = set(label_dict.values())
    for num, label in enumerate(labels):
        label_number_dict[label] = num

    label_user_dict = {num: [] for num in range(len(labels))}
    for i in range(num_user):
        label_user_dict[label_number_dict[label_dict[i]]].append(i)

    shortest_num = min([len(user_list) for user_list in label_user_dict.values()])
    for label in label_user_dict.keys():
        if len(label_user_dict[label]) > shortest_num:
            logger.info(f'len(user_list) = {len(label_user_dict[label])} > {shortest_num}')
            label_user_dict[label] = random.sample(label_user_dict[label], shortest_num)
            logger.info(f'len(user_list) = {len(label_user_dict[label])} = {shortest_num}')

    return label_user_dict


if __name__ == '__main__':
    config_path = 'configs/attack_config.json'
    args = initialize_settings(config_path)

    save_path = f'./exp_results/{args.model}/{args.dataset}/{args.method}/'
    if not os.path.exists(save_path):
        print('attack model not exist')
    else:
        args.add_attribute("save_path", save_path)
        logger = construct_logger(args)
        args.record_config(logger)
        args.add_attribute("logger", logger)
        model_path = construct_weight_path_wocheck(args)
        if not os.path.exists(model_path):
            args.logger.info(f'Target model path {model_path} does not exist')
        else:
            args.add_attribute("target_model_path", model_path)

        args.logger.info('loading user matrix...')
        device = torch.device(args.device)
        model = torch.load(args.target_model_path, map_location='cpu', weights_only=False)
        user_mat = model.get_user_embedding_weight().detach().numpy()

        args.logger.info('loading label user dict...')
        label_dict = get_label_dict(args)
        label_user_dict = sample_attack_sample(label_dict, user_mat.shape[0], 4, args.logger)

        main_attack_mutlti(user_mat, label_user_dict, args)
