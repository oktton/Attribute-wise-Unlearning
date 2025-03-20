import os

from models.lightgcn import LightGCN
from models.ncf import NCF


def prepare_rec_model(user_num, item_num, train_dataset, args):
    args.logger.info(f'prepare {args.model} model...')
    if args.model == 'ncf':
        return NCF(user_num, item_num)
    else:
        save_path = os.path.join(args.dataset_path, f'{args.dataset}', f'preprocess')
        sparse_graph = train_dataset.get_sparse_graph(save_path)
        return LightGCN(sparse_graph, user_num, item_num)
