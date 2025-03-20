import os
import time

import numpy as np
import torch
from torch import optim
from tqdm import tqdm

from methods.d2d_retrain import get_topk_list, marginal_loss
from data.data_utils import prepare_data, load_positive_dict, get_label_dict
from utils.evaluate import top100_metrics
from utils.save_utils import construct_weight_path

def compute_Laplacian_matrix(attribute_user_list, device):
    all_user = []
    for i in range(len(attribute_user_list)):
        all_user.extend(attribute_user_list[i])
    A = torch.zeros(len(all_user), len(all_user)).to(device)
    for i in attribute_user_list[0]:
        for j in attribute_user_list[1]:
            A[i, j] = 1
            A[j, i] = 1
    D = torch.diag(torch.sum(A, dim=1))
    Laplacian_matrix = D - A
    return Laplacian_matrix


def u2u(args):
    device = torch.device(args.device)
    train_loader, test_loader, user_num, item_num = prepare_data(args)
    positive_dict = load_positive_dict(args)
    model = torch.load(args.target_model_path, map_location="cpu", weights_only=False)
    model.to(device)
    
    optimize_param = model.get_require_grad_user_embedding()
    optimizer = optim.Adam(optimize_param.parameters(), lr=args.lr)

    user_attribute_dict = get_label_dict(args)
    attribute_user_list = []

    label_set = set(user_attribute_dict.values())
    for _, label_name in enumerate(label_set):
        label_list = []
        for x in range(len(user_attribute_dict)):
            if user_attribute_dict[x] == label_name:
                label_list.append(x)
        attribute_user_list.append(np.array(label_list))

    topk_items = get_topk_list(model, test_loader, 20, positive_dict, item_num)

    count, best_hr, best_ndcg = 0, 0, 0
    total_time = 0
    calculate_matrix_start_time = time.time()
    Laplacian_matrix = compute_Laplacian_matrix(attribute_user_list, device)
    calculate_matrix_time = time.time() - calculate_matrix_start_time
    for epoch in tqdm(range(500)):
        model.train()
        start_time = time.time()

        optimizer.zero_grad()
        user_embedding = model.get_require_grad_user_embedding()
        final_matrix = torch.mm(torch.mm(user_embedding.weight.t(), Laplacian_matrix), user_embedding.weight)
        loss_defense = torch.trace(final_matrix)
        loss_bpr, _ = marginal_loss(model, topk_items, 20, user_num, 0.05, device)

        total_loss = loss_defense + args.au_trade_off * loss_bpr
        total_loss.backward()
        optimizer.step()
        count += 1
        
        elapsed_time = time.time() - start_time
        total_time += elapsed_time
        if epoch % 50 == 0 or epoch == 499:
            model.eval()
            hr_list, ndcg_list = top100_metrics(model, test_loader)
            hrAT10, ndcgAT10 = hr_list[1], ndcg_list[1]
            args.logger.info(
                f"The time elapse of epoch {epoch:03d} is " + time.strftime("%H: %M: %S", time.gmtime(elapsed_time)))
            args.logger.info(f"epoch defense loss = {loss_defense.item()}, bpr loss = {loss_bpr.item()}")
            for idx, topk in zip([0, 1, 2, 3], [5, 10, 15, 20]):
                args.logger.info(f"HR@{topk}: {hr_list[idx]:.4f}\tNDCG@{topk}: {ndcg_list[idx]:.4f}")

            best_hr, best_ndcg, best_epoch = hrAT10, ndcgAT10, epoch
            save_path = construct_weight_path(args)
            torch.save(model, os.path.join(save_path, f'epoch{epoch}.pth'))

    args.logger.info(f"total_time: "+ time.strftime("%H: %M: %S", time.gmtime(total_time + calculate_matrix_time)))
    args.logger.info(f"End. epoch {best_epoch:03d}: HR = {best_hr:.4f}, NDCG = {best_ndcg:.4f}")
    torch.save(model, os.path.join(construct_weight_path(args), f'final_model.pth'))
