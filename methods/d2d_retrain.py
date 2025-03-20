import os
import time
import shutil

import numpy as np
import torch
import torch.nn as nn
from torch import optim
from tqdm import tqdm
from utils.prepare_models import prepare_rec_model

from data.data_utils import prepare_data, load_positive_dict, get_label_dict
from utils.evaluate import top100_metrics
from utils.save_utils import construct_weight_path


def get_topk_list(origin_model, test_loader, topk, positive_dict, item_num):
    user_topk_list = []

    device = next(origin_model.parameters()).device
    origin_model.eval()

    full_items = [i for i in range(item_num)]

    for user, item, label in test_loader:
        user_id = user[0].item()
        neg_items = list(set(full_items) - set(positive_dict[user_id]))
        new_user = torch.tensor([user_id] * len(neg_items), dtype=torch.long).to(device)
        new_item = torch.tensor(neg_items, dtype=torch.long).to(device)

        with torch.no_grad():
            predictions = origin_model(new_user, new_item)
            _, indices = torch.topk(predictions, topk)
            recommends = torch.take(new_item, indices).cpu().numpy().tolist()
            user_topk_list.append(recommends)

    return user_topk_list


def rbk(source, target, kernel_mul=2.0, kernel_num=5, fix_sigma=None):
    n_samples = int(source.size()[0]) + int(target.size()[0])
    total = torch.cat([source, target], dim=0)

    batch_size = 4096
    column_list = []
    for i_idx in range(0, n_samples, batch_size):
        i_end_idx = min(i_idx + batch_size, n_samples)
        batch_total0 = total[i_idx:i_end_idx]
        row_list = []
        for j_idx in range(0, n_samples, batch_size):
            j_end_idx = min(j_idx + batch_size, n_samples)
            batch_total1 = total[j_idx:j_end_idx]

            dist = torch.cdist(batch_total1, batch_total0, p=2) ** 2
            row_list.append(dist)
        column_list.append(torch.cat(row_list, dim=0))
    L2_distance = torch.cat(column_list, dim=1)

    if fix_sigma:
        bandwidth = fix_sigma
    else:
        bandwidth = torch.sum(L2_distance.data) / (n_samples ** 2 - n_samples)

    bandwidth /= kernel_mul ** (kernel_num // 2)
    bandwidth_list = [bandwidth * (kernel_mul ** i) for i in range(kernel_num)]

    bandwidth_list = torch.tensor(bandwidth_list)
    bandwidth_expand = bandwidth_list.view(-1, 1, 1).to(L2_distance.device)

    L2_distance_expand = L2_distance.view(1, L2_distance.size(0), L2_distance.size(1))
    kernel_val = torch.exp(-L2_distance_expand / bandwidth_expand).sum(dim=0)

    return kernel_val




def mmd_loss(source, target, kernel_mul=2.0, kernel_num=5, fix_sigma=None):
    source_dim = int(source.size(0))
    kernels = rbk(source, target, kernel_mul=kernel_mul, kernel_num=kernel_num, fix_sigma=fix_sigma)

    XX = kernels[:source_dim, :source_dim]
    YY = kernels[source_dim:, source_dim:]
    XY = kernels[:source_dim, source_dim:]
    YX = kernels[source_dim:, :source_dim]
    loss = torch.mean(XX) + torch.mean(YY) - torch.mean(XY) - torch.mean(YX)
    return loss


def marginal_loss(model, topk_items, top_k, user_num, marginal_lambda, device, batch_users=[]):
    if len(batch_users) == 0:
        all_user = [[i for i in range(user_num)]]
        items = torch.tensor(topk_items, dtype=torch.long).to(device)
    else:
        all_user = [batch_users]
        items = torch.tensor(topk_items[batch_users, :], dtype=torch.long).to(device)

    new_t = torch.tensor(all_user, dtype=torch.long).to(device)
    new_t = new_t.t()
    expanded_tensor = new_t.repeat(1, top_k)
    users = expanded_tensor

    pred = model(users, items)
    pred = torch.reshape(pred, (len(all_user[0]), top_k))

    loss_x = 0
    loss_per_id = torch.zeros((user_num)).to(device)

    for n_index in range(top_k - 1):
        margin = pred[:, n_index + 1] - pred[:, n_index] + marginal_lambda
        loss_per_id += torch.max(torch.zeros_like(margin), margin)
        loss_x += torch.sum(torch.max(torch.zeros_like(margin), margin))

    return loss_x, loss_per_id


def d2d(args):
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
    for epoch in tqdm(range(500)):
        model.train()
        start_time = time.time()

        optimizer.zero_grad()
        loss_defense = 0
        user_embedding = model.get_require_grad_user_embedding()
        attribute_user_anchor = attribute_user_list[0]
        for j in range(1, len(attribute_user_list)):
            loss_defense += mmd_loss(
                user_embedding(torch.tensor(attribute_user_list[j], dtype=torch.int32, device=device)),
                user_embedding(torch.tensor(attribute_user_anchor, dtype=torch.int32, device=device))
            )

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

    args.logger.info(f"total_time: "+ time.strftime("%H: %M: %S", time.gmtime(total_time)))
    args.logger.info(f"End. epoch {best_epoch:03d}: HR = {best_hr:.4f}, NDCG = {best_ndcg:.4f}")
    torch.save(model, os.path.join(construct_weight_path(args), f'final_model.pth'))

def retrain(args):
    device = torch.device(args.device)
    train_loader, test_loader, user_num, item_num = prepare_data(args)
    model = prepare_rec_model(user_num, item_num, train_loader.dataset, args)

    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    loss_function = nn.BCEWithLogitsLoss()
    user_attribute_dict = get_label_dict(args)
    attribute_user_list = []

    label_set = set(user_attribute_dict.values())
    for _, label_name in enumerate(label_set):
        label_list = []
        for x in range(len(user_attribute_dict)):
            if user_attribute_dict[x] == label_name:
                label_list.append(x)
        attribute_user_list.append(np.array(label_list))

    count, best_hr, best_ndcg = 0, 0, 0
    total_time = 0
    for epoch in range(args.epochs):
        model.train()
        start_time = time.time()
        train_loader.dataset.ng_sample()

        for user, item, label in train_loader:
            user = user.to(device)
            item = item.to(device)
            label = label.float().to(device)

            model.zero_grad()
            prediction = model(user, item)
            loss = loss_function(prediction, label)
            if count % 50 == 0:
                attribute_user_anchor = attribute_user_list[0]
                user_embedding = model.get_require_grad_user_embedding()
                loss_defense = 0
                for j in range(1, len(attribute_user_list)):
                    loss_defense += mmd_loss(
                        user_embedding(torch.tensor(attribute_user_list[j], dtype=torch.int32, device=device)),
                        user_embedding(torch.tensor(attribute_user_anchor, dtype=torch.int32, device=device))
                    )
                total_loss = loss + args.retrain_trade_off * loss_defense
            else:
                total_loss = loss
            total_loss.backward()
            optimizer.step()
            count += 1
        elapsed_time = time.time() - start_time
        total_time += elapsed_time

        model.eval()
        hr_list, ndcg_list = top100_metrics(model, test_loader)
        hrAT10, ndcgAT10 = hr_list[1], ndcg_list[1]
        args.logger.info(
            f"The time elapse of epoch {epoch:03d} is " + time.strftime("%H: %M: %S", time.gmtime(elapsed_time)))
        for idx, topk in zip([0, 1, 2, 3], [5, 10, 15, 20]):
            args.logger.info(f"HR@{topk}: {hr_list[idx]:.4f}\tNDCG@{topk}: {ndcg_list[idx]:.4f}")

        if hrAT10 > best_hr:
            best_hr, best_ndcg, best_epoch = hrAT10, ndcgAT10, epoch
            save_path = construct_weight_path(args)
            torch.save(model, os.path.join(save_path, f'epoch{epoch}.pth'))

    args.logger.info(f"total_time: "+ time.strftime("%H: %M: %S", time.gmtime(total_time)))
    args.logger.info(f"End. epoch {best_epoch:03d}: HR = {best_hr:.4f}, NDCG = {best_ndcg:.4f}")
    best_model = os.path.join(save_path, f'epoch{best_epoch}.pth')
    shutil.copy(best_model, os.path.join(save_path, f'final_model.pth'))