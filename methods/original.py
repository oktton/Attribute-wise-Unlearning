import os
import time
import shutil

import torch
import torch.nn as nn
import torch.optim as optim

from data.data_utils import prepare_data, load_positive_dict
from utils.evaluate import top100_metrics
from utils.prepare_models import prepare_rec_model
from utils.save_utils import construct_weight_path


def original(args):
    device = torch.device(args.device)
    train_loader, test_loader, user_num, item_num = prepare_data(args)
    positive_dict = load_positive_dict(args)
    model = prepare_rec_model(user_num, item_num, train_loader.dataset, args)

    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    loss_function = nn.BCEWithLogitsLoss()

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
            loss.backward()
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
    args.logger.info(f"End. Best epoch {best_epoch:03d}: HR = {best_hr:.4f}, NDCG = {best_ndcg:.4f}")
    best_model = os.path.join(save_path, f'epoch{best_epoch}.pth')
    shutil.copy(best_model, os.path.join(save_path, f'final_model.pth'))