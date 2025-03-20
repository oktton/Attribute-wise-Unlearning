import os
import time

import torch
from torch import nn
from torch.autograd import Function
import shutil

from data.data_utils import prepare_data, load_positive_dict, get_label_dict
from utils.evaluate import top100_metrics
from utils.prepare_models import prepare_rec_model
from utils.save_utils import construct_weight_path


class GRL_(Function):
    @staticmethod
    def forward(ctx, input, grad_scaling):
        ctx.grad_scaling = grad_scaling
        return input

    @staticmethod
    def backward(ctx, grad_output):
        return -ctx.grad_scaling * grad_output, None


class GradientReversalLayer(nn.Module):
    def __init__(self, grad_scaling):
        super().__init__()
        self.grad_scaling = grad_scaling

    def forward(self, input):
        grl = GRL_.apply
        # return grl(input=input, grad_scaling=self.grad_scaling)
        return grl(input, self.grad_scaling)

    def extra_repr(self) -> str:
        return f"grad_scaling={self.grad_scaling}"


class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x


class AdvModel(nn.Module):
    def __init__(self, grad_scaling, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.gradient_reversal = GradientReversalLayer(grad_scaling)
        self.mlp = MLP(input_dim, hidden_dim, output_dim)

    def forward(self, input_indices):
        x = self.gradient_reversal(input_indices)
        x = self.mlp(x)
        return x


def adv(args):
    device = torch.device(args.device)
    train_loader, test_loader, user_num, item_num = prepare_data(args)
    positive_dict = load_positive_dict(args)
    model = prepare_rec_model(user_num, item_num, train_loader.dataset, args)

    grad_scaling = 0.5
    num_class = 2
    adv_model = AdvModel(grad_scaling, model.get_embedding_dim(), hidden_dim=16, output_dim = num_class)

    model.to(device)
    adv_model.to(device)
    optimizer = torch.optim.Adam(list(model.parameters()) + list(adv_model.parameters()), lr=args.lr)

    rec_loss_function = nn.BCEWithLogitsLoss()
    adv_loss_function = nn.CrossEntropyLoss()

    user_attribute_dict = get_label_dict(args)
    category_to_index = {name: cnt for cnt, name in enumerate(set(user_attribute_dict.values()))}

    count, best_hr, best_ndcg = 0, 0, 0
    total_time = 0
    for epoch in range(args.epochs):
        model.train()
        adv_model.train()
        start_time = time.time()
        train_loader.dataset.ng_sample()
        adv_trade_off = 1


        epoch_rec_loss = 0.0
        epoch_adv_loss = 0.0

        for user, item, label in train_loader:
            user = user.to(device)
            item = item.to(device)
            label = label.float().to(device)

            optimizer.zero_grad()
            rec_prediction = model(user, item)
            rec_loss = rec_loss_function(rec_prediction, label)
            epoch_rec_loss += rec_loss.item() / len(user)

            user_embedding = model.get_user_embedding(user)
            adv_prediction = adv_model(user_embedding)
            attribute_label = [user_attribute_dict[u.cpu().numpy().item()] for u in user]
            one_hot_tensors = [
                torch.eye(len(category_to_index))[category_to_index[category]] for category in attribute_label
            ]
            attribute_label = torch.stack(one_hot_tensors).to(device)
            adv_loss = adv_loss_function(adv_prediction, attribute_label)
            epoch_adv_loss += adv_loss.item() / user_embedding.shape[0]

            total_loss = rec_loss + adv_trade_off * adv_loss
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
        args.logger.info(f"epoch rec loss = {epoch_rec_loss}, adv loss = {epoch_adv_loss}")
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