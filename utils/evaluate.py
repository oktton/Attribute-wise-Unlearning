import numpy as np
import torch


def hit(gt_item, pred_items):
    if gt_item in pred_items:
        return 1
    return 0


def ndcg(gt_item, pred_items):
    if gt_item in pred_items:
        index = pred_items.index(gt_item)
        return np.reciprocal(np.log2(index + 2))
    return 0


def top100_metrics(model, test_loader):
    topk_list = [5, 10, 15, 20]
    hr_5_list, hr_10_list, hr_15_list, hr_20_list = [], [], [], []
    ndcg_5_list, ndcg_10_list, ndcg_15_list, ndcg_20_list = [], [], [], []

    device = next(model.parameters()).device
    for user, item, label in test_loader:
        user = user.to(device)
        item = item.to(device)

        predictions = model(user, item)

        gt_item = item[0].item()
        _, indices = torch.topk(predictions, 100)
        for topk in topk_list:
            recommends = torch.take(item, indices[:topk]).cpu().numpy().tolist()
            eval(f'hr_{topk}_list').append(hit(gt_item, recommends))
            eval(f'ndcg_{topk}_list').append(ndcg(gt_item, recommends))

    hr_return, ndcg_return = [], []
    for topk in topk_list:
        hr_return.append(np.mean(eval(f'hr_{topk}_list')))
        ndcg_return.append(np.mean(eval(f'ndcg_{topk}_list')))
    return hr_return, ndcg_return

