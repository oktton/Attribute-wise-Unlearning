# Attribute-wise-Unlearning

With the growing privacy concerns in recommender systems, recommendation unlearning, i.e., forgetting the impact of specific learned targets, is getting increasing attention. Existing studies predominantly use training data, i.e., model inputs, as the unlearning target. However, we find that attackers can extract private information,
i.e., gender, race, and age, from a trained model even if it has not been explicitly encountered during training. We name this unseen information as attribute and treat it as the unlearning target. To protect the sensitive attribute of users, Attribute Unlearning (AU) aims to degrade attacking performance and make target attributes indistinguishable. In this paper, we focus on a strict but practical setting of AU, namely Post-Training Attribute Unlearning (PoTAU), where unlearning can only be performed after the training of the recommendation model is completed. To address the PoT-AU problem in recommender systems, we design a two-component loss function that consists of i) distinguishability loss: making attribute labels indistinguishable from attackers, and ii) regularization loss: preventing drastic changes in the model that result in a negative impact on recommendation performance. Specifically, we investigate
two types of distinguishability measurements, i.e., user-to-user and      distribution-to-distribution. We use the stochastic gradient descent algorithm to optimize our proposed loss. Extensive experiments on three real-world datasets demonstrate the effectiveness of our proposed methods.

[Making Users Indistinguishable: Attribute-wise Unlearning in Recommender Systems](https://arxiv.org/abs/2310.05847)  
Yuyuan Li, Chaochao Chen, Xiaolin Zheng, Yizhao Zhang, Zhongxuan Han, Dan Meng, Jun Wang 
ACM International Conference on Multimedia (ACM MM), 2023

## Prerequisites 
* python==3.10.9
* numpy==1.24.3
* torch==2.6.0+cu118
* xgboost==2.1.2
* scikit-learn==1.5.2
* pandas==2.2.3
* scipy==1.14.1
* tqdm==4.66.6

## Getting Started
1. Clone this repo:  
```
   git clone https://github.com/oktton/Attribute-wise-Unlearning  
   cd Attribute-wise-Unlearning
``` 
2. Create a Virtual Environment  
```
   conda create -n AU python=3.10.9
   conda activate AU
```
3. Install all the dependencies  
```
   pip install -r requirements.txt
```
## Dataset
The complete source code, including datasets, is available on both [BaiduCloud](https://pan.baidu.com/s/1Clq7_lFf5D1Di_y4pJKkLA?pwd=bqru)  and [Google Drive](https://drive.google.com/file/d/1Ffe7Vv4pI4Icz2vyj2PqUtSUTBfGVWOl/view).


## Parameters
The source code contains two configuration files: `exp_config.json` and `attack_config.json`. The parameters in `exp_config.json` are as follow. You only need to modify `method`, `model`, `dataset`, `device`, `au_trade_off` and `retrain_trade_off`.
```
{
  "target_model": "final_model.pth",
  "method": "original",
  "model": "ncf",
  "lr": 0.001,
  "batch_size": 256,
  "epochs": 25,
  "dataset_path": "data/datasets/",
  "dataset": "ml-100k",
  "num_ng": 4,
  "test_num_ng": 99,
  "device": "cuda:0",
  "au_trade_off": 1e-6,
  "retrain_trade_off": 1
}
```
The parameters in `attack_config.json` are as follow. You only need to modify `method`, `model`, `dataset` and `device`.
```
{
  "attack": "attack",
  "target_model": "final_model.pth",
  "method": "original",
  "model": "ncf",
  "dataset_path": "data/datasets/",
  "dataset": "ml-100k",
  "device": "cuda:0"
}
```

## Train 
After modifying the `exp_config.json` file, run the following command to start training:
```
python main.py
```

## Attack
After training, modify the `attack_config.json` file and run the following command to conduct attack experiments:
```
python attack.py
```
## Citation
If you find our code/models useful, please consider citing our paper: 
```
@inproceedings{li2023making,
  title={Making users indistinguishable: Attribute-wise unlearning in recommender systems},
  author={Li, Yuyuan and Chen, Chaochao and Zheng, Xiaolin and Zhang, Yizhao and Han, Zhongxuan and Meng, Dan and Wang, Jun},
  booktitle={Proceedings of the 31st ACM International Conference on Multimedia},
  pages={984--994},
  year={2023}
}
```