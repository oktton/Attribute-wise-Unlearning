import torch
import torch.nn as nn


class NCF(nn.Module):
    def __init__(self, user_num, item_num, factor_num=32, num_layers=3, dropout=0.0):
        super(NCF, self).__init__()

        self.embed_user_gmf = nn.Embedding(user_num, factor_num)
        self.embed_item_gmf = nn.Embedding(item_num, factor_num)
        self.embed_user_mlp = nn.Embedding(user_num, factor_num * (2 ** (num_layers - 1)))
        self.embed_item_mlp = nn.Embedding(item_num, factor_num * (2 ** (num_layers - 1)))

        self.dropout = dropout
        mlp_modules = []
        for i in range(num_layers):
            input_size = factor_num * (2 ** (num_layers - i))
            mlp_modules.append(nn.Dropout(p=self.dropout))
            mlp_modules.append(nn.Linear(input_size, input_size // 2))
            mlp_modules.append(nn.ReLU())
        self.mlp_layers = nn.Sequential(*mlp_modules)

        predict_size = factor_num * 2
        self.predict_layer = nn.Linear(predict_size, 1)

        self.init_weight()

    def init_weight(self):
        nn.init.normal_(self.embed_user_gmf.weight, std=0.01)
        nn.init.normal_(self.embed_user_mlp.weight, std=0.01)
        nn.init.normal_(self.embed_item_gmf.weight, std=0.01)
        nn.init.normal_(self.embed_item_mlp.weight, std=0.01)

        for m in self.mlp_layers:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
        nn.init.kaiming_uniform_(self.predict_layer.weight, a=1, nonlinearity='sigmoid')

        for m in self.modules():
            if isinstance(m, nn.Linear) and m.bias is not None:
                m.bias.data.zero_()

    def forward(self, user, item):
        embed_user_gmf = self.embed_user_gmf(user)
        embed_item_gmf = self.embed_item_gmf(item)
        output_gmf = embed_user_gmf * embed_item_gmf

        embed_user_mlp = self.embed_user_mlp(user)
        embed_item_mlp = self.embed_item_mlp(item)
        interaction = torch.cat((embed_user_mlp, embed_item_mlp), -1)
        output_mlp = self.mlp_layers(interaction)

        concat = torch.cat((output_gmf, output_mlp), -1)
        prediction = self.predict_layer(concat)
        return prediction.view(-1)

    def get_embedding_dim(self):
        return self.embed_user_gmf.weight.shape[1]

    def get_user_embedding(self, users):
        return self.embed_user_gmf(users)

    def get_require_grad_user_embedding(self):
        return self.embed_user_gmf

    def get_user_embedding_weight(self):
        return self.embed_user_gmf.weight
