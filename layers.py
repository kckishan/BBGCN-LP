import torch
from torch import nn
from torch_sparse import spmm


class GraphConvolution(nn.Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """

    def __init__(self, in_features, out_features, dropout_rate=0.1):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight_matrix = torch.nn.Parameter(torch.Tensor(self.in_features, self.out_features))
        self.bias = torch.nn.Parameter(torch.Tensor(1, self.out_features))
        self.dropout = dropout_rate
        self.act_layer_fn = nn.ELU()
        self.norm_layer = nn.LayerNorm(self.out_features)

        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.xavier_uniform_(self.weight_matrix)
        torch.nn.init.xavier_uniform_(self.bias)

    def forward(self, normalized_adjacency_matrix, features, mask=None):
        """
        Doing a forward pass.
        :param normalized_adjacency_matrix: Normalized adjacency matrix.
        :param features: Feature matrix.
        :return base_features: Convolved features.
        :return mask: binary mask.
        """
        base_features = torch.mm(features, self.weight_matrix)
        base_features = spmm(normalized_adjacency_matrix["indices"],
                             normalized_adjacency_matrix["values"],
                             base_features.shape[0],
                             base_features.shape[0],
                             base_features)
        base_features = self.act_layer_fn(base_features + self.bias)
        base_features = self.norm_layer(base_features)
        return base_features

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'


class BBConv(nn.Module):
    def __init__(self, in_features, out_features):
        super(BBConv, self).__init__()

        self.in_features = in_features
        self.out_features = out_features
        self.weight_matrix = torch.nn.Parameter(torch.Tensor(self.in_features, self.out_features))
        self.bias = torch.nn.Parameter(torch.Tensor(1, self.out_features))

        self.act_layer_fn = nn.ELU()
        self.norm_layer = nn.LayerNorm(self.out_features)

        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.xavier_uniform_(self.weight_matrix)
        torch.nn.init.xavier_uniform_(self.bias)

    def forward(self, adj, features):
        if len(features.size()) == 3:
            B, S, D = features.shape
            base_features = torch.einsum('nsf, fg -> nsg', features, self.weight_matrix)  # [n * s * f] x [f * g] => [n * g]
            base_features = base_features.view(B, -1)

            base_features = spmm(adj["indices"],
                                 adj["values"],
                                 base_features.shape[0],
                                 base_features.shape[0],
                                 base_features)

            base_features = base_features.view(B, S, self.out_features)

        elif len(features.size()) == 2:
            base_features = torch.mm(features, self.weight_matrix)
            base_features = spmm(adj["indices"],
                                 adj["values"],
                                 base_features.shape[0],
                                 base_features.shape[0],
                                 base_features)

        base_features = self.act_layer_fn(base_features + self.bias)
        base_features = self.norm_layer(base_features)
        return base_features


class GraphConvBlock(nn.Module):
    """
    Mask the output of fully connected layer with binary vectors from Beta-Bernoulli prior and add residual
    """

    def __init__(self, in_neurons, out_neurons, dropout=0.5, residual=False):
        super(GraphConvBlock, self).__init__()

        self.conv1 = BBConv(in_neurons, out_neurons)
        self.residual = residual

    def forward(self, adj, x, mask=None, num_samples=5):
        # adj: N x N
        # x: N x S x F

        residual = x
        out = self.conv1(adj, x)

        if len(out.size()) == 2:
            out = out.unsqueeze(1).expand(-1, num_samples, -1)

        if mask is not None:
            out = out * mask.unsqueeze(0)

        if self.residual:
            out += residual

        return out
