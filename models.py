import torch
from torch import nn
from layers import GraphConvolution, GraphConvBlock
from torch.nn import functional as F
from ArchitectureSampler import SampleNetworkArchitecture
from time import time


class GCN(nn.Module):
    def __init__(self, args, nfeat, nhid, z_dim,  dropout=0.5, device=None):
        super(GCN, self).__init__()

        self.args = args
        self.dropout = dropout
        self.layers = nn.ModuleList([GraphConvolution(nfeat, nhid, args.dropout)])

        # we simply add K layers at initialization
        # Note that we can also dynamically add new layers based on the inferred depth
        for i in range(args.n_layers):
            self.layers.append(GraphConvolution(nhid, nhid, args.dropout))

        self.bilinear = nn.Bilinear(nhid, nhid, z_dim)
        self.decoder = nn.Sequential(nn.Linear(z_dim, z_dim),
                                     nn.ELU(),
                                     nn.Linear(z_dim, 1)
                                     )

    def encode(self, adj, x):
        for i, layer in enumerate(self.layers):
            residual = x
            x = layer(adj, x)
            if i > 0:
                x = x + residual
        x = F.dropout(x, self.dropout, training=True)
        return x

    def decode(self, nodeA, nodeB):
        feat = F.elu(self.bilinear(nodeA, nodeB))
        predictions = self.decoder(feat)
        return predictions

    def forward(self, normalized_adjacency_matrix, features, idx):
        """
        Forward pass.
        :param normalized adjacency_matrix: Target matrix as a dict with indices and values.
        :param features: Feature matrix.
        :return predictions: Label predictions.
                latent_features: latent representations of nodes
        """
        latent_features = self.encode(normalized_adjacency_matrix, features)
        feat_p1 = latent_features[idx[0]]
        feat_p2 = latent_features[idx[1]]
        predictions = self.decode(feat_p1, feat_p2)
        return predictions, latent_features


class BBGCN(nn.Module):
    def __init__(self, args, nfeat, nhid, z_dim,  dropout=0.5, device=None):
        super(BBGCN, self).__init__()

        self.args = args
        self.device = device
        self.dropout = dropout
        self.architecture_sampler = SampleNetworkArchitecture(args, self.device)

        self.layers = nn.ModuleList([GraphConvBlock(nfeat, nhid, args.dropout).to(self.device)])

        # we simply add K layers at initialization
        # Note that we can also dynamically add new layers based on the inferred depth
        for i in range(self.args.truncation):
            self.layers.append(GraphConvBlock(nhid, nhid, args.dropout, residual=True).to(self.device))

        self.bilinear = nn.Bilinear(nhid, nhid, z_dim)
        self.decoder = nn.Sequential(nn.Linear(z_dim, z_dim),
                                     nn.ELU(),
                                     nn.Linear(z_dim, 1)
                                     )

    def _forward(self, x, adj, mask_matrix, threshold):
        """

        Parameters
        ----------
        x : input data matrix
        mask_matrix : matrix that corresponds to latent neural structure from Beta-Bernoulli Process
        threshold : number of layers in sampled architecture

        Returns
        -------
        out : Output from the sampled architecture
        """
        if not self.training and threshold > len(self.layers):
            threshold = len(self.layers)

        for k in range(threshold):
            mask = mask_matrix[:, :, k]
            x = self.layers[k](adj, x, mask)

        return x

    def encode(self, adj, features, num_samples=5):
        Z, n_layers, thresholds = self.architecture_sampler(num_samples)
        act_vec = self._forward(features, adj, Z, n_layers)

        return act_vec

    def decode(self, nodeA, nodeB):
        feat = F.elu(self.bilinear(nodeA, nodeB))
        predictions = self.decoder(feat)
        return predictions

    def forward(self, normalized_adjacency_matrix, features, idx, num_samples=5):
        """
        Forward pass.
        :param normalized adjacency_matrix: Target matrix as a dict with indices and values.
        :param features: Feature matrix.
        :return predictions: Label predictions.
                latent_features: latent representations of nodes
        """
        latent_features = self.encode(normalized_adjacency_matrix, features, num_samples=num_samples)
        latent_features = latent_features.permute(1, 0, 2)
        feat_p1 = latent_features[:, idx[0]]
        feat_p2 = latent_features[:, idx[1]]
        predictions = self.decode(feat_p1, feat_p2)
        return predictions, latent_features


