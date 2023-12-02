import numpy as np
import scipy.sparse as sp
import torch
from torch.utils import data
import pandas as pd
from texttable import Texttable



def tab_printer(args):
    """
    Function to print the logs in a nice tabular format.
    :param args: Parameters used for the model.
    """
    args = vars(args)
    keys = sorted(args.keys())
    t = Texttable()
    t.add_rows([["Parameter", "Value"]])
    t.add_rows([[k.replace("_", " ").capitalize(), args[k]] for k in keys])
    print(t.draw())

# borrowed from https://github.com/benedekrozemberczki/MixHop-and-N-GCN
def normalize(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx


def normalize_adjacency_matrix(A, I):
    """
    Creating a normalized adjacency matrix with self loops.
    :param A: Sparse adjacency matrix.
    :param I: Identity matrix.
    :return A_tile_hat: Normalized adjacency matrix.
    """
    A_tilde = A + 2 * I
    degrees = A_tilde.sum(axis=0)[0].tolist()
    D = sp.diags(degrees, [0])
    D = D.power(-0.5)
    A_tilde_hat = D.dot(A_tilde).dot(D)
    return A_tilde_hat


def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)


def create_propagator_matrix(A, device, explain=False):
    """
    Creating a propagator matrix.
    :param graph: NetworkX graph.
    :return propagator: Dictionary of matrix indices and values.
    """
    I = sp.eye(A.shape[0])
    A_tilde_hat = normalize_adjacency_matrix(A, I)
    propagator = dict()
    A_tilde_hat = sp.coo_matrix(A_tilde_hat).astype(np.float32)
    ind = np.concatenate([A_tilde_hat.row.reshape(-1, 1), A_tilde_hat.col.reshape(-1, 1)], axis=1)
    propagator["indices"] = torch.LongTensor(ind.T).to(device)
    propagator["values"] = torch.FloatTensor(A_tilde_hat.data).to(device)
    if explain:
        return A, propagator
    else:
        return propagator


def adj_to_propagation_matrix(edge_index, values, device):
    propagator = dict()
    propagator["indices"] = edge_index.to(device)
    propagator["values"] = values.to(device)
    return propagator


def features_to_sparse(features, device):
    """
    Reading the feature matrix stored as JSON from the disk.
    :param path: Path to the JSON file.
    :return out_features: Dict with index and value tensor.
    """
    index_1, index_2 = features.nonzero()
    values = [1.0]*len(index_1)
    node_count = features.shape[0]
    feature_count = features.shape[1]
    features = sp.coo_matrix((values, (index_1, index_2)),
                                 shape=(node_count, feature_count),
                                 dtype=np.float32)
    out_features = dict()
    ind = np.concatenate([features.row.reshape(-1, 1), features.col.reshape(-1, 1)], axis=1)
    out_features["indices"] = torch.LongTensor(ind.T).to(device)
    out_features["values"] = torch.FloatTensor(features.data).to(device)
    out_features["dimensions"] = features.shape
    return out_features

# borrowed from https://github.com/kexinhuang12345/SkipGNN
class Data_DDI(data.Dataset):
    # df : a list of data, which includes an index for the pair, an index for entity1 and entity2, from a list that combines all the entities. we want the
    def __init__(self, idx_map, labels, df):
        'Initialization'
        self.labels = labels
        self.idx_map = idx_map
        self.df = df

    def __len__(self):
        'Denotes the total number of samples'
        return len(self.df)

    def __getitem__(self, index):
        'Generates one sample of data'
        # Select sample
        # Load data and get label
        idx1 = self.idx_map[self.df.iloc[index].Drug1_ID]
        idx2 = self.idx_map[self.df.iloc[index].Drug2_ID]
        y = self.labels[index]
        return y, (idx1, idx2)


class Data_PPI(data.Dataset):
    # df : a list of data, which includes an index for the pair, an index for entity1 and entity2, from a list that combines all the entities. we want the
    def __init__(self, idx_map, labels, df):
        'Initialization'
        self.labels = labels
        self.idx_map = idx_map
        self.df = df

    def __len__(self):
        'Denotes the total number of samples'
        return len(self.df)

    def __getitem__(self, index):
        'Generates one sample of data'
        # Select sample
        # Load data and get label
        idx1 = self.idx_map[self.df.iloc[index].Protein1_ID]
        idx2 = self.idx_map[self.df.iloc[index].Protein2_ID]
        y = self.labels[index]
        return y, (idx1, idx2)


class Data_DTI(data.Dataset):
    # df : a list of data, which includes an index for the pair, an index for entity1 and entity2, from a list that combines all the entities. we want the
    def __init__(self, idx_map, labels, df):
        'Initialization'
        self.labels = labels
        self.idx_map = idx_map
        self.df = df

    def __len__(self):
        'Denotes the total number of samples'
        return len(self.df)

    def __getitem__(self, index):
        'Generates one sample of data'
        # Select sample
        # Load data and get label
        idx1 = self.idx_map[self.df.iloc[index].Drug_ID]
        idx2 = self.idx_map[self.df.iloc[index].Protein_ID]
        y = self.labels[index]
        return y, (idx1, idx2)


class Data_GDI(data.Dataset):
    # df : a list of data, which includes an index for the pair, an index for entity1 and entity2, from a list that combines all the entities. we want the
    def __init__(self, idx_map, labels, df):
        'Initialization'
        self.labels = labels
        self.idx_map = idx_map
        self.df = df

    def __len__(self):
        'Denotes the total number of samples'
        return len(self.df)

    def __getitem__(self, index):
        'Generates one sample of data'
        # Select sample
        # Load data and get label
        idx1 = self.idx_map[str(self.df.iloc[index].Gene_ID)]
        idx2 = self.idx_map[self.df.iloc[index].Disease_ID]
        y = self.labels[index]
        return y, (idx1, idx2)

class Data_Ecoli(data.Dataset):
    # df : a list of data, which includes an index for the pair, an index for entity1 and entity2, from a list that combines all the entities. we want the
    def __init__(self, idx_map, labels, df):
        'Initialization'
        self.df = df
        self.idx_map = idx_map

    def __len__(self):
        'Denotes the total number of samples'
        return len(self.df)

    def __getitem__(self, index):
        'Generates one sample of data'
        # Select sample
        # Load data and get label
        idx1 = self.idx_map[self.df.iloc[index].Gene1_ID]
        idx2 = self.idx_map[self.df.iloc[index].Gene2_ID]
        y = self.df.iloc[index].label
        return y, (idx1, idx2)


def load_data_link_prediction_DDI(path, network_type, inp, device, explain=False):
    print('Loading DDI dataset...')
    path_up = f'./data/{network_type}'
    df_data = pd.read_csv(path + '/train.csv')
    df_drug_list = pd.read_csv(path_up + '/ddi_unique_smiles.csv')

    idx = df_drug_list['Drug1_ID'].tolist()
    idx = np.array(idx)
    idx_map = {j: i for i, j in enumerate(idx)}

    df_data_t = df_data[df_data.label == 1]
    edges_unordered = df_data_t[['Drug1_ID', 'Drug2_ID']].values

    if inp == 'node2vec':
        emb = pd.read_csv(path + '/ddi.emb', skiprows=1, header=None, sep=' ').sort_values(by=[0]).set_index([0])
        new_index = [idx_map[idx] for idx in emb.index]
        emb = emb.reindex(new_index)

        for i in np.setdiff1d(np.arange(1514), emb.index.values):
            emb.loc[i] = (np.sum(emb.values, axis=0) / emb.values.shape[0])

        features = emb.sort_index().values
        features = normalize(features)
    elif inp == 'one_hot':
        features = np.eye(1514)

    features =torch.FloatTensor(features).to(device)

    edges = np.array(list(map(idx_map.get, edges_unordered.flatten()))).reshape(edges_unordered.shape)
    adj = sp.coo_matrix((np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])),
                        shape=(len(idx), len(idx)),
                        dtype=np.float32)

    # build symmetric adjacency matrix
    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
    adj = create_propagator_matrix(adj, device, explain)

    return adj, features, idx_map


def load_data_link_prediction_PPI(path, network_type, inp, device, explain=False):
    print('Loading PPI dataset...')
    path_up = f'./data/{network_type}'
    df_data = pd.read_csv(path + '/train.csv')
    df_drug_list = pd.read_csv(path_up + '/protein_list.csv')

    idx = df_drug_list['Protein1_ID'].tolist()
    idx = np.array(idx)
    idx_map = {j: i for i, j in enumerate(idx)}

    df_data_t = df_data[df_data.label == 1]
    edges_unordered = df_data_t[['Protein1_ID', 'Protein2_ID']].values

    if inp == 'node2vec':
        emb = pd.read_csv(path + '/ppi.emb', skiprows=1, header=None, sep=' ').sort_values(by=[0]).set_index([0])
        new_index = [idx_map[idx] for idx in emb.index]
        emb = emb.reindex(new_index)

        for i in np.setdiff1d(np.arange(5604), emb.index.values):
            emb.loc[i] = (np.sum(emb.values, axis=0) / emb.values.shape[0])
        features = emb.sort_index().values
        features = normalize(features)
    elif inp == 'one_hot':
        features = np.eye(5604)

    features =torch.FloatTensor(features).to(device)


    edges = np.array(list(map(idx_map.get, edges_unordered.flatten()))).reshape(edges_unordered.shape)
    adj = sp.coo_matrix((np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])),
                        shape=(len(idx), len(idx)),
                        dtype=np.float32)

    # build symmetric adjacency matrix
    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
    adj = create_propagator_matrix(adj, device, explain)

    return adj, features, idx_map


def load_data_link_prediction_DTI(path, network_type, inp, device, explain=False):
    print('Loading DTI dataset...')
    path_up = f'./data/{network_type}'
    df_data = pd.read_csv(path + '/train.csv')
    df_drug_list = pd.read_csv(path_up + '/entity_list.csv')

    idx = df_drug_list['Entity_ID'].tolist()
    idx = np.array(idx)
    idx_map = {j: i for i, j in enumerate(idx)}

    df_data_t = df_data[df_data.label == 1]
    edges_unordered = df_data_t[['Drug_ID', 'Protein_ID']].values

    if inp == 'node2vec':
        emb = pd.read_csv(path + '/dti.emb', skiprows=1, header=None, sep=' ').sort_values(by=[0]).set_index([0])
        new_index = [idx_map[idx] for idx in emb.index]
        emb = emb.reindex(new_index)

        for i in np.setdiff1d(np.arange(7343), emb.index.values):
            emb.loc[i] = (np.sum(emb.values, axis=0) / emb.values.shape[0])
        features = emb.sort_index().values

        features = normalize(features)
    elif inp == 'one_hot':
        features = np.eye(7343)

    features =torch.FloatTensor(features).to(device)

    edges = np.array(list(map(idx_map.get, edges_unordered.flatten()))).reshape(edges_unordered.shape)
    adj = sp.coo_matrix((np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])),
                        shape=(len(idx), len(idx)),
                        dtype=np.float32)

    # build symmetric adjacency matrix
    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
    adj = create_propagator_matrix(adj, device, explain)
    return adj, features, idx_map


def load_data_link_prediction_GDI(path, network_type, inp, device, explain=False):
    print('Loading GDI dataset...')
    path_up = f'./data/{network_type}'
    df_data = pd.read_csv(path + '/train.csv')
    df_drug_list = pd.read_csv(path_up + '/entity_list.csv')
    idx = df_drug_list['Entity_ID'].tolist()
    idx = np.array(idx)
    idx_map = {j: i for i, j in enumerate(idx)}

    df_data_t = df_data[df_data.label == 1]
    df_data_t['Gene_ID'] = df_data_t['Gene_ID'].apply(str)
    edges_unordered = df_data_t[['Gene_ID', 'Disease_ID']].values

    if inp == 'node2vec':
        emb = pd.read_csv(path + '/gdi.emb', skiprows=1, header=None, sep=' ').sort_values(by=[0]).set_index([0])
        new_index = [idx_map[idx] for idx in emb.index]
        emb = emb.reindex(new_index)

        for i in np.setdiff1d(np.arange(19783), emb.index.values):
            emb.loc[i] = (np.sum(emb.values, axis=0) / emb.values.shape[0])
        features = emb.sort_index().values
        features = normalize(features)
    elif inp == 'one_hot':
        features = np.eye(19783)

    features =torch.FloatTensor(features).to(device)

    edges = np.array(list(map(idx_map.get, edges_unordered.flatten()))).reshape(edges_unordered.shape)

    adj = sp.coo_matrix((np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])),
                        shape=(len(idx), len(idx)),
                        dtype=np.float32)

    # build symmetric adjacency matrix
    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
    adj = create_propagator_matrix(adj, device, explain)

    return adj, features, idx_map


def load_data_link_prediction_Ecoli(path, network_type, inp, device, explain=False):
    print('Loading Ecoli dataset...')
    df_data = pd.read_csv(path + '/train.csv')

    idx = sorted(list(set(df_data.Gene1_ID.tolist() + df_data.Gene2_ID.tolist())))
    idx = np.array(idx)
    idx_map = {j: i for i, j in enumerate(idx)}

    df_data_t = df_data[df_data.label == 1]
    edges_unordered = df_data_t[['Gene1_ID', 'Gene2_ID']].values

    features = pd.read_csv(path + '/data_standard.txt', sep=" ").values
    features =torch.FloatTensor(features).to(device)

    edges = np.array(list(map(idx_map.get, edges_unordered.flatten()))).reshape(edges_unordered.shape)

    adj = sp.coo_matrix((np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])),
                        shape=(len(idx), len(idx)),
                        dtype=np.float32)

    # build symmetric adjacency matrix
    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
    adj = create_propagator_matrix(adj, device, explain)

    return adj, features, idx_map


def load_data(args, explain=False):
    data_path = f"./data/{args.network_type}/fold{args.fold_id}"
    if args.ratio:
        data_path = f"./data/{args.network_type}/{args.train_percent}/fold{args.fold_id}"

    if args.network_type == 'DDI':
        adj, features, idx_map = load_data_link_prediction_DDI(data_path, args.network_type, args.input_type, args.device, explain)
        Data_class = Data_DDI
    elif args.network_type == 'PPI':
        adj, features, idx_map = load_data_link_prediction_PPI(data_path, args.network_type, args.input_type, args.device, explain)
        Data_class = Data_PPI
    elif args.network_type == 'DTI':
        adj, features, idx_map = load_data_link_prediction_DTI(data_path, args.network_type, args.input_type, args.device, explain)
        Data_class = Data_DTI
    elif args.network_type == 'GDI':
        adj, features, idx_map = load_data_link_prediction_GDI(data_path, args.network_type, args.input_type, args.device, explain)
        Data_class = Data_GDI
    elif args.network_type == 'ecoli':
        adj, features, idx_map = load_data_link_prediction_Ecoli(data_path, args.network_type, args.input_type,
                                                               args.device, explain)
        Data_class = Data_Ecoli

    return adj, features, idx_map, Data_class