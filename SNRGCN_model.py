import pickle
import math
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.nn.parameter import Parameter
import random
# Path
Feature_Path = "./Feature/"

# Seed
SEED = 2020
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.set_device(0)
    torch.cuda.manual_seed(SEED)

# GraphPPIS parameters
EMBEDDING = "e"  # b for BLOSUM62; e for evolutionary features (PSSM+HMM)
MAP_TYPE = "d"  # d for discrete maps; c for continuous maps
MAP_CUTOFF = 14

INPUT_DIM = (34 if EMBEDDING == "b" else 69)
HIDDEN_DIM = 256
LAYER =8
DROPOUT = 0.1
ALPHA = 0.7
LAMBDA = 1.5
VARIANT = True  # From GCNII

LEARNING_RATE = 1E-3
WEIGHT_DECAY = 0
BATCH_SIZE = 1
NUM_CLASSES = 2  # [not bind, bind]
NUMBER_EPOCHS = 50
device = torch.device('cuda:7' if torch.cuda.is_available() else 'cpu')


def embedding(sequence_name, seq, embedding_type):
    if embedding_type == "b":
        seq_embedding = []
        Max_blosum = np.array([4, 5, 6, 6, 9, 5, 5, 6, 8, 4, 4, 5, 5, 6, 7, 4, 5, 11, 7, 4])
        Min_blosum = np.array([-3, -3, -4, -4, -4, -3, -4, -4, -3, -4, -4, -3, -3, -4, -4, -3, -2, -4, -3, -3])
        with open(Feature_Path + "blosum/blosum_dict.pkl", "rb") as f:
            blosum_dict = pickle.load(f)
        for aa in seq:
            seq_embedding.append(blosum_dict[aa])
        seq_embedding = (np.array(seq_embedding) - Min_blosum) / (Max_blosum - Min_blosum)
    elif embedding_type == "e":
        pssm_feature = np.load(Feature_Path + "pssm/" + sequence_name + '.npy')
        hmm_feature = np.load(Feature_Path + "hmm/" + sequence_name + '.npy')
        seq_embedding = np.concatenate([pssm_feature, hmm_feature], axis=1)
    return seq_embedding.astype(np.float32)


def get_dssp_features(sequence_name):
    dssp_feature = np.load(Feature_Path + "dssp/" + sequence_name + '.npy')
    return dssp_feature.astype(np.float32)


def norm_dis(mx):  # from SPROF
    return 2 / (1 + (np.maximum(mx, 4) / 4))


def normalize(mx):
    rowsum = np.array(mx.sum(1))
    r_inv = (rowsum ** -0.5).flatten()
    r_inv[np.isinf(r_inv)] = 0
    r_mat_inv = np.diag(r_inv)
    result = r_mat_inv @ mx @ r_mat_inv
    return result


def load_graph(sequence_name):
    dismap = np.load(Feature_Path + "distance_map/" + sequence_name + ".npy")
    mask = ((dismap >= 0) * (dismap <= MAP_CUTOFF))
    if MAP_TYPE == "d":
        adjacency_matrix = mask.astype(np.int)
    elif MAP_TYPE == "c":
        adjacency_matrix = norm_dis(dismap)
        adjacency_matrix = mask * adjacency_matrix
    norm_matrix = normalize(adjacency_matrix.astype(np.float32))
    return norm_matrix


def get_node_features(sequence_name):
    # phychem_feature = np.load(Feature_Path + "phychem/" + sequence_name + '.npy')
    psaia_feature = np.load(Feature_Path + "psaia/" + sequence_name + '.npy')
    dssp_feature = np.load(Feature_Path + "dssp/" + sequence_name + '.npy')
    pssm_feature = np.load(Feature_Path + "pssm/" + sequence_name + '.npy')
    hmm_feature = np.load(Feature_Path + "hmm/" + sequence_name + '.npy')
    RAA_feature = np.load(Feature_Path + "RAA/" + sequence_name + '.npy')
    pro2Vec_feature = np.load(Feature_Path + "pro2Vec/" + sequence_name + '.npy')
    hydrophobicity_feature = np.load(Feature_Path + "hyd/" + sequence_name + '.npy')
    anchor_feature = np.load(Feature_Path + "anchor/" + sequence_name + '.npy')
    #hydrophobicity_feature,
    node_features = np.concatenate(
        #[dssp_feature, pssm_feature,  RAA_feature, hmm_feature,  anchor_feature,pro2Vec_feature,psaia_feature], axis=1)
        [dssp_feature,pssm_feature, RAA_feature, hmm_feature, anchor_feature, pro2Vec_feature, psaia_feature], axis=1)
    #print("node",node_features.shape)
    return node_features

class ProDataset(Dataset):
    def __init__(self, dataframe):
        self.names = dataframe['ID'].values
        self.sequences = dataframe['sequence'].values
        self.labels = dataframe['label'].values

    def __getitem__(self, index):
        sequence_name = self.names[index]
        sequence = self.sequences[index]
        label = np.array(self.labels[index])

        sequence_embedding = embedding(sequence_name, sequence, EMBEDDING)
        structural_features = get_dssp_features(sequence_name)
        #node_features = np.concatenate([sequence_embedding, structural_features], axis = 1)
        node_features = get_node_features(sequence_name)
        graph = load_graph(sequence_name)

        return sequence_name, sequence, label, node_features, graph

    def __len__(self):
        return len(self.labels)


class GraphConvolution(nn.Module):
    def __init__(self, in_features, out_features, residual=False, variant=False):
        super(GraphConvolution, self).__init__()
        self.variant = variant
        if self.variant:
            self.in_features = 2 * in_features
        else:
            self.in_features = in_features

        self.out_features = out_features
        self.residual = residual
        self.weight = Parameter(torch.FloatTensor(self.in_features, self.out_features))
        self.weight1 = Parameter(torch.FloatTensor(self.in_features, self.out_features))
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.out_features)
        self.weight.data.uniform_(-stdv, stdv)
        self.weight1.data.uniform_(-stdv, stdv)
    def aggregate(self, x, adj):
        A = adj.cpu().detach().numpy()
        sample_num = 3
        mask = torch.zeros(adj.shape[0], adj.shape[0])  # .to(device)

        for index in range(A.shape[0]):
            count = list(np.flatnonzero(A[index]))
            #count.remove(index)  # 去除中心结点

            if len(count) >= sample_num:
                count = random.sample(count, sample_num)  # 采样
            else:
                count = np.random.choice(count, size=sample_num, replace=True)
            for ele in count:
                mask[index, ele] = 1
        num_neigh = mask.sum(1, keepdim=True)
        mask = mask.div(num_neigh).to(device)
        agg = mask.mm(x)
        #mask = mask.to(device)
        #localadj = torch.mul(adj, mask)
        return agg#localadj

    def forward(self, input, adj, h0, lamda, alpha, l):
        theta = min(1, math.log(lamda / l + 1))

        localadj = self.aggregate(input, adj)
        #localhi = torch.spmm(localadj, input)
        #localsupport = torch.cat([localhi, input], 1)
        r = (1 - alpha) * input + alpha * h0+0.1*localadj
        #localsupport = theta * torch.mm(localsupport, self.weight1)+(1 - theta) * r

        hi = torch.spmm(adj, input)

        support = torch.cat([hi, input], 1)
        output = theta * torch.mm(support, self.weight) + (1 - theta) *r
        return output

class deepGCN(nn.Module):
    def __init__(self, nlayers, nfeat, nhidden, nclass, dropout, lamda, alpha, variant):
        super(deepGCN, self).__init__()
        self.convs = nn.ModuleList()
        for _ in range(nlayers):
            self.convs.append(GraphConvolution(nhidden, nhidden, variant=variant, residual=True))
        self.fcs = nn.ModuleList()
        self.fcs.append(nn.Linear(nfeat, nhidden))
        #self.fcs.append(nn.Linear(nhidden,64))
        self.fcs.append(nn.Linear(nhidden, nclass))
        self.act_fn = nn.ReLU()
        self.dropout = dropout
        self.alpha = alpha
        self.lamda = lamda


    def forward(self, x, adj):
        _layers = []
        x = F.dropout(x, self.dropout, training=self.training)
        layer_inner = self.act_fn(self.fcs[0](x))
        #layer_inner = self.aggregate(layer_inner, adj)
        #layer_inner=torch.cat([layer_inner, agg], 1)
        #layer_inner=self.act_fn(layer_inner+agg)
        _layers.append(layer_inner)

        for i, con in enumerate(self.convs):
            layer_inner = F.dropout(layer_inner, self.dropout, training=self.training)
            layer_inner = self.act_fn(con(layer_inner, adj, _layers[0], self.lamda, self.alpha, i + 1))

        layer_inner = F.dropout(layer_inner, self.dropout, training=self.training)

        #layer_inner=torch.cat([layer_inner, agg], 1)
        layer_inner = self.fcs[-1](layer_inner)
        return layer_inner

class GraphPPIS(nn.Module):
    def __init__(self, nlayers, nfeat, nhidden, nclass, dropout, lamda, alpha, variant):
        super(GraphPPIS, self).__init__()

        self.deep_gcn = deepGCN(nlayers=nlayers, nfeat=nfeat, nhidden=nhidden, nclass=nclass,
                                dropout=dropout, lamda=lamda, alpha=alpha, variant=variant)
        self.criterion = nn.CrossEntropyLoss()  # automatically do softmax to the predicted value and one-hot to the label
        self.optimizer = torch.optim.Adam(self.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)

    def forward(self, x, adj):  # x.shape = (seq_len, FEATURE_DIM); adj.shape = (seq_len, seq_len)
        x = x.float()
        output = self.deep_gcn(x, adj)  # output.shape = (seq_len, NUM_CLASSES)
        return output


