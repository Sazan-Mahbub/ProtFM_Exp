# -*- encoding:utf8 -*-


import os
import time
import pickle
import torch as t
import numpy as np
from torch.utils import data
import gzip
from time import time
# my lib
from utils.config import DefaultConfig
import torch
import dgl
import threading

class dataSet(data.Dataset):
    def __init__(self, window_size, protBert_file=None, label_file=None, protein_list_file=None):
        super(dataSet, self).__init__()
        """
                All mean and variance for benchmark_dataset
                array elements: average_dist, C_alpha_dist, std_of_dist, relative_surface_angle
                mean_from_all_sets = [31.2310132,  1.6107959,  30.97167031,  1.55963611]
                std_from_all_sets = [16.54872725,  0.46084827, 16.62007193,  0.69138049]
                mean_from_train_set = [31.83509173  1.61227208 31.581271    1.56021911] 
                std_from_train_set = [16.79204272  0.4606557  16.86194361  0.69076342]
                """
        self.edge_feat_mean = [31.83509173, 1.56021911]
        self.edge_feat_std = [16.79204272, 0.69076342]

        self.all_protBert_feature = []
        for pb_file in protBert_file:
            with gzip.open(pb_file, "rb") as fp_pb:
                temp_pb = pickle.load(fp_pb)[pb_file.split('.')[0].split('/')[-1]]
            self.all_protBert_feature.extend(temp_pb)
        print('protein_list_file:', protein_list_file)
        self.dset = ''
        if 'dset159' in protein_list_file:
            self.all_dist_matrix = pickle.load(gzip.open('dset159_ppisp_dist_matrix_map.pkl.gz', 'rb'))
            self.all_angle_matrix = pickle.load(gzip.open('dset159_ppisp_angle_matrix_map2.pkl.gz', 'rb'))
            print('dist matrix and angle matrix of dset159 loaded.')
            self.dset = 'dset159'
        else:
            self.all_dist_matrix = pickle.load(gzip.open('ppisp_dist_matrix_map.pkl.gz', 'rb'))
            self.all_angle_matrix = pickle.load(gzip.open('ppisp_angle_matrix_map.pkl.gz', 'rb'))

        self.all_label = []
        for lab_file in label_file:
            with open(lab_file, "rb") as fp_label:
                temp_label = pickle.load(fp_label)
            self.all_label.extend(temp_label)

        with open(protein_list_file, "rb") as list_label:
            self.protein_list = pickle.load(list_label)

        self.Config = DefaultConfig()
        self.max_seq_len = self.Config.max_sequence_length
        self.window_size = window_size
        self.zero_vector = [0 for i in range(1024)]
        self.neighbourhood_size = 21

        # self.all_graphs = self.generate_all_graphs()
        # print(threading.get_ident(), ': All graphs generated.')

    def __getitem__(self, index):
        t0=time()
        # print('index:', index)
        count, id_idx, ii, dset, protein_id, seq_length = self.protein_list[index]
        # window_size = self.window_size
        id_idx = int(id_idx)
        # win_start = ii - window_size
        # win_end = ii + window_size
        seq_length = int(seq_length)
        label_idx = ii#(win_start + win_end) // 2

        _all_protBert_feature_ = self.all_protBert_feature[id_idx][:self.max_seq_len]

        if self.dset != 'dset159':
            if id_idx in {67, 129, 184, 223}:
                _all_protBert_feature_ = self.get_cleaned_features(_all_protBert_feature_, id_idx)
            # _all_protBert_feature_ = _all_protBert_feature_.tolist()
        seq_len = _all_protBert_feature_.shape[0]
        if seq_len < self.max_seq_len:
            temp = np.zeros([self.max_seq_len, _all_protBert_feature_.shape[1]])
            temp[:seq_len, :] = _all_protBert_feature_
            _all_protBert_feature_ = temp

        local_features = None#[np.zeros([1]) for _ in range(1)] # dummy, to keep consistency in code

        label = self.all_label[id_idx][label_idx]
        label = np.array(label, dtype=np.float32)

        label_idx_onehot = np.zeros([self.max_seq_len])
        label_idx_onehot[label_idx] = 1

        # _all_protBert_feature_ = np.stack(_all_protBert_feature_)
        _all_protBert_feature_ = _all_protBert_feature_[np.newaxis, :, :]

        # local_features = np.stack(local_features)
        # local_features = local_features[np.newaxis, :, :]

        label_idx_onehot = label_idx_onehot[np.newaxis, :]

        # G = self.all_graphs[id_idx]
        # print(torch.from_numpy(np.array([src, dst])).long())
        # if time() - t0 > .05:
        #     print('Batch generation time:', time() - t0)
        # print(G.adj())#; exit()
        temp__ = np.array([
            self.all_dist_matrix[id_idx]['dist_matrix'][:self.max_seq_len, :self.max_seq_len, 0],
            self.all_angle_matrix[id_idx]['angle_matrix'][:self.max_seq_len, :self.max_seq_len]
        ])

        edge_feat = 1_000_000 * np.ones([2, self.max_seq_len, self.max_seq_len]) 
        edge_feat[:, :temp__.shape[1], :temp__.shape[2]] = temp__
        
        return torch.from_numpy(_all_protBert_feature_).type(torch.FloatTensor), \
               local_features, \
               torch.from_numpy(label).type(torch.FloatTensor), \
               torch.from_numpy(np.array(label_idx_onehot)).long(), \
               torch.from_numpy(edge_feat).type(torch.FloatTensor) # torch.from_numpy(np.array([src, dst])).long()


    def __len__(self):
        return len(self.protein_list)

    def get_cleaned_features(self, protbert_features, idx):
        if idx == 67:
            protbert_features = protbert_features[2:]
        elif idx == 129:
            protbert_features = protbert_features[1:]
        elif idx == 184:
            protbert_features = protbert_features[1:]
        elif idx == 223:
            protbert_features = protbert_features[list(range(87))+list(range(88,123))+list(range(125,192))]
        return protbert_features

    def generate_all_graphs(self):
        graph_list = {}
        for id_idx in self.all_dist_matrix:
            G = dgl.DGLGraph()
            G.add_nodes(self.max_seq_len)
            neighborhood_indices = self.all_dist_matrix[id_idx]['dist_matrix'][:self.max_seq_len, :self.max_seq_len, 0] \
                                       .argsort()[:, 1:self.neighbourhood_size]
            if neighborhood_indices.max() > 499 or neighborhood_indices.min() < 0:
                print(neighborhood_indices.max(), neighborhood_indices.min())
                raise
            edge_feat = np.array([
                self.all_dist_matrix[id_idx]['dist_matrix'][:self.max_seq_len, :self.max_seq_len, 0],
                self.all_angle_matrix[id_idx]['angle_matrix'][:self.max_seq_len, :self.max_seq_len]
            ])
            edge_feat = np.transpose(edge_feat, (1, 2, 0))
            edge_feat = (edge_feat - self.edge_feat_mean) / self.edge_feat_std  # standardize features

            self.add_edges_custom(G,
                                  neighborhood_indices,
                                  # self.all_dist_matrix[id_idx]['dist_matrix'][:self.max_seq_len, :self.max_seq_len, [0, 1, 2]]
                                  edge_feat
                                  )
            graph_list[id_idx]= G
        return  graph_list

    def add_edges_custom(self, G, neighborhood_indices, edge_features):
        t1 = time()
        size = neighborhood_indices.shape[0]
        neighborhood_indices = neighborhood_indices.tolist()
        issues_in_nbr_data = []
        # for _ in range(32*12):
        src = []
        dst = []
        temp_edge_features = []
        for center in range(size):
            src += neighborhood_indices[center]
            dst += [center] * (self.neighbourhood_size - 1)
            for nbr in neighborhood_indices[center]:
                temp_edge_features += [np.abs(edge_features[center, nbr])]
        if len(src) != len(dst):
            print('src dst', len(src), len(dst))
            raise
        # print(threading.get_ident(), '>',time()-t1)
        t1 = time()
        # for _ in range(32*12):
        G.add_edges(src, dst)
        G.edata['ex'] = torch.tensor(np.array(temp_edge_features)) ##np.array(temp_edge_features)
        

def graph_collate(samples):
    # The input `samples` is a list of pairs
    #  (graph, label).
    protbert_data, local_data, label, label_idx_onehot, distmap_batch = map(list, zip(*samples)) 
    # print(protbert_data[0].shape, local_data[0].shape, label[0].shape, label_idx_onehot[0].shape)
    distmap_batch = torch.stack(distmap_batch)
    # print('>', [x.size() for x in label_idx_onehot])
    protbert_data = torch.cat(protbert_data)
    # print('protbert_data:',protbert_data.shape)
    # local_data = torch.tensor(local_data)
    # print(local_data.shape)
    label = torch.tensor(label)
    # print('label:',label.shape)
    label_idx_onehot = torch.cat(label_idx_onehot)
    # print(label_idx_onehot.shape)
    
    return protbert_data, local_data, label, label_idx_onehot, distmap_batch

def graph_collate_legacy(samples):
    # The input `samples` is a list of pairs
    #  (graph, label).
    protbert_data, local_data, label, label_idx_onehot, graph_batch = map(list, zip(*samples))
    # print(protbert_data[0].shape, local_data[0].shape, label[0].shape, label_idx_onehot[0].shape)
    # print(graph_batch[0])
    graph_batch = dgl.batch(graph_batch)
    # print(graph_batch)
    # print('>', [x.size() for x in label_idx_onehot])
    protbert_data = torch.cat(protbert_data)
    # print('protbert_data:',protbert_data.shape)
    # local_data = torch.tensor(local_data)
    # print(local_data.shape)
    label = torch.tensor(label)
    # print('label:',label.shape)
    label_idx_onehot = torch.cat(label_idx_onehot)
    # print(label_idx_onehot.shape)
    # graph_batch

    return protbert_data, local_data, label, label_idx_onehot, graph_batch

