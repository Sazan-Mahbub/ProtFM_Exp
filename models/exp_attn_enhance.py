# -*- encoding:utf8 -*-

import os
import time
import sys
import numpy as np

import torch
from torch import nn
from torch.autograd import Variable

# from basic_module import BasicModule
from models.BasicModule import BasicModule
from dgl.nn.pytorch.conv import GATConv

sys.path.append("../")
from utils.config import DefaultConfig

configs = DefaultConfig()


class ConvsLayer(BasicModule):
    def __init__(self, ):
        super(ConvsLayer, self).__init__()

        self.kernels = configs.kernels
        hidden_channels = configs.cnn_chanel
        in_channel = 1
        features_L = configs.max_sequence_length
        seq_dim = configs.seq_dim
        dssp_dim = configs.dssp_dim
        pssm_dim = configs.pssm_dim
        W_size = 32  # seq_dim + dssp_dim + pssm_dim

        padding1 = (self.kernels[0] - 1) // 2
        padding2 = (self.kernels[1] - 1) // 2
        padding3 = (self.kernels[2] - 1) // 2
        self.conv1 = nn.Sequential()
        self.conv1.add_module("conv1",
                              nn.Conv2d(in_channel, hidden_channels,
                                        padding=(padding1, 0),
                                        kernel_size=(self.kernels[0], W_size)))
        self.conv1.add_module("ReLU", nn.PReLU())
        self.conv1.add_module("pooling1", nn.MaxPool2d(kernel_size=(features_L, 1), stride=1))

        self.conv2 = nn.Sequential()
        self.conv2.add_module("conv2",
                              nn.Conv2d(in_channel, hidden_channels,
                                        padding=(padding2, 0),
                                        kernel_size=(self.kernels[1], W_size)))
        self.conv2.add_module("ReLU", nn.ReLU())
        self.conv2.add_module("pooling2", nn.MaxPool2d(kernel_size=(features_L, 1), stride=1))

        self.conv3 = nn.Sequential()
        self.conv3.add_module("conv3",
                              nn.Conv2d(in_channel, hidden_channels,
                                        padding=(padding3, 0),
                                        kernel_size=(self.kernels[2], W_size)))
        self.conv3.add_module("ReLU", nn.ReLU())
        self.conv3.add_module("pooling3", nn.MaxPool2d(kernel_size=(features_L, 1), stride=1))

    def forward(self, x):
        features1 = self.conv1(x)
        features2 = self.conv2(x)
        features3 = self.conv3(x)
        features = t.cat((features1, features2, features3), 1)
        shapes = features.data.shape
        features = features.view(shapes[0], shapes[1] * shapes[2] * shapes[3])

        return features


class ExpModel(BasicModule):
    
    def __init__(self, class_nums, window_size, ratio=None):
        super(ExpModel, self).__init__()
        global configs
        configs.kernels = [13, 15, 17]
        self.dropout = configs.dropout = 0.2

        seq_dim = configs.seq_dim * configs.max_sequence_length

        self.seq_layers = nn.Sequential()
        self.seq_layers.add_module("seq_embedding_layer",
                                   nn.Linear(seq_dim, seq_dim))
        self.seq_layers.add_module("seq_embedding_ReLU",
                                   nn.ReLU())

        seq_dim = configs.seq_dim
        dssp_dim = configs.dssp_dim
        pssm_dim = configs.pssm_dim
        local_dim = (window_size * 2 + 1) * (pssm_dim + dssp_dim + seq_dim)
        if ratio:
            configs.cnn_chanel = (local_dim * int(ratio[0])) // (int(ratio[1]) * 3)
        input_dim = configs.cnn_chanel * 3 + local_dim


        self.protbert_mlp =  nn.Sequential(
            nn.Linear(1024, 64),
            nn.Dropout(.5),
            nn.LeakyReLU())
        
        self.outLayer = nn.Sequential(
            nn.Linear(64, class_nums),
            nn.Sigmoid())
        
        
        self.outLayer_protbert = nn.Sequential(
            nn.Linear(64, class_nums),
            nn.Sigmoid())
        
        self.conv_encoder = nn.Sequential(
            nn.Conv1d(in_channels=1024,
                      out_channels=32,
                      kernel_size=7, stride=1,
                      padding=7 // 2, dilation=1, groups=1,
                      bias=True, padding_mode='zeros'),
            nn.LeakyReLU(negative_slope=.01),
            nn.BatchNorm1d(num_features=32,
                           eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.Dropout(.5)  # it was .2
        )

        # self.gat_layer = GATConv(in_feats=32,
        #                          out_feats=32,
        #                          num_heads=1,
        #                          feat_drop=0.0,
        #                          attn_drop=0.0,
        #                          negative_slope=0.2,
        #                          residual=False,
        #                          activation=None)
        from models import EdgeAggregatedGAT as eagat
        # from models import EdgeAggregatedGAT_attention_visual as eagat
        config_dict = eagat.config_dict
        config_dict['feat_drop'] = 0.2
        config_dict['edge_feat_drop'] = 0.1
        config_dict['attn_drop'] = 0.2

        # self.gat_layer = eagat.MultiHeadGATLayer(
        #                             in_dim=32,
        #                             out_dim=32,
        #                             edge_dim=2,
        #                             num_heads=1,
        #                             use_bias=False,
        #                             merge='cat',
        #                             config_dict=config_dict)
        
        
        
        
        self.Cosine_Similarity = nn.CosineSimilarity(dim=-1, eps=1e-6)
        self.experimental_transformer_layer = eagat.TransformerEncoderLayer(d_model=64, 
                                                                            nhead=2, 
                                                                            dim_feedforward=64, 
                                                                            dropout=0.1, 
                                                                            activation="relu")

        ### change the name to run different experiments. need to fix :-)
        # self.forward = self.forward_attn_reg 
        self.forward = self.forward_inject_struct 
        

    

    def forward_attn_reg(self, protbert_feature, local_features, label_idx_onehot, graph_batch):
        shapes = protbert_feature.data.shape
        protbert_feature = self.protbert_mlp(protbert_feature)
#         print('>>', protbert_feature.shape, graph_batch.shape); exit()

        graph_batch = 1 - torch.exp(-1*graph_batch[:, :, :, :])
#         attn_mask = (graph_batch[:, 0, :, :] != 1_000_000).bool()
#         attn_mask = torch.stack([attn_mask, attn_mask], 1)
#         attn_mask = attn_mask.view([-1, attn_mask.shape[2], attn_mask.shape[3]])
        protbert_feature, tx_attn = self.experimental_transformer_layer(protbert_feature)
        
        attention_loss = (tx_attn.mean(1)*graph_batch[:, 0, :, :]).sum() #... # loss(head_attn_scores[:, 0], graph_batch[:, 0]) + loss(head_attn_scores[:, 1], graph_batch[:, 1]) 
        
        protbert_feature = protbert_feature[torch.nonzero(label_idx_onehot.view([shapes[0], 500])==1, as_tuple=True)]
        
        protbert_output = self.outLayer_protbert(protbert_feature)
        
        return None, protbert_output, attention_loss, None ## quick fix :-) 
    
    
    def forward_inject_struct(self, protbert_feature, local_features, label_idx_onehot, graph_batch):
        shapes = protbert_feature.data.shape
        protbert_feature = self.protbert_mlp(protbert_feature)
        graph_batch = torch.exp(-1*graph_batch[:, 0, :, :])
        attn_mask = torch.stack([graph_batch,graph_batch], 1)
        attn_mask = attn_mask.view([-1, attn_mask.shape[2], attn_mask.shape[3]])
#         print(attn_mask.shape); exit()
        
        ## https://pytorch.org/docs/stable/generated/torch.nn.MultiheadAttention.html
        protbert_feature, tx_attn = self.experimental_transformer_layer(protbert_feature,
                                                                        attn_mask=attn_mask, 
                                                                        average_attn_weights=False)       
        attention_loss = None 
        protbert_feature = protbert_feature[torch.nonzero(label_idx_onehot.view([shapes[0], 500])==1, as_tuple=True)]
        
        protbert_output = self.outLayer_protbert(protbert_feature)
        
        return None, protbert_output, attention_loss, None ## quick fix :-) 
    
    
    
    