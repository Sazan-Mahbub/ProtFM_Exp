# -*- encoding:utf8 -*-

import os
import time
import sys
import numpy as np

import torch as t
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

        # self.multi_CNN = nn.Sequential()
        # self.multi_CNN.add_module("layer_convs",
        #                           ConvsLayer())

        # self.DNN1 = nn.Sequential()
        # self.DNN1.add_module("DNN_layer1",
        #                      nn.Linear(716, 512))
        # self.DNN1.add_module("ReLU1",
        #                      nn.ReLU())
        # # self.dropout_layer = nn.Dropout(self.dropout)
        # self.DNN2 = nn.Sequential()
        # self.DNN2.add_module("DNN_layer2",
        #                      nn.Linear(512, 256))
        # self.DNN2.add_module("ReLU2",
        #                      nn.ReLU())

        self.protbert_mlp =  nn.Sequential(
            nn.Linear(1024, 64),
            nn.Dropout(.5),
            nn.LeakyReLU())
        
        self.outLayer = nn.Sequential(
            nn.Linear(64, class_nums),
            nn.Sigmoid())
        
        
        self.protBERT_emb_dim = 32*2
        self.outLayer_protbert = nn.Sequential(
            nn.Linear(self.protBERT_emb_dim, class_nums),
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

        self.gat_layer = eagat.MultiHeadGATLayer(
                                    in_dim=32,
                                    out_dim=32,
                                    edge_dim=2,
                                    num_heads=1,
                                    use_bias=False,
                                    merge='cat',
                                    config_dict=config_dict)
        self.Cosine_Similarity = nn.CosineSimilarity(dim=-1, eps=1e-6)
        self.experimental_transformer_layer = eagat.TransformerEncoderLayer(d_model=self.protBERT_emb_dim, 
                                                                            nhead=1, 
                                                                            dim_feedforward=self.protBERT_emb_dim, 
                                                                            dropout=0.1, 
                                                                            activation="relu")
        

    def forward(self, protbert_feature, local_features, label_idx_onehot, graph_batch):
        shapes = protbert_feature.data.shape
        features = protbert_feature.squeeze(1).permute(0, 2, 1)
        features = self.conv_encoder(features)
        features = features.permute(0, 2, 1).contiguous() # https://github.com/agrimgupta92/sgan/issues/22#issuecomment-452980941
        gat_output, head_attn_scores = self.gat_layer(graph_batch, features.view([shapes[0]*500, 32])) 
        
        features2 = gat_output.view([shapes[0], 500, 32])
        # print(features2.shape, features.shape); exit()
        #z.repeat(1,2).view(shapes[0],2,shapes[1])
        features = t.cat((features2, features), 2) ## sazan: uncomment later
        # features = features2
        # print(features2.shape, features.shape)
        features = features.view([shapes[0], 500, 64])[t.nonzero(label_idx_onehot.view([shapes[0], 500])==1, as_tuple=True)]

        protbert_feature = self.protbert_mlp(protbert_feature) ## projecting to a lower-dimensional space
#         print('1>', protbert_feature.shape)
        protbert_feature, tx_attn = self.experimental_transformer_layer(protbert_feature) ## extra experimental layer
#         print('2>', protbert_feature.shape)
        protbert_feature = protbert_feature[t.nonzero(label_idx_onehot.view([shapes[0], 500])==1, as_tuple=True)]
#         print('3>', protbert_feature.shape)
        # features = features.gather(1, )
        # print(features2.shape, features.shape)
        # features = self.DNN1(features)
        # # features =self.dropout_layer(features)
        # features = self.DNN2(features)
        cos_sim = self.Cosine_Similarity(features.detach(), protbert_feature) ## making a detached copy so that the constraint is not imposed on the gnn as well.
        
        egret_output = self.outLayer(features)
        protbert_output = self.outLayer_protbert(protbert_feature)
        # print('output', features.shape, head_attn_scores.shape); exit()
        return egret_output, protbert_output, cos_sim, head_attn_scores