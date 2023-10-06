import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable



#### start transformer encoder 
## Ref: https://buomsoo-kim.github.io/attention/2020/04/27/Attention-mechanism-21.md/ 

def _get_activation_fn(activation):
    if activation == "relu":
        return F.relu
    elif activation == "gelu":
        return F.gelu

    raise RuntimeError("activation should be relu/gelu, not {}".format(activation))

class TransformerEncoderLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1, activation="relu"):
        super(TransformerEncoderLayer, self).__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)
        
    def __setstate__(self, state):
        if 'activation' not in state:
            state['activation'] = F.relu
        super(TransformerEncoderLayer, self).__setstate__(state)

    def forward(self, src, attn_mask=None, average_attn_weights=False, src_key_padding_mask=None):
        # type: (Tensor, Optional[Tensor], Optional[Tensor]) -> Tensor
        src2, weights = self.self_attn(src, src, src, attn_mask=attn_mask,
                                       key_padding_mask=src_key_padding_mask, 
                                       need_weights=True, 
                                       average_attn_weights=average_attn_weights)
        assert src2.shape[1] == weights.shape[-1]
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        return src, weights
    

class TransformerEncoderLayer_CrossModalAttention(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1, activation="relu"):
        super(TransformerEncoderLayer_CrossModalAttention, self).__init__()
        
        self.self_attn_d1 = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.self_attn_d2 = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        
        # Implementation of Feedforward model
        _mult_ = 2
        self.linear1 = nn.Linear(d_model * _mult_, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model * _mult_)

        self.norm1 = nn.LayerNorm(d_model * _mult_)
        self.norm2 = nn.LayerNorm(d_model * _mult_)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)
        
    def __setstate__(self, state):
        if 'activation' not in state:
            state['activation'] = F.relu
        super(TransformerEncoderLayer, self).__setstate__(state)

    def forward(self, src_d1, src_d2, attn_mask=None, average_attn_weights=False, src_key_padding_mask=None):
        # type: (Tensor, Optional[Tensor], Optional[Tensor]) -> Tensor
        src_d11, weights1 = self.self_attn(src_d1, src_d2, src_d2, attn_mask=attn_mask,
                                       key_padding_mask=src_key_padding_mask, 
                                       need_weights=True, 
                                       average_attn_weights=average_attn_weights)
        src_d22, weights2 = self.self_attn(src_d2, src_d1, src_d1, attn_mask=attn_mask,
                                       key_padding_mask=src_key_padding_mask, 
                                       need_weights=True, 
                                       average_attn_weights=average_attn_weights)
        assert src_d11.shape[1] == weights1.shape[-1]
        assert src_d22.shape[1] == weights2.shape[-1]
        src_d1 = src_d1 + self.dropout1(src_d11)
        src_d2 = src_d2 + self.dropout1(src_d22)
        src = torch.cat([src_d1, src_d2], -1) ## change it to add?
        src = self.norm1(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        return src, None ##quick fix :)
    
#### end transformer encoder



class GATLayer(nn.Module):
    def __init__(self, in_dim, out_dim, edge_dim, use_bias, config_dict=None):
        super(GATLayer, self).__init__()
        # experimental hyperparams
        self.apply_attention = True
        # self.use_edge_features = False
        self.transform_edge_for_att_calc = False
        self.apply_attention_on_edge = False
        self.aggregate_edge = False ###
        # self.edge_transform = False
        self.edge_dependent_attention = False ###
        self.self_loop = False # or skip connection
        self.self_node_transform = False and self.self_loop
        self.activation = None #nn.LeakyReLU(negative_slope=0.01)
        if config_dict is not None:
          self.apply_attention = config_dict['apply_attention']
          # self.use_edge_features = config_dict['use_edge_features']
          self.transform_edge_for_att_calc = config_dict['transform_edge_for_att_calc']
          self.apply_attention_on_edge = config_dict['apply_attention_on_edge']
          self.aggregate_edge = config_dict['aggregate_edge']
          # self.edge_transform = config_dict['edge_transform']
          self.edge_dependent_attention = config_dict['edge_dependent_attention']
          self.self_loop = config_dict['self_loop'] # or skip connection
          self.self_node_transform = config_dict['self_node_transform'] and self.self_loop
          self.activation = config_dict['activation']
          self.feat_drop = nn.Dropout(config_dict['feat_drop'])
          self.attn_drop = nn.Dropout(config_dict['attn_drop'])
          self.edge_feat_drop = nn.Dropout(config_dict['edge_feat_drop'])
          # self.edge_attn_drop = nn.Dropout(config_dict['edge_attn_drop'])
          self.use_batch_norm = config_dict['use_batch_norm']

        # equation (1)
        self.fc = nn.Linear(in_dim, out_dim, bias=use_bias)
        self.bn_fc = nn.BatchNorm1d(num_features=out_dim, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True) if self.use_batch_norm else nn.Identity()
        # equation (2)
        if self.edge_dependent_attention:
          self.attn_fc = nn.Linear(2 * out_dim+edge_dim, 1, bias=use_bias)
        else:
          self.attn_fc = nn.Linear(2 * out_dim, 1, bias=use_bias)
        if self.aggregate_edge:
          self.fc_edge = nn.Linear(edge_dim, out_dim, bias=use_bias)
          self.bn_fc_edge = nn.BatchNorm1d(num_features=out_dim, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True) if self.use_batch_norm else nn.Identity()
        if self.self_node_transform:
          self.fc_self = nn.Linear(in_dim, out_dim, bias=use_bias)
          self.bn_fc_self = nn.BatchNorm1d(num_features=out_dim, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True) if self.use_batch_norm else nn.Identity()
        if self.transform_edge_for_att_calc:
          self.fc_edge_for_att_calc = nn.Linear(edge_dim, edge_dim, bias=use_bias)
          self.bn_fc_edge_for_att_calc = nn.BatchNorm1d(num_features=edge_dim, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True) if self.use_batch_norm else nn.Identity()
        self.reset_parameters()

    def reset_parameters(self):
        """Reinitialize learnable parameters."""
        gain = nn.init.calculate_gain('relu')
        nn.init.xavier_normal_(self.fc.weight, gain=gain)
        nn.init.xavier_normal_(self.attn_fc.weight, gain=gain)
        if self.aggregate_edge:
          nn.init.xavier_normal_(self.fc_edge.weight, gain=gain)
        if self.self_node_transform:
          nn.init.xavier_normal_(self.fc_self.weight, gain=gain)
        if self.transform_edge_for_att_calc:
          nn.init.xavier_normal_(self.fc_edge_for_att_calc.weight, gain=gain)
          
    def edge_attention(self, edges):
        # edge UDF for equation (2)
        if self.edge_dependent_attention:
          if self.transform_edge_for_att_calc:
            z2 = torch.cat([edges.src['z'], edges.dst['z'], self.bn_fc_edge_for_att_calc(self.fc_edge_for_att_calc(edges.data['ex']))], dim=1)
          else:
            z2 = torch.cat([edges.src['z'], edges.dst['z'], edges.data['ex']], dim=1)
        else:
          z2 = torch.cat([edges.src['z'], edges.dst['z']], dim=1)
        a = self.attn_fc(z2)

        if self.aggregate_edge:
            ez = self.bn_fc_edge(self.fc_edge(edges.data['ex']))
            return {'e': F.leaky_relu(a, negative_slope=0.2), 'ez': ez}
          # else:
          #   ez = edges.data['ex']

        return {'e': F.leaky_relu(a)}

    def message_func(self, edges):
        # message UDF for equation (3) & (4)
        if self.aggregate_edge:
          return {'z': edges.src['z'], 'e': edges.data['e'], 'ez': edges.data['ez']}
        else:
          return {'z': edges.src['z'], 'e': edges.data['e']}

    def reduce_func(self, nodes):
        # reduce UDF for equation (3) & (4)
        # equation (3)
        if not self.apply_attention:
          h = torch.sum(nodes.mailbox['z'], dim=1)
        else:
          alpha = self.attn_drop(F.softmax(nodes.mailbox['e'], dim=1))
          # equation (4)
          h = torch.sum(alpha * nodes.mailbox['z'], dim=1)
        if self.aggregate_edge:
          if self.apply_attention_on_edge:
            h = h + torch.sum(alpha * nodes.mailbox['ez'], dim=1)
          else:
            h = h + torch.sum(nodes.mailbox['ez'], dim=1)
        return {'h': h}

    def forward(self, g, nfeatures):
        # equation (1)
        g = g.local_var()
        nfeatures = self.feat_drop(nfeatures)
        g.edata['ex'] = self.edge_feat_drop(g.edata['ex'])
        g.ndata['z'] = self.bn_fc(self.fc(nfeatures))

        # equation (2)
        g.apply_edges(self.edge_attention)
        # equation (3) & (4)
        g.update_all(self.message_func, self.reduce_func)
        
        if self.self_loop:
          if self.self_node_transform:
            g.ndata['h'] = g.ndata['h'] + self.bn_fc_self(self.fc_self(nfeatures))
          else:
            g.ndata['h'] = g.ndata['h'] + nfeatures
        
        if self.activation is not None:
          g.ndata['h'] = self.activation(g.ndata['h'])
        
        return g.ndata.pop('h')


class MultiHeadGATLayer(nn.Module):
    def __init__(self, in_dim, out_dim, edge_dim, num_heads, use_bias, merge='cat', config_dict=None):
        super(MultiHeadGATLayer, self).__init__()
        self.heads = nn.ModuleList()
        for i in range(num_heads):
            self.heads.append(GATLayer(in_dim, out_dim, edge_dim, use_bias, config_dict=config_dict)) #in_dim, out_dim, edge_dim, use_bias, config_dict=None
        self.merge = merge

    def forward(self, g, h):
        head_outs = [attn_head(g, h) for attn_head in self.heads]
        if self.merge == 'cat':
            # concat on the output feature dimension (dim=1)
            return torch.cat(head_outs, dim=1), None
        else:
            # merge using average
            return torch.mean(torch.stack(head_outs)), None


class GAT(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, num_heads, config_dict=None):
        super(GAT, self).__init__()
        self.layer1 = MultiHeadGATLayer(in_dim, hidden_dim, num_heads, config_dict=config_dict)
        self.bn1 = nn.BatchNorm1d(num_features=hidden_dim * num_heads, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        # Be aware that the input dimension is hidden_dim*num_heads since
        # multiple head outputs are concatenated together. Also, only
        # one attention head in the output layer.
        self.layer2 = MultiHeadGATLayer(hidden_dim * num_heads, out_dim, 1, config_dict=config_dict)
        self.bn2 = nn.BatchNorm1d(num_features=out_dim, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        
    def forward(self, g, h):
        h = self.layer1(g, h)
        h = F.elu(h)
        h = self.bn1(h)
        h = self.layer2(g, h)
        h = self.bn2(h)
        return h


    
    
config_dict = {
    'use_batch_norm': False,
    'feat_drop': 0.0,
    'attn_drop': 0.0,
    'edge_feat_drop': 0.0,
    # 'edge_attn_drop': 0.0,
    'hidden_dim' : 32,
    'out_dim' : 32, #512
    'apply_attention' : True,
    # 'use_edge_features' : True,
    'transform_edge_for_att_calc': True, # whether the edge features will be linearly transformed before being used for attention score calculations.
    'apply_attention_on_edge': True, # whether the calculated attention scores will be used for a weighted sum of the edge-features.
    'aggregate_edge' : True, # whether the edges will also be aggregated with the central node.
    # 'edge_transform' : True, # must be True for aggregate_edge.
    'edge_dependent_attention' : True, # whether edge-features will be used for attention score calculation.
    'self_loop' : False, # or skip connection.
    'self_node_transform' : True, # for self_loop (or skip connection), whether we will use a separate linear transformation of the central note
    'activation' : None #nn.LeakyReLU(negative_slope=.0) # the only activation/non-linearity in the module. Whether the output (hidden state) will be activated with some non-linearity. Used negative_slope=1 for linear activation
}


