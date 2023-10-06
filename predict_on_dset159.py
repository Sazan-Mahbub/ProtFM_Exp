#-*- encoding:utf8 -*-

import os
import time


import pickle
import numpy as np
import torch
from torch.optim import lr_scheduler
from torch.nn.init import xavier_normal,xavier_normal_
from torch import nn
import torch.utils.data.sampler as sampler


from utils.config import DefaultConfig
from models.deep_ppi import DeepPPI
from generator import data_generator
from generator.data_generator import graph_collate


from evaluation import compute_roc, compute_aupr, compute_mcc, micro_score,acc_score, compute_performance


configs = DefaultConfig()
THREADHOLD = 0.2

class AverageMeter(object):
    """
    Computes and stores the average and current value
    Copied from: https://github.com/pytorch/examples/blob/master/imagenet/main.py
    """
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def weight_init(m):
    if isinstance(m,nn.Conv2d):
        xavier_normal_(m.weight.data)
    elif isinstance(m,nn.Linear):
        xavier_normal_(m.weight.data)
    
def test(model, loader,path_dir,pre_num=1):

    # Model on eval mode
    model.eval()
    length = len(loader)
    result = []
    all_trues = []

    for batch_idx, (protbert_data, local_data, label, label_idx_onehot, graph_batch) in enumerate(loader):
        # Create vaiables
        local_var = local_data
        # Create vaiables
        with torch.no_grad():
            # graph_batch = dgl.batch(graph_batch, node_attrs='__ALL__', edge_attrs='__ALL__')
            if torch.cuda.is_available():
                protbert_var = torch.autograd.Variable(protbert_data.cuda().float())
                # local_var = torch.autograd.Variable(local_data.cuda().float())
                target_var = torch.autograd.Variable(label.cuda().float())
                label_idx_onehot = torch.autograd.Variable(label_idx_onehot.cuda().float())
                graph_batch.edata['ex'] = torch.autograd.Variable(graph_batch.edata['ex'].cuda().float())
            else:
                protbert_var = torch.autograd.Variable(protbert_data.float())
                # local_var = torch.autograd.Variable(local_data.float())
                target_var = torch.autograd.Variable(label.float())
                label_idx_onehot = torch.autograd.Variable(label_idx_onehot.float())
                graph_batch.edata['ex'] = torch.autograd.Variable(graph_batch.edata['ex'].float())

            # compute output
            t0 = time.time()
            output = model(protbert_var, local_var, label_idx_onehot, graph_batch)
            print(batch_idx, output.shape)
            shapes = output.data.shape
            output = output.view(shapes[0]*1)
            result.append(output.data.cpu().numpy())
            all_trues.append(label.numpy())
        

    #caculate
    all_trues = np.concatenate(all_trues, axis=0)
    all_preds = np.concatenate(result, axis=0)

    auc = compute_roc(all_preds, all_trues)
    aupr = compute_aupr(all_preds, all_trues)
    f_max, p_max, r_max, t_max, predictions_max = compute_performance(all_preds,all_trues)
    acc = acc_score(predictions_max,all_trues)
    mcc = compute_mcc(predictions_max, all_trues)

    print(
        'acc:%0.6f,F_value:%0.6f, precision:%0.6f,recall:%0.6f,auc:%0.6f,aupr:%0.6f,mcc:%0.6f,threadhold:%0.6f\n' % (
        acc, f_max, p_max, r_max,auc, aupr,mcc,t_max))
    


    predict_result = {}
    predict_result["pred"] = all_preds
    predict_result["label"] = all_trues
    result_file = "{0}/dset159_test_predict.pkl".format(path_dir)
    with open(result_file,"wb") as fp:
        pickle.dump(predict_result,fp)


def predict(model_file,test_data,window_size,path_dir,ratio):
    # test_sequences_file = ['data_cache/{0}_sequence_data.pkl'.format(key) for key in test_data]
    # test_dssp_file = ['data_cache/{0}_dssp_data.pkl'.format(key) for key in test_data]
    # test_pssm_file = ['data_cache/{0}_pssm_data.pkl'.format(key) for key in test_data]
    test_protBERT_file = ['data_cache/ProtTrans_Bert_{0}_features_deepPPISP.pkl.gz'.format(key) for key in test_data]
    test_label_file = ['data_cache/{0}_label.pkl'.format(key) for key in test_data]
    all_list_file = 'data_cache/dset159_all_dset_list.pkl'
    test_list_file = 'data_cache/dset159_test_amino_acid_list_cleaned_19_proteins.pkl'
    # parameters
    batch_size = configs.batch_size

    print(test_list_file)
    #parameters
    batch_size = configs.batch_size

    # Datasets
    test_dataSet = data_generator.dataSet(window_size, test_protBERT_file, test_label_file,
                                             all_list_file)
    # Models
    
    with open(test_list_file,"rb") as fp:
        test_list = pickle.load(fp)
    print(test_list.__len__(), test_list[0])
    test_samples = sampler.SubsetRandomSampler(test_list)
    test_loader = torch.utils.data.DataLoader(test_dataSet, batch_size=batch_size,
                                              sampler=test_samples, pin_memory=(torch.cuda.is_available()),
                                               num_workers=configs.num_workers, drop_last=False, collate_fn=graph_collate)

    # Models
    class_nums = 1
    model = DeepPPI(class_nums,window_size,ratio)
    model.load_state_dict(torch.load(model_file))
    model = model.cuda()
    # print(test_list.__len__(), test_list[0])
    test(model, test_loader,path_dir)

    print('Done!')



if __name__ == '__main__':


    ratio_list = (2,1) #glboal:local
    window_size = 3
    path_dir = "./checkpoints/deep_ppi_saved_models"
    
    datas = ["dset159"]
    if not os.path.exists(path_dir):
        os.makedirs(path_dir)

    for saved_model_number in [4]:
        model_file = "{0}/DeepPPI_model_{1}.dat".format(path_dir, saved_model_number)
        predict(model_file,datas,window_size,path_dir,ratio_list)

