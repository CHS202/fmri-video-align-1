import os
import datetime
import shutil
import numpy as np
import torch
import torch.nn as nn
from scipy.stats import pearsonr
from transforms.spatial import Preprocessing
import torchvision
from sklearn.metrics.pairwise import cosine_similarity
import torch.nn.functional as F
import pandas as pd
from scipy.stats import rankdata
import sys
from collections import OrderedDict
from .layer_contribution import calculate_layer_contributions, calculate_layer_contributions_v2, calculate_layer_correlation, calculate_layer_contributions_v3
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import SGDRegressor
from sklearn.decomposition import IncrementalPCA
from sklearn.metrics import mean_squared_error, r2_score
from collections import OrderedDict
from models.single_class import SingleClassWrapper
def local2global_path(opt,test_svm=False):
    if opt.root_path != '':
        opt.video_path = os.path.join(opt.data_root_path, opt.video_path)
        opt.annotation_path = os.path.join(opt.data_root_path, opt.annotation_path)
        opt.video_raw_path = os.path.join(opt.data_root_path, opt.video_raw_path)
        opt.neural_video_path = os.path.join(opt.data_root_path, opt.neural_video_path)
        opt.neural_video_raw_path = os.path.join(opt.data_root_path, opt.neural_video_raw_path)
        if opt.debug:
            opt.result_path = "debug/minor"
        # opt.result_path = os.path.join(opt.root_path, opt.result_path)
        opt.result_path = os.path.join('BrainGuided', opt.result_path)
        if opt.expr_name == '':
            # now = datetime.datetime.now()
            # now = now.strftime('result_%Y%m%d_%H%M%S')
            if opt.co_train == True:
                if opt.use_model:
                    use_model = '1'
                else:
                    use_model = '0'
                co_train = 'co_train'+'_alpha='+str(opt.alpha)+'_use_model='+use_model
            else:
                co_train = 'not_co_train'

            if opt.get_layer_contribution:
                opt.result_path = opt.model_pretrained[:-20]
            else:
                if opt.behavior == False:
                    if opt.network_choose == 'shufflenet_v1':
                        if opt.random_choice == True:
                            now = f'new_result/{opt.task}/sig_test_add_pfc/'+opt.dataset_choose+'/'+opt.network_choose+'_1.5x/'+'/result_' + opt.dataset_choose + '_split=' + str(opt.split) + '_' + co_train + '_lr=' + str(opt.learning_rate)+'_'+str(opt.sig_test_run) + '_' + str(opt.video_num)
                        else:
                            if opt.data_use == 'mean':
                                now = f'new_result/{opt.task}/sig_test_add_pfc/' + opt.dataset_choose + '/' + opt.network_choose + '_1.5x/' + '/result_' + opt.dataset_choose + '_split=' + str(opt.split) + '_' + co_train + '_lr=' + str(opt.learning_rate) + '_' + str(opt.sig_test_run)
                            else:
                                now = f'new_result/{opt.task}/sig_test_add_pfc/' + opt.dataset_choose + '/' + opt.network_choose + '_1.5x/' + '/result_' + opt.dataset_choose + '_split=' + str(opt.split) + '_' + co_train + '_lr=' + str(opt.learning_rate) + '_' + str(opt.sig_test_run) + '_' + opt.data_use
                    elif opt.network_choose == 'mobilenet_v2':
                        now = f'new_result/{opt.task}/sig_test_add_pfc/' + opt.dataset_choose + '/' + opt.network_choose + '_0.45x/' + '/result_' + opt.dataset_choose + '_split=' + str(opt.split) + '_' + co_train + '_lr=' + str(opt.learning_rate) + '_' + str(opt.sig_test_run) + '_' + opt.data_use

                    else:
                        if opt.random_choice == True:
                            now = f'new_result/{opt.task}/sig_test_add_pfc/' + opt.dataset_choose + '/' + opt.network_choose  + '/result_' + opt.dataset_choose + '_split=' + str(opt.split) + '_' + co_train + '_lr=' + str(opt.learning_rate) + '_' + str(opt.sig_test_run) + '_' + str(opt.video_num)
                        else:
                            if opt.data_use == 'mean':
                                now = f'new_result/{opt.task}/sig_test_add_pfc/' + opt.dataset_choose + '/' + opt.network_choose + '/result_' + opt.dataset_choose + '_split=' + str(opt.split) + '_' + co_train + '_lr=' + str(opt.learning_rate) + '_' + str(opt.sig_test_run)
                            else:
                                now = f'new_result/{opt.task}/sig_test_add_pfc/' + opt.dataset_choose + '/' + opt.network_choose + '/result_' + opt.dataset_choose + '_split=' + str(opt.split) + '_' + co_train + '_lr=' + str(opt.learning_rate) + '_' + str(opt.sig_test_run) + '_' + opt.data_use
                elif opt.behavior == True:
                    if opt.network_choose == 'shufflenet_v1':
                        now = f'new_result/{opt.task}/sig_test_add_pfc/'+opt.dataset_choose+'/'+opt.network_choose+'_1.5x/result_' + opt.dataset_choose + '_split=' + str(
                        opt.split) + '_lr=' + str(opt.learning_rate)+'_'+str(opt.sig_test_run)+'_behavior_'+opt.behavior_data
                    elif opt.network_choose == 'mobilenet_v2':
                        now = f'new_result/{opt.task}/sig_test_add_pfc/' + opt.dataset_choose + '/' + opt.network_choose + '_0.45x/result_' + opt.dataset_choose + '_split=' + str(opt.split) + '_lr=' + str(opt.learning_rate) + '_' + str(opt.sig_test_run) + '_behavior_' + opt.behavior_data
                    else:
                        now = f'new_result/{opt.task}/sig_test_add_pfc/' + opt.dataset_choose + '/' + opt.network_choose + '/result_' + opt.dataset_choose + '_split=' + str(opt.split) + '_lr=' + str(opt.learning_rate) + '_' + str(opt.sig_test_run) + '_behavior_' + opt.behavior_data
                opt.result_path = os.path.join(opt.result_path, now)
            if opt.single_annot_class == True:
                opt.result_path = os.path.join(opt.result_path, str(opt.target_class))
            print('result_path:',opt.result_path)
        else:
            opt.result_path = os.path.join(opt.result_path, opt.expr_name)

            if os.path.exists(opt.result_path):
                shutil.rmtree(opt.result_path)
            if not test_svm:
                os.mkdir(opt.result_path)

        opt.log_path = os.path.join(opt.result_path, "tensorboard")
        opt.ckpt_path = os.path.join(opt.result_path,"checkpoints")
        if not test_svm:
            if not os.path.exists(opt.log_path):
                os.makedirs(opt.log_path)
            if not os.path.exists(opt.ckpt_path):
                os.mkdir(opt.ckpt_path)
    else:
        raise Exception


def get_spatial_transform(opt, mode):
    if mode == "train":
        return Preprocessing(size=opt.sample_size, is_aug=True, center=False)
    elif mode == "val":
        return Preprocessing(size=opt.sample_size, is_aug=False, center=True)
    elif mode == "test":
        return Preprocessing(size=opt.sample_size, is_aug=False, center=False)
    else:
        raise Exception


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

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


def process_data_item(opt, data_item):
    visual, target,  visualization_item, annot = data_item
    target = target.cuda()

    visual = visual.cuda()

    batch = visual.size(0)
    return visual, target,  visualization_item, batch, annot.cuda()

def process_iscience_data_item(opt, data_item):
    visual, visualization_item = data_item

    visual = visual.cuda()

    batch = visual.size(0)
    return visual, visualization_item, batch

def process_neural_data_item(opt, data_item):
    visual, neural_response,  visualization_item, target = data_item # add target because mocified Neural dataset for LSTM training
    visual = visual.cuda()
    batch = visual.size(0)
    # print('threshold=', opt.voxel_select_threshold)
    if opt.data_use == 'mean':
        if opt.dapello == True:
            # Dapello et al. needs raw (n_samples x n_voxels) activations, not an RDM.
            # Voxel counts can differ across subjects, so concatenate along the
            # feature dimension rather than averaging RDMs.
            voxel_list = []
            for i in range(5):
                voxel_select = neural_response['sub-' + str(i + 1).zfill(2)].cuda()
                voxel_list.append(voxel_select)
            RSA_output = torch.cat(voxel_list, dim=1)  # (batch, sum of per-subject voxels)
        else:
            RSA = torch.zeros([5,batch,batch]).cuda()
            for i in range(5):
                voxel_select = neural_response['sub-'+str(i+1).zfill(2)].cuda()
                # voxel_select = neural_response['Subject'+str(i+1)].cuda()
                if opt.rho4rdm == True:
                    # voxel_select_numpy = voxel_select.cpu().numpy()
                    # ranks_numpy = rankdata(voxel_select_numpy, method='average', axis=1)
                    # del voxel_select_numpy
                    # voxel_ranks = torch.tensor(ranks_numpy, dtype=torch.float32, device=voxel_select.device)
                    # RSA[i] = torch.corrcoef(voxel_ranks)
                    # del voxel_ranks
                    voxel_ranks = torch.argsort(torch.argsort(voxel_select, dim=1), dim=1).float()
                    RSA[i] = torch.corrcoef(voxel_ranks)
                else:
                    voxel_select = torch.div(voxel_select,torch.norm(voxel_select,p=2,dim=1).reshape(batch,1))
                    RSA[i] = torch.mm(voxel_select,voxel_select.transpose(1,0))
            # if opt.RSA_similarity_print == True:
            #     across_subject = np.zeros((5, 5))
            #     p_value_model = np.zeros((5, 5))
            #     for i in range(5):
            #         for j in range(5):
            #             across_subject[i, j] = pearsonr(RSA[i].ravel(), RSA[j].ravel())[0]
            #             p_value_model[i, j] = pearsonr(RSA[i].ravel(), RSA[j].ravel())[1]
            #     print(across_subject)
            RSA_output = torch.mean(RSA,dim=0)
    else:
        voxel_select = neural_response[opt.data_use].cuda()
        if opt.dapello == True:
            # raw activations, unnormalized and untransformed — centering happens inside the CKA loss
            RSA_output = voxel_select
        elif opt.rho4rdm == True:
            voxel_select_numpy = voxel_select.cpu().numpy()
            ranks_numpy = rankdata(voxel_select_numpy, method='average', axis=1)
            voxel_ranks = torch.tensor(ranks_numpy, dtype=torch.float32, device=voxel_select.device)
            RSA_output = torch.corrcoef(voxel_ranks)
        else:
            voxel_select = torch.div(voxel_select, torch.norm(voxel_select, p=2, dim=1).reshape(batch, 1))
            RSA_output = torch.mm(voxel_select,voxel_select.transpose(1,0))
    if opt.align_each_roi == True:
        return visual,  RSA_output, batch,visualization_item, target
    else:
        return visual,  RSA_output, batch,visualization_item
def process_neural_data_item_v2(opt, data_item):
    visual, neural_response, visualization_item, target = data_item
    visual = visual.cuda()
    target = target.cuda()                          # <-- keep target on GPU
    batch = visual.size(0)

    unique_labels = torch.unique(target, sorted=True)
    num_classes = unique_labels.shape[0]

    def compute_label_rdm(voxel_select):
        voxel_dim = voxel_select.shape[1]
        label_means = torch.zeros(num_classes, voxel_dim, device=voxel_select.device)
        for idx, lbl in enumerate(unique_labels):
            mask = (target == lbl)
            label_means[idx] = voxel_select[mask].mean(dim=0)
        if opt.rho4rdm:
            label_ranks = torch.argsort(torch.argsort(label_means, dim=1), dim=1).float()
            rdm = torch.corrcoef(label_ranks)
        else:
            normed = torch.div(
                label_means,
                torch.norm(label_means, p=2, dim=1, keepdim=True).clamp(min=1e-8)
            )
            rdm = torch.mm(normed, normed.t())
        return rdm

    if opt.data_use == 'mean':
        RSA = torch.zeros([5, num_classes, num_classes]).cuda()
        for i in range(5):
            voxel_select = neural_response['sub-' + str(i + 1).zfill(2)].cuda()
            RSA[i] = compute_label_rdm(voxel_select)
        RSA_output = torch.mean(RSA, dim=0)
    else:
        voxel_select = neural_response[opt.data_use].cuda()
        RSA_output = compute_label_rdm(voxel_select)

    return visual, RSA_output, batch, visualization_item, target   # <-- added target
def process_behavior_data_item(opt, data_item):
    visual, behavior_response, visualization_item = data_item
    visual = visual.cuda()
    batch = visual.size(0)
    behavior_response = behavior_response.cuda()
    behavior_response = torch.div(behavior_response,torch.norm(behavior_response,p=2,dim=1).reshape(batch,1))
    RSA_output = torch.mm(behavior_response,behavior_response.transpose(1,0))
    return visual,  RSA_output, batch,visualization_item

def run_model(opt, inputs, model, criterion=None, i=0, print_attention=False, period=30, return_attention=False,test_svm=False):
    if not test_svm:
        visual, target = inputs
        # print('visual device',visual.device)
        outputs = model(visual)
        y_pred, alpha, beta, gamma,fSCT = outputs
        if opt.loss_func == 'bce':
            y_pred = torch.sigmoid(y_pred)
            # y_pred = (y_pred > 0.5).float()
            target = target.float()
        elif opt.loss_func == 'mse':
            target = target.float()
            # print(y_pred.shape,target.shape)
        loss = criterion(y_pred, target)
        if i % period == 0 and print_attention:
            print('====alpha====')
            print(alpha[:, 0, :])
            print('====beta====')
            print(beta[:, 0, 0:512:32])
            print('====gamma====')
            print(gamma)
        if not return_attention:
            return y_pred, loss
        else:
            return y_pred, loss, [alpha, beta, gamma]
    else:
        visual, target = inputs
        outputs = model(visual,test_svm=test_svm)
        y_pred, alpha, beta, gamma, fSCT = outputs
        return fSCT
    
def run_model_get_contribution(opt, train_loader,neural_loader, model):
    print("# ---------------------------------------------------------------------- #")
    print('Getting layer contributions')
    model.eval()

    contribution_all = {}
    mse_all = 0
    r2_all = 0
    corr_all = {}
    dataloader_iterator1 = iter(neural_loader)
    # print("length of train_loader:",len(train_loader))
    for i, train_data_item in enumerate(train_loader):
        try:
            neural_data_item = next(dataloader_iterator1)
        except StopIteration:
            dataloader_iterator1 = iter(neural_loader)
            neural_data_item = next(dataloader_iterator1)

        neural_visual, RSA_output, neural_batch_size,visual_item, target = process_neural_data_item_v2(opt, neural_data_item)  # !!!!!!!!!!!!!!!rdm shape = 4*4!!!!!!!!!!!!!!
        # neural_visual, RSA_output, neural_batch_size,visual_item = process_neural_data_item(opt, neural_data_item)

        _, neural_response,  _, _ = neural_data_item
        voxel_select = neural_response[opt.data_use].cuda()
        # print("voxel_select.shape:",voxel_select.shape)

        if opt.network_choose != 'vaa':
            visual_p = model.input_process(neural_visual)
        if opt.network_choose == 'resnet_18':
            new_m = torchvision.models._utils.IntermediateLayerGetter(model.resnet, {'0': 'conv1', '4': 'conv5','5':'conv9','6':'conv13','7':'conv17'})
            out = new_m(visual_p)
        elif opt.network_choose == 'squeezenet':
            new_m = torchvision.models._utils.IntermediateLayerGetter(model.CNN.features,{'0': 'conv1', '6': 'Fire3', '9': 'Fire5','12': 'Fire7', '14': 'Fire9'})
            out = new_m(visual_p)
        elif opt.network_choose == 'shufflenet_v1':
            new_m = torchvision.models._utils.IntermediateLayerGetter(model.CNN,{'conv1': 'conv1', 'layer1': 'layer1', 'layer2':'layer2','layer3': 'layer3'})
            out = new_m(visual_p)
        elif opt.network_choose == 'shufflenet_v2':
            new_m = torchvision.models._utils.IntermediateLayerGetter(model.CNN,{'conv1': 'conv1', 'conv_last':'conv_last'})
            new_m_features = torchvision.models._utils.IntermediateLayerGetter(model.CNN.features,{'3': 'features3', '11': 'features11', '15': 'features15'})
            # new_m = torchvision.models._utils.IntermediateLayerGetter(model.CNN,{'conv1': 'conv1', 'features': 'features', 'conv_last':'conv_last'})
            out = new_m(visual_p)
            x = model.CNN.conv1(visual_p) 
            # Pass through maxpool
            x = model.CNN.maxpool(x)
            out_features = new_m_features(x)
            out.update(out_features)
            # order the key as 'conv1', 'features3', 'features11', 'features15', 'conv_last'
            desired_order = ['conv1', 'features3', 'features11', 'features15', 'conv_last']
            # Rebuild the dictionary in the new order
            out = OrderedDict((k, out[k]) for k in desired_order)
            # print("shufflenet_v2's out.keys():",out.keys())
        elif opt.network_choose == 'mobilenet_v1':
            new_m = torchvision.models._utils.IntermediateLayerGetter(model.CNN.features,{'0': '64channels', '1': '128channels', '3': '256channels','5': '512channels', '11': '1024channels','13':'2048channels'})
            out = new_m(visual_p)
        elif opt.network_choose == 'mobilenet_v2':
            new_m = torchvision.models._utils.IntermediateLayerGetter(model.CNN.features,{'0': '14channels', '1': '7channels', '2': '10channels', '4': '14channels', '7': '28channels', '11': '43channels', '14': '72channels', '17': '144channels','18': '1280channels'})
            # new_m = torchvision.models._utils.IntermediateLayerGetter(model.CNN.features,{'0': '14channels', '17': '144channels','18': '1280channels'})
            out = new_m(visual_p)

        # print(out.keys())
        if opt.contribution_method == 'ridge':
            # output, contribution, mse, r2 = calculate_layer_contributions(voxel_select, out, opt)
            output, contribution, mse, r2 = calculate_layer_contributions_v2(voxel_select, out, opt)
            mse_all += mse
            r2_all += r2
            # sum contribution to contribution_all and average
            for k in contribution:
                if k not in contribution_all.keys():
                    contribution_all[k] = contribution[k]
                else:
                    contribution_all[k] += contribution[k]

            # if i % 10 == 0:
            #     # print('layer contribution: %d' % i)
            #     # print("voxel_select.shape:",voxel_select.shape)
            #     plot_output(output, voxel_select, i, opt)
            # print(contribution)
        elif opt.contribution_method == 'rdm_corr':
            corr = calculate_layer_correlation(neural_visual, RSA_output, out, i, target, opt)
            for k in corr:
                if k not in corr_all.keys():
                    corr_all[k] = corr[k]
                else:
                    corr_all[k] += corr[k]
    if opt.contribution_method == 'ridge':
        mse_all = mse_all / len(train_loader)
        r2_all = r2_all / len(train_loader)
        # average contribution
        for k in contribution_all:
            contribution_all[k] = contribution_all[k] / len(train_loader) 
        # np.savetxt(os.path.join(opt.result_path, f'raw_fmri_{opt.roi}_layer_contributions_{opt.data_use}_v2.csv'), 
        np.savetxt(os.path.join(opt.result_path, f'raw_fmri_{opt.roi}_layer_contributions_{opt.data_use}_v2.csv'),
                [[name, contrib] for name, contrib in contribution_all.items()], 
                delimiter=',', fmt='%s', header='layer,contribution')
        # save mse and r2 in the same txt file 
        # np.savetxt(os.path.join(opt.result_path, f'raw_fmri_{opt.roi}_mse_r2_{opt.data_use}_v2.csv'), 
        np.savetxt(os.path.join(opt.result_path, f'raw_fmri_{opt.roi}_mse_r2_{opt.data_use}_v2.csv'),
                [[mse_all, r2_all]], 
                delimiter=',', fmt='%s', header='mse,r2')
    elif opt.contribution_method == 'rdm_corr':
        for k in corr_all:
            corr_all[k] = corr_all[k] / len(train_loader)
        np.savetxt(os.path.join(opt.result_path, f'rf_{opt.roi}_layer_corr_{opt.data_use}.csv'), 
                [[name, corr] for name, corr in corr_all.items()], 
                delimiter=',', fmt='%s', header='layer,correlation')

        
def run_model_get_contribution_v2(opt, neural_train_loader, neural_val_loader, model):
    print("# ---------------------------------------------------------------------- #")
    print('Starting Batch-Updated Ridge Regression (Incremental Learning)')
    model.eval()

    ipca_dict = {}
    layer_metadata = []
    x_scaler = StandardScaler()
    n_voxels = None
    n_features = None

    # ---------------------------------------------------------
    # Phase 1: Fit IncrementalPCA only (no regression yet)
    # ---------------------------------------------------------
    print("Phase 1: Fitting IncrementalPCA...")
    for i, neural_data_item in enumerate(neural_train_loader):
        with torch.no_grad():
            neural_visual, _, _, _ = process_neural_data_item(opt, neural_data_item)
            visual_p = model.input_process(neural_visual)
            out = get_intermediate_outputs(model, visual_p, opt)

        for key, feats in out.items():
            feats_flat = feats.detach().cpu().numpy().reshape(feats.shape[0], -1)
            if key not in ipca_dict:
                ipca_dict[key] = IncrementalPCA(n_components=min(15, feats_flat.shape[0]))
            ipca_dict[key].partial_fit(feats_flat)

    # ---------------------------------------------------------
    # Phase 2: Fit x_scaler properly across ALL batches
    # ---------------------------------------------------------
    print("Phase 2: Fitting scaler across all training batches...")
    y_mean_per_voxel = None
    y_var_per_voxel  = None
    y_count = 0

    for i, neural_data_item in enumerate(neural_train_loader):
        with torch.no_grad():
            neural_visual, _, _, _ = process_neural_data_item(opt, neural_data_item)
            _, neural_response, _, _ = neural_data_item
            y_batch = neural_response[opt.data_use].cpu().numpy().astype(np.float64)
            visual_p = model.input_process(neural_visual)
            out = get_intermediate_outputs(model, visual_p, opt)

        reduced = []
        current_col_idx = 0
        for key, feats in out.items():
            feats_flat = feats.detach().cpu().numpy().reshape(feats.shape[0], -1)
            feats_pca  = ipca_dict[key].transform(feats_flat)
            reduced.append(feats_pca)
            if i == 0:
                n_comp = feats_pca.shape[1]
                layer_metadata.append({"name": key, "start": current_col_idx, "end": current_col_idx + n_comp})
                current_col_idx += n_comp

        X_batch = np.hstack(reduced).astype(np.float64)
        x_scaler.partial_fit(X_batch)  # accumulate mean/var properly

        # Accumulate y statistics for per-voxel z-scoring
        n = y_batch.shape[0]
        if y_mean_per_voxel is None:
            y_mean_per_voxel = np.zeros(y_batch.shape[1], dtype=np.float64)
            y_var_per_voxel  = np.zeros(y_batch.shape[1], dtype=np.float64)
        # Welford online algorithm for stable mean/var
        y_count += n
        delta = y_batch - y_mean_per_voxel
        y_mean_per_voxel += delta.sum(axis=0) / y_count
        y_var_per_voxel  += ((y_batch - y_mean_per_voxel) * delta).sum(axis=0)

    y_std_per_voxel = np.sqrt(y_var_per_voxel / y_count)
    y_std_per_voxel = np.where(y_std_per_voxel < 1e-8, 1.0, y_std_per_voxel)

    # ---------------------------------------------------------
    # Phase 3: Accumulate XtX and XtY (with z-scored Y)
    # ---------------------------------------------------------
    print("Phase 3: Accumulating XtX and XtY...")
    XtX = None
    XtY = None

    for neural_data_item in neural_train_loader:
        with torch.no_grad():
            neural_visual, _, _, _ = process_neural_data_item(opt, neural_data_item)
            _, neural_response, _, _ = neural_data_item
            y_batch = neural_response[opt.data_use].cpu().numpy().astype(np.float64)
            visual_p = model.input_process(neural_visual)
            out = get_intermediate_outputs(model, visual_p, opt)

        reduced = []
        for key, feats in out.items():
            feats_flat = feats.detach().cpu().numpy().reshape(feats.shape[0], -1)
            feats_pca  = ipca_dict[key].transform(feats_flat)
            reduced.append(feats_pca)

        X_batch = np.hstack(reduced).astype(np.float64)
        X_batch = x_scaler.transform(X_batch)

        # Z-score Y per voxel
        y_batch = (y_batch - y_mean_per_voxel) / y_std_per_voxel

        if XtX is None:
            n_features = X_batch.shape[1]
            n_voxels   = y_batch.shape[1]
            XtX = np.zeros((n_features, n_features), dtype=np.float64)
            XtY = np.zeros((n_features, n_voxels),   dtype=np.float64)

        XtX += X_batch.T @ X_batch
        XtY += X_batch.T @ y_batch
        del X_batch, y_batch

    XtX_random = None
    XtY_random = None
    if opt.split == 1 and opt.roi == "EVC":
        # get random data
        print("XtX max:", XtX.max(), "min:", XtX.min(), "shape:", XtX.shape)
        print("XtY max:", XtY.max(), "min:", XtY.min(), "shape:", XtY.shape)
        # generate random XtX and XtY based on shape and min/max
        XtX_random = np.random.rand(XtX.shape[0], XtX.shape[1]) * (XtX.max() - XtX.min()) + XtX.min()
        XtY_random = np.random.rand(XtY.shape[0], XtY.shape[1]) * (XtY.max() - XtY.min()) + XtY.min()

    # ---------------------------------------------------------
    # Solve for multiple alphas, pick best on validation
    # ---------------------------------------------------------
    alphas = np.logspace(1, 3, 3)  # 10 to 1e3
    best_alpha, best_r2, best_weights = None, -np.inf, None

    # Collect val data once (in z-scored space)
    print("Selecting best alpha via validation R²...")
    all_y_val_raw, all_X_val = [], []

    for val_item in neural_val_loader:
        with torch.no_grad():
            val_visual, _, _, _ = process_neural_data_item(opt, val_item)
            _, val_response, _, _ = val_item
            y_val = val_response[opt.data_use].cpu().numpy().astype(np.float64)
            visual_p = model.input_process(val_visual)
            out_val  = get_intermediate_outputs(model, visual_p, opt)

        val_features = []
        for key, feats in out_val.items():
            feats_flat = feats.detach().cpu().numpy().reshape(feats.shape[0], -1)
            val_features.append(ipca_dict[key].transform(feats_flat))

        X_val_batch = np.hstack(val_features).astype(np.float64)
        X_val_batch = x_scaler.transform(X_val_batch)
        all_X_val.append(X_val_batch)
        all_y_val_raw.append(y_val)

    X_val_total   = np.vstack(all_X_val)
    y_val_total     = np.vstack(all_y_val_raw)
    y_val_scaled  = (y_val_total - y_mean_per_voxel) / y_std_per_voxel  # for R² computation

    for alpha in alphas:
        A = XtX + alpha * np.eye(n_features)
        W = np.linalg.solve(A, XtY)
        pred_scaled = X_val_total @ W
        r2_a = np.mean([r2_score(y_val_scaled[:, v], pred_scaled[:, v]) for v in range(n_voxels)])
        print(f"  alpha={alpha:.1e}  R²={r2_a:.4f}")
        if r2_a > best_r2:
            best_r2, best_alpha, best_weights = r2_a, alpha, W.copy()

    print(f"Best alpha: {best_alpha:.1e}  Best R²: {best_r2:.4f}")
    all_weights = best_weights.T  # (n_voxels, n_features)

    # Final predictions — unscale back to original space for interpretability
    pred_val_scaled = X_val_total @ best_weights          # (N_val, V) in z-score space
    pred_val_total  = pred_val_scaled * y_std_per_voxel + y_mean_per_voxel  # unscale

    mse_val = np.mean((y_val_total - pred_val_total) ** 2)
    r2_val  = np.mean([r2_score(y_val_total[:, v], pred_val_total[:, v]) for v in range(n_voxels)])
    print(f'MSE: {mse_val:.4f}  R2: {r2_val:.4f}')
    plot_output(pred_val_total, y_val_total, 0, opt)

    # ---------------------------------------------------------
    # Phase 5: Layer Contributions (on Validation Data)
    # Uses: c_l = cov(pred_l, y) / cov(pred_all, y) per voxel
    # Guarantees sum(contributions) == 1 regardless of R²
    # ---------------------------------------------------------
    print("Phase 5: Calculating Layer Contributions...")
    contributions = {}

    # Pre-compute cov(pred_all, y) per voxel as the shared denominator
    cov_total_per_voxel = np.array([
        np.cov(pred_val_scaled[:, v], y_val_scaled[:, v], ddof=0)[0, 1]
        for v in range(n_voxels)
    ])

    for meta in layer_metadata:
        name = meta['name']
        start, end = meta['start'], meta['end']

        phi_l  = X_val_total[:, start:end]   # (N_val, F_l)
        W_l    = all_weights[:, start:end]    # (V, F_l)
        pred_l = phi_l @ W_l.T               # (N_val, V)

        # if opt.split == 1:
        #     df_weights = pd.DataFrame(
        #         W_l,
        #         index=[f"Voxel_{i}" for i in range(W_l.shape[0])],
        #         columns=[f"Comp_{i}" for i in range(W_l.shape[1])]
        #     )
        #     df_weights.to_csv(os.path.join(
        #         opt.result_path,
        #         f"ns_w_{opt.roi}_{opt.network_choose}_{name}_{opt.data_use}_val_{opt.split}.csv"
        #     ))
        #     del df_weights

        voxel_contribs = []
        for v in range(n_voxels):
            cov_layer_y = np.cov(pred_l[:, v], y_val_scaled[:, v], ddof=0)[0, 1]
            denom = cov_total_per_voxel[v]

            if abs(denom) > 1e-10:
                c = cov_layer_y / denom   # sum over layers == 1 exactly
            else:
                c = 0.0

            voxel_contribs.append(c)

        contributions[name] = np.mean(voxel_contribs)

    print(f'Contributions: {contributions}')
    print(f'Sum of contributions: {sum(contributions.values()):.6f}')  # Should be ~1.0

    np.savetxt(
        os.path.join(opt.result_path, f'ns_{opt.roi}_lc_{opt.data_use}_val_{opt.split}.csv'),
        [[name, contrib] for name, contrib in contributions.items()],
        delimiter=',', fmt='%s', header='layer,contribution'
    )
    np.savetxt(
        os.path.join(opt.result_path, f'ns_{opt.roi}_r2_{opt.data_use}_val_{opt.split}.csv'),
        [[mse_val, r2_val]],
        delimiter=',', fmt='%s', header='mse,r2'
    )

    if XtX_random is not None: # get random data
        print("Other Phase: Random Data...")
        X_val_total_random = np.random.rand(X_val_total.shape[0], X_val_total.shape[1]) * (X_val_total.max() - X_val_total.min()) + X_val_total.min()
        best_alpha, best_r2, best_weights = None, -np.inf, None
        for alpha in alphas:
            A = XtX_random + alpha * np.eye(n_features)
            W = np.linalg.solve(A, XtY_random)
            pred_scaled = X_val_total_random @ W
            r2_a = np.mean([r2_score(y_val_scaled[:, v], pred_scaled[:, v]) for v in range(n_voxels)])
            print(f"  alpha={alpha:.1e}  R²={r2_a:.4f}")
            if r2_a > best_r2:
                best_r2, best_alpha, best_weights = r2_a, alpha, W.copy()
        print(f"Best alpha: {best_alpha:.1e}  Best R²: {best_r2:.4f}")
        all_weights = best_weights.T  # (n_voxels, n_features)

        # Final predictions — unscale back to original space for interpretability
        pred_val_scaled = X_val_total_random @ best_weights          # (N_val, V) in z-score space
        pred_val_total  = pred_val_scaled * y_std_per_voxel + y_mean_per_voxel  # unscale

        mse_val = np.mean((y_val_total - pred_val_total) ** 2)
        r2_val  = np.mean([r2_score(y_val_total[:, v], pred_val_total[:, v]) for v in range(n_voxels)])
        print(f'MSE: {mse_val:.4f}  R2: {r2_val:.4f}')
        plot_output(pred_val_total, y_val_total, 0, opt, filename=f'random_data_{opt.data_use}_{opt.split}.png')
        np.savetxt(
            os.path.join(opt.result_path, f'ns_r2_{opt.data_use}_val_random_{opt.split}.csv'),
            [[mse_val, r2_val]],
            delimiter=',', fmt='%s', header='mse,r2'
        )

    # sys.exit(0)

# v3 is corrected version of v2
def run_model_get_contribution_v3(opt, neural_train_loader, neural_val_loader, model): 
    print("# ---------------------------------------------------------------------- #")
    print('Starting Batch-Updated Ridge Regression (Incremental Learning)')
    model.eval()

    # ── Config ─────────────────────────────────────────────────────────────────
    VARIANCE_TARGET  = 0.99
    N_VOXEL_SAMPLE   = 384
    IPCA_CEIL        = 64          # max components fitted; trimmed after Phase 1
    ACCUMULATE_BATCHES = 2          # tune to your memory budget
                                # effective batch = 16 × 8 = 128 samples

    device           = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    buffer = {}                     # key → list of np arrays
    ipca_dict       = {}
    layer_metadata  = []
    n_components_dict = {}          # per-key: how many PCs cover 99% variance
    pca_gpu         = {}            # per-key: (mean, components[:n_keep]) on GPU
    x_scaler        = StandardScaler()
    voxel_indices   = None          # sampled once after first y_batch is seen
    n_voxels        = N_VOXEL_SAMPLE
    n_features      = None

    torch.manual_seed(42)
    torch.cuda.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    torch.backends.cudnn.deterministic = True
    # ──────────────────────────────────────────────────────────────────────────
    # Phase 1: Collect pooled features on GPU (much smaller now after pooling)
    # ──────────────────────────────────────────────────────────────────────────
    # Add pooling helper
    pool_cache = {}

    def pool_and_flatten(key, feats, pool_size=(6, 6)):
        """Spatially pool 4D feature maps before flattening.
        Accepts either torch.Tensor or np.ndarray."""
        # Convert numpy → tensor if needed
        if isinstance(feats, np.ndarray):
            feats = torch.from_numpy(feats).to(device)

        if feats.dim() == 4:                                   # (B, C, H, W)
            if key not in pool_cache:
                pool_cache[key] = torch.nn.AdaptiveAvgPool2d(pool_size).to(device)
            feats = pool_cache[key](feats)                     # (B, C, 6, 6)
        return feats.reshape(feats.shape[0], -1)               # (B, C*36)
    
    print("Phase 1: Collecting features for GPU PCA...")
    feat_buffer_gpu = {}   # key → list of GPU tensors

    for i, neural_data_item in enumerate(neural_train_loader):
        with torch.no_grad():
            neural_visual, _, _, _ = process_neural_data_item(opt, neural_data_item)
            visual_p = model.input_process(neural_visual)
            out = get_intermediate_outputs(model, visual_p, opt)

        for key, feats in out.items():
            # print("key:", key, "feats.shape:", feats.shape)
            feats_flat = pool_and_flatten(key, feats)          # GPU tensor, small now
            # print("feats_flat.shape:", feats_flat.shape)
            feat_buffer_gpu.setdefault(key, []).append(feats_flat.float())


    # Fit PCA per layer on GPU using randomized SVD
    print(f"Fitting GPU PCA for {VARIANCE_TARGET*100:.0f}% variance...")
    pca_gpu = {}

    for key, chunks in feat_buffer_gpu.items():
        X = torch.cat(chunks, dim=0)                           # (N_total, n_features)
        # print(f"  {key}: {X.shape[0]} samples, {X.shape[1]} features")
        X_mean = X.mean(0, keepdim=True)
        X_c = X - X_mean                                       # center

        # Randomized PCA — n_components is the search ceiling, not the final count
        # q: power iterations (higher = more accurate, more memory)
        # niter: oversampling (higher = more accurate)
        U, S, V = torch.pca_lowrank(X_c, q=min(IPCA_CEIL, X.shape[1]),
                                    center=False, niter=4)

        # S are singular values → variance = S² / (N-1)
        var = (S ** 2) / (X.shape[0] - 1)
        cumvar = torch.cumsum(var, dim=0) / var.sum()

        # Find n_keep for 99% variance
        n_keep = int((cumvar < VARIANCE_TARGET).sum().item()) + 1
        n_keep = min(n_keep, len(cumvar))
        print(f"  {key}: keeping {n_keep}/{len(cumvar)} components "
            f"({cumvar[n_keep-1].item()*100:.2f}% variance)")

        pca_gpu[key] = {
            "mean":       X_mean.squeeze(0),                   # (n_features,)
            "components": V[:, :n_keep].T,                     # (n_keep, n_features)
        }

        del X, X_c, U, S, V, chunks
        torch.cuda.empty_cache()

    feat_buffer_gpu.clear()
    # sys.exit(0)

    def pca_transform_gpu(key: str, feats) -> torch.Tensor:
        """Accepts raw tensor (B,C,H,W) or (B,F). Pools → centers → projects."""
        # feats can be tensor or numpy — pool_and_flatten handles both
        x = pool_and_flatten(key, feats).float()               # always a GPU tensor now
        mean  = pca_gpu[key]["mean"]
        comps = pca_gpu[key]["components"]                     # (n_keep, n_features)
        return (x - mean) @ comps.T                            # (B, n_keep)

    # ──────────────────────────────────────────────────────────────────────────
    # Phase 2: Fit x_scaler + determine voxel sample indices + y stats
    # ──────────────────────────────────────────────────────────────────────────
    print("Phase 2: Fitting scaler and sampling voxels...")
    y_mean_per_voxel = None
    y_var_per_voxel  = None
    y_count          = 0

    for i, neural_data_item in enumerate(neural_train_loader):
        with torch.no_grad():
            neural_visual, _, _, _ = process_neural_data_item(opt, neural_data_item)
            _, neural_response, _, _ = neural_data_item
            y_raw = neural_response[opt.data_use].cpu().numpy().astype(np.float64)
            visual_p = model.input_process(neural_visual)
            out = get_intermediate_outputs(model, visual_p, opt)

        # One-time: sample voxel indices
        if voxel_indices is None:
            total_voxels = y_raw.shape[1]
            assert total_voxels >= N_VOXEL_SAMPLE, \
                f"Only {total_voxels} voxels available, cannot sample {N_VOXEL_SAMPLE}"
            rng = np.random.default_rng(seed=42)        # fix seed for reproducibility
            voxel_indices = rng.choice(total_voxels, size=N_VOXEL_SAMPLE, replace=False)
            voxel_indices = np.sort(voxel_indices)
            print(f"  Sampled {N_VOXEL_SAMPLE} voxels from {total_voxels} total.")

        y_batch = y_raw[:, voxel_indices]               # (B, 382)

        # GPU PCA transform → CPU for sklearn scaler
        reduced = []
        current_col_idx = 0
        for key, feats in out.items():
            # feats_flat  = feats.detach().cpu().numpy().reshape(feats.shape[0], -1)
            feats_pca_t = pca_transform_gpu(key, feats)
            feats_pca   = feats_pca_t.cpu().numpy()
            reduced.append(feats_pca)
            if i == 0:
                n_comp = feats_pca.shape[1]
                layer_metadata.append({"name": key, "start": current_col_idx,
                                       "end": current_col_idx + n_comp})
                current_col_idx += n_comp

        X_batch = np.hstack(reduced).astype(np.float64)
        x_scaler.partial_fit(X_batch)

        # Welford online mean/var for y
        n = y_batch.shape[0]
        if y_mean_per_voxel is None:
            y_mean_per_voxel = np.zeros(N_VOXEL_SAMPLE, dtype=np.float64)
            y_var_per_voxel  = np.zeros(N_VOXEL_SAMPLE, dtype=np.float64)
        y_count += n
        delta = y_batch - y_mean_per_voxel
        y_mean_per_voxel += delta.sum(axis=0) / y_count
        y_var_per_voxel  += ((y_batch - y_mean_per_voxel) * delta).sum(axis=0)

    y_std_per_voxel = np.sqrt(y_var_per_voxel / y_count)
    y_std_per_voxel = np.where(y_std_per_voxel < 1e-8, 1.0, y_std_per_voxel)

    # Move scaler params + y stats to GPU for fast transform
    scaler_mean_t  = torch.tensor(x_scaler.mean_,               dtype=torch.float32, device=device)
    scaler_scale_t = torch.tensor(x_scaler.scale_,              dtype=torch.float32, device=device)
    y_mean_t       = torch.tensor(y_mean_per_voxel,             dtype=torch.float32, device=device)
    y_std_t        = torch.tensor(y_std_per_voxel,              dtype=torch.float32, device=device)

    def get_X_gpu(out_dict) -> torch.Tensor:
        """PCA-project + scale all layers; returns (B, n_features) on GPU."""
        parts = []
        for key, feats in out_dict.items():
            # feats_flat = feats.detach().cpu().numpy().reshape(feats.shape[0], -1)
            parts.append(pca_transform_gpu(key, feats))
        X = torch.cat(parts, dim=1).float()
        return (X - scaler_mean_t) / scaler_scale_t    # z-score on GPU

    # ──────────────────────────────────────────────────────────────────────────
    # Phase 3: Accumulate XtX and XtY on GPU
    # ──────────────────────────────────────────────────────────────────────────
    print("Phase 3: Accumulating XtX and XtY on GPU...")
    XtX_gpu = None
    XtY_gpu = None

    for neural_data_item in neural_train_loader:
        with torch.no_grad():
            neural_visual, _, _, _ = process_neural_data_item(opt, neural_data_item)
            _, neural_response, _, _ = neural_data_item
            y_raw   = neural_response[opt.data_use].cpu().numpy().astype(np.float64)
            visual_p = model.input_process(neural_visual)
            out      = get_intermediate_outputs(model, visual_p, opt)

        X_gpu = get_X_gpu(out)                                  # (B, F)  float32 GPU
        y_sub = y_raw[:, voxel_indices]                         # (B, 382)
        y_t   = torch.tensor(y_sub, dtype=torch.float32, device=device)
        y_t   = (y_t - y_mean_t) / y_std_t                     # z-score on GPU

        if XtX_gpu is None:
            n_features = X_gpu.shape[1]
            XtX_gpu = torch.zeros((n_features, n_features), dtype=torch.float64, device=device)
            XtY_gpu = torch.zeros((n_features, N_VOXEL_SAMPLE), dtype=torch.float64, device=device)

        X64 = X_gpu.double()
        y64 = y_t.double()
        XtX_gpu += X64.T @ X64
        XtY_gpu += X64.T @ y64
        del X_gpu, y_t, X64, y64

    XtX_random_gpu = None
    XtY_random_gpu = None
    # if opt.split == 1 and opt.roi == "EVC":
    #     print("XtX max:", XtX_gpu.max().item(), "min:", XtX_gpu.min().item())
    #     print("XtY max:", XtY_gpu.max().item(), "min:", XtY_gpu.min().item())
    #     lo, hi = XtX_gpu.min(), XtX_gpu.max()
    #     XtX_random_gpu = torch.rand_like(XtX_gpu) * (hi - lo) + lo
    #     lo, hi = XtY_gpu.min(), XtY_gpu.max()
    #     XtY_random_gpu = torch.rand_like(XtY_gpu) * (hi - lo) + lo

    # ──────────────────────────────────────────────────────────────────────────
    # Validation data (collected once, entirely on GPU)
    # ──────────────────────────────────────────────────────────────────────────
    print("Collecting validation features on GPU...")
    all_X_val_gpu  = []
    all_y_val_raw  = []

    for val_item in neural_val_loader:
        with torch.no_grad():
            val_visual, _, _, _ = process_neural_data_item(opt, val_item)
            _, val_response, _, _ = val_item
            y_val = val_response[opt.data_use].cpu().numpy().astype(np.float64)
            visual_p = model.input_process(val_visual)
            out_val  = get_intermediate_outputs(model, visual_p, opt)

        all_X_val_gpu.append(get_X_gpu(out_val))
        all_y_val_raw.append(y_val[:, voxel_indices])

    X_val_gpu    = torch.cat(all_X_val_gpu, dim=0).double()     # (N_val, F)  GPU
    y_val_total  = np.vstack(all_y_val_raw)                     # (N_val, 382) CPU
    y_val_t      = torch.tensor(y_val_total, dtype=torch.float64, device=device)
    y_val_scaled_gpu = (y_val_t - y_mean_t.double()) / y_std_t.double()

    # ──────────────────────────────────────────────────────────────────────────
    # Alpha search (all on GPU)
    # ──────────────────────────────────────────────────────────────────────────
    alphas = np.logspace(3, 3, 1)
    best_alpha, best_r2, best_weights_gpu = None, -np.inf, None
    eye_gpu = torch.eye(n_features, dtype=torch.float64, device=device)

    print("Selecting best alpha via validation R²...")
    for alpha in alphas:
        A = XtX_gpu + alpha * eye_gpu
        W = torch.linalg.solve(A, XtY_gpu)                     # (F, V)  GPU
        pred_scaled_gpu = X_val_gpu @ W                        # (N_val, V)
        # R² on GPU
        ss_res = ((y_val_scaled_gpu - pred_scaled_gpu) ** 2).sum(0)
        ss_tot = ((y_val_scaled_gpu - y_val_scaled_gpu.mean(0)) ** 2).sum(0)
        r2_a   = (1 - ss_res / (ss_tot + 1e-10)).mean().item()
        print(f"  alpha={alpha:.1e}  R²={r2_a:.4f}")
        if r2_a > best_r2:
            best_r2, best_alpha, best_weights_gpu = r2_a, alpha, W.clone()

    print(f"Best alpha: {best_alpha:.1e}  Best R²: {best_r2:.4f}")
    all_weights_gpu = best_weights_gpu.T                        # (V, F)  GPU

    # Final predictions — unscale to original space
    pred_val_scaled_gpu = X_val_gpu @ best_weights_gpu
    pred_val_total_gpu  = pred_val_scaled_gpu * y_std_t.double() + y_mean_t.double()
    pred_val_total      = pred_val_total_gpu.cpu().numpy()

    r2_val  = float((1 - ((y_val_total - pred_val_total)**2).sum(0) /
                         ((y_val_total - y_val_total.mean(0))**2 + 1e-10).sum(0)).mean())
    print(f'R2: {r2_val:.4f}')
    plot_output(pred_val_total, y_val_total, 0, opt)

    # ──────────────────────────────────────────────────────────────────────────
    # Phase 5: Layer Contributions (GPU)
    # ──────────────────────────────────────────────────────────────────────────
    print("Phase 5: Calculating Layer Contributions...")
    contributions = {}

    # cov(pred_all, y) per voxel — denominator shared across layers
    def batch_cov(a, b):
        """Cov(a[:,v], b[:,v]) for all v simultaneously. Shape: (V,)"""
        a_c = a - a.mean(0)
        b_c = b - b.mean(0)
        return (a_c * b_c).mean(0)

    cov_total = batch_cov(pred_val_scaled_gpu, y_val_scaled_gpu)   # (V,)  GPU

    for meta in layer_metadata:
        name  = meta['name']
        s, e  = meta['start'], meta['end']

        phi_l  = X_val_gpu[:, s:e]                 # (N_val, F_l)  — PCA features for this layer only
        W_l    = all_weights_gpu[:, s:e]            # (V, F_l)      — regression weights for this layer
        pred_l = phi_l @ W_l.T                     # (N_val, V)     — this layer's contribution to prediction

        cov_layer = batch_cov(pred_l, y_val_scaled_gpu)            # (V,) How much this single layer's prediction co-varies with the true response
        mask      = cov_total.abs() > 1e-10                        # avoid dividing by ~0
        c_per_vox = torch.where(mask, cov_layer / cov_total,       # contribution for active voxels
                                torch.zeros_like(cov_layer))       # 0 for flat/dead voxels
        contributions[name] = c_per_vox.mean().item()              # average across all 382 voxels

    print(f'Contributions: {contributions}')
    print(f'Sum of contributions: {sum(contributions.values()):.6f}')

    np.savetxt(
        os.path.join(opt.result_path, f'rf_{opt.roi}_lc_{opt.data_use}_val_{opt.split}.csv'),
        [[name, contrib] for name, contrib in contributions.items()],
        delimiter=',', fmt='%s', header='layer,contribution'
    )
    np.savetxt(
        os.path.join(opt.result_path, f'rf_{opt.roi}_r2_{opt.data_use}_val_{opt.split}.csv'),
        [[r2_val]],
        delimiter=',', fmt='%s', header='r2'
    )

    # ── Optional: Random baseline (GPU) ───────────────────────────────────────
    if XtX_random_gpu is not None:
        print("Other Phase: Random Data...")
        X_val_rand_gpu = (torch.rand_like(X_val_gpu) *
                          (X_val_gpu.max() - X_val_gpu.min()) + X_val_gpu.min())
        best_alpha, best_r2, best_weights_gpu = None, -np.inf, None
        for alpha in alphas:
            A = XtX_random_gpu + alpha * eye_gpu
            W = torch.linalg.solve(A, XtY_random_gpu)
            pred_scaled_gpu = X_val_rand_gpu @ W
            ss_res = ((y_val_scaled_gpu - pred_scaled_gpu) ** 2).sum(0)
            ss_tot = ((y_val_scaled_gpu - y_val_scaled_gpu.mean(0)) ** 2).sum(0)
            r2_a   = (1 - ss_res / (ss_tot + 1e-10)).mean().item()
            print(f"  alpha={alpha:.1e}  R²={r2_a:.4f}")
            if r2_a > best_r2:
                best_r2, best_alpha, best_weights_gpu = r2_a, alpha, W.clone()

        print(f"Best alpha: {best_alpha:.1e}  Best R²: {best_r2:.4f}")
        pred_rand_scaled = X_val_rand_gpu @ best_weights_gpu
        pred_rand_total  = (pred_rand_scaled * y_std_t.double() + y_mean_t.double()).cpu().numpy()
        r2_val  = float((1 - ((y_val_total - pred_rand_total)**2).sum(0) /
                             ((y_val_total - y_val_total.mean(0))**2 + 1e-10).sum(0)).mean())
        print(f'R2: {r2_val:.4f}')
        plot_output(pred_rand_total, y_val_total, 0, opt,
                    filename=f'rf_random_data_{opt.data_use}_{opt.split}.png')
        np.savetxt(
            os.path.join(opt.result_path, f'rf_r2_{opt.data_use}_val_random_{opt.split}.csv'),
            [[r2_val]],
            delimiter=',', fmt='%s', header='r2'
        )

def run_model_get_contribution_each_roi(opt, val_loader,
                                        neural_loader_evc, neural_loader_tos,
                                        neural_loader_ppa, neural_loader_rsc,
                                        neural_loader_pfc, model):
    """
    rdm_corr version of run_model_get_contribution that loops over all five ROIs
    (evc, tos, ppa, rsc, pfc) per batch, like co_train_epoch_each_roi_add_pfc does
    for training. neural_loader_pfc may be None (or an "empty" loader) when pfc
    data isn't available for this subject/split -- pfc is skipped cleanly in that case.
    """
    print("# ---------------------------------------------------------------------- #")
    print('Getting layer contributions (per-ROI, rdm_corr)')
    model.eval()
 
    roi_loaders = {
        'evc': neural_loader_evc,
        'tos': neural_loader_tos,
        'ppa': neural_loader_ppa,
        'rsc': neural_loader_rsc,
        'pfc': neural_loader_pfc,
    }
    active_rois = [roi for roi, loader in roi_loaders.items() if loader is not None]
    skipped_rois = [roi for roi in roi_loaders if roi not in active_rois]
    if skipped_rois:
        print(f"No neural loader for {skipped_rois} -- skipping (pfc is commonly unavailable).")
    if not active_rois:
        print("No ROI loaders available at all -- nothing to do.")
        return
 
    dataloader_iterators = {roi: iter(roi_loaders[roi]) for roi in active_rois}
 
    corr_all = {roi: {} for roi in active_rois}
    norm_all = {roi: {} for roi in active_rois}
    with torch.no_grad():
        for i, train_data_item in enumerate(val_loader):
            for roi in active_rois:
                try:
                    neural_data_item = next(dataloader_iterators[roi])
                except StopIteration:
                    dataloader_iterators[roi] = iter(roi_loaders[roi])
                    neural_data_item = next(dataloader_iterators[roi])
    
                neural_visual, RSA_output, neural_batch_size, visual_item, target = process_neural_data_item(opt, neural_data_item)
    
                out = get_intermediate_outputs_v0(model, neural_visual, opt)
    
                corr = calculate_layer_correlation(neural_visual, RSA_output, out, i, target, roi, opt)
                del neural_visual, RSA_output
                layer_norms = compute_layer_rdm_norm(out, opt)
    
                for k in corr:
                    corr_all[roi][k] = corr_all[roi].get(k, 0) + corr[k]
                for k in layer_norms:
                    norm_all[roi][k] = norm_all[roi].get(k, 0) + layer_norms[k]
 
    n_batches = len(val_loader)
    header_parts = []
    row = []
    for roi in active_rois:
        for k in corr_all[roi]:
            corr_all[roi][k] /= n_batches
        for k in norm_all[roi]:
            norm_all[roi][k] /= n_batches
 
        layer_names = list(corr_all[roi].keys())
        header_parts += [f"{name}_{roi}_corr" for name in layer_names]
        header_parts += [f"{name}_{roi}_norm" for name in layer_names]
        row += [corr_all[roi][k] for k in layer_names]
        row += [norm_all[roi][k] for k in layer_names]
 
    header = ','.join(header_parts)
    np.savetxt(os.path.join(opt.result_path, f'rf_layer_corr_norm_each_roi_{opt.data_use}.csv'),
               [row], delimiter=',', fmt='%s', header=header, comments='')

def _upper_tri(mat):
    n = mat.shape[0]
    idx = torch.triu_indices(n, n, offset=1, device=mat.device)
    return mat[idx[0], idx[1]]
 
 
def compute_layer_rdm_norm(out, opt):
    """Frobenius norm of each layer's RDM (cosine, or rank-based if opt.rho4rdm)."""
    norms = {}
    with torch.no_grad():
        for k, v in out.items():
            v_flat = v.reshape(v.size(0), -1).contiguous()
            if opt.rho4rdm:
                ranks = torch.argsort(torch.argsort(v_flat, dim=1), dim=1).float()
                rdm = torch.corrcoef(ranks)
            else:
                rdm = torch.nn.functional.cosine_similarity(
                    v_flat.unsqueeze(1), v_flat.unsqueeze(0), dim=-1
                )
            rdm = torch.nan_to_num(rdm, nan=0.0)
            del v, v_flat
            norms[k] = torch.norm(rdm, p='fro').item()
    return norms

def run_model_get_contribution_per_layer(opt, neural_train_loader, neural_val_loader, model):
    """
    Fit a separate ridge regression per layer and record each layer's R² on the
    validation set.  Results are saved to:
        {opt.result_path}/{opt.roi}_{model_name}_layer_acc.csv
    """
    print("# ---------------------------------------------------------------------- #")
    print('Starting Per-Layer Ridge Regression')
    model.eval()

    # ── Config ─────────────────────────────────────────────────────────────────
    VARIANCE_TARGET    = 0.99
    N_VOXEL_SAMPLE     = 384
    IPCA_CEIL          = 64
    ALPHA              = 1e3          # fixed regularisation; add a search if needed

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    torch.backends.cudnn.deterministic = True

    voxel_indices  = None
    layer_metadata = []              # filled once during Phase 1
    pool_cache     = {}
    pca_gpu        = {}

    # ── Helpers ────────────────────────────────────────────────────────────────
    def pool_and_flatten(key, feats, pool_size=(6, 6)):
        if isinstance(feats, np.ndarray):
            feats = torch.from_numpy(feats).to(device)
        if feats.dim() == 4:
            if key not in pool_cache:
                pool_cache[key] = torch.nn.AdaptiveAvgPool2d(pool_size).to(device)
            feats = pool_cache[key](feats)
        return feats.reshape(feats.shape[0], -1)

    def pca_transform_gpu(key, feats):
        x     = pool_and_flatten(key, feats).float()
        mean  = pca_gpu[key]["mean"]
        comps = pca_gpu[key]["components"]          # (n_keep, n_features)
        return (x - mean) @ comps.T                 # (B, n_keep)

    # ──────────────────────────────────────────────────────────────────────────
    # Phase 1: Collect features and fit per-layer PCA
    # ──────────────────────────────────────────────────────────────────────────
    print("Phase 1: Fitting per-layer GPU PCA...")
    feat_buffer_gpu = {}

    for neural_data_item in neural_train_loader:
        with torch.no_grad():
            neural_visual, _, _, _ = process_neural_data_item(opt, neural_data_item)
            visual_p = model.input_process(neural_visual)
            out = get_intermediate_outputs(model, visual_p, opt)

        for key, feats in out.items():
            feats_flat = pool_and_flatten(key, feats)
            feat_buffer_gpu.setdefault(key, []).append(feats_flat.float())

    for key, chunks in feat_buffer_gpu.items():
        X      = torch.cat(chunks, dim=0)
        X_mean = X.mean(0, keepdim=True)
        X_c    = X - X_mean

        U, S, V = torch.pca_lowrank(X_c, q=min(IPCA_CEIL, X.shape[1]),
                                    center=False, niter=4)
        var    = (S ** 2) / (X.shape[0] - 1)
        cumvar = torch.cumsum(var, dim=0) / var.sum()
        n_keep = int((cumvar < VARIANCE_TARGET).sum().item()) + 1
        n_keep = min(n_keep, len(cumvar))
        print(f"  {key}: {n_keep}/{len(cumvar)} PCs "
              f"({cumvar[n_keep-1].item()*100:.2f}% variance)")

        pca_gpu[key] = {
            "mean":       X_mean.squeeze(0),
            "components": V[:, :n_keep].T,
        }
        del X, X_c, U, S, V, chunks
        torch.cuda.empty_cache()

    feat_buffer_gpu.clear()

    # ──────────────────────────────────────────────────────────────────────────
    # Phase 2: Fit per-layer StandardScaler + voxel sampling
    # ──────────────────────────────────────────────────────────────────────────
    print("Phase 2: Fitting per-layer scalers and sampling voxels...")
    layer_scalers = {}   # key → StandardScaler

    for i, neural_data_item in enumerate(neural_train_loader):
        with torch.no_grad():
            neural_visual, _, _, _ = process_neural_data_item(opt, neural_data_item)
            _, neural_response, _, _ = neural_data_item
            y_raw    = neural_response[opt.data_use].cpu().numpy().astype(np.float64)
            visual_p = model.input_process(neural_visual)
            out      = get_intermediate_outputs(model, visual_p, opt)

        # One-time: sample voxels + record layer metadata
        if voxel_indices is None:
            total_voxels = y_raw.shape[1]
            assert total_voxels >= N_VOXEL_SAMPLE
            rng           = np.random.default_rng(seed=42)
            voxel_indices = np.sort(rng.choice(total_voxels, size=N_VOXEL_SAMPLE, replace=False))
            print(f"  Sampled {N_VOXEL_SAMPLE} voxels from {total_voxels} total.")

            col = 0
            for key, feats in out.items():
                pca_feats = pca_transform_gpu(key, feats)
                n_comp    = pca_feats.shape[1]
                layer_metadata.append({"name": key, "start": col, "end": col + n_comp})
                col += n_comp
                layer_scalers[key] = StandardScaler()

        for key, feats in out.items():
            feats_pca = pca_transform_gpu(key, feats).cpu().numpy()
            layer_scalers[key].partial_fit(feats_pca)

    # Move per-layer scaler params to GPU tensors for fast transform
    scaler_params_gpu = {}
    for key, sc in layer_scalers.items():
        scaler_params_gpu[key] = {
            "mean":  torch.tensor(sc.mean_,  dtype=torch.float32, device=device),
            "scale": torch.tensor(sc.scale_, dtype=torch.float32, device=device),
        }

    def get_layer_X_gpu(key, feats):
        """PCA-project + z-score for a single layer. Returns (B, n_keep) GPU tensor."""
        x     = pca_transform_gpu(key, feats).float()
        mean  = scaler_params_gpu[key]["mean"]
        scale = scaler_params_gpu[key]["scale"]
        return ((x - mean) / scale).double()

    # ══════════════════════════════════════════════════════════════════
    # Phase 3: Per-layer XtX / XtY accumulation  （加入 y_mean 追蹤）
    # ══════════════════════════════════════════════════════════════════
    print("Phase 3: Accumulating per-layer XtX and XtY...")
    XtX_per_layer = {}
    XtY_per_layer = {}
    y_sum_accum   = None      # ★ 新增
    N_train_total = 0          # ★ 新增

    for neural_data_item in neural_train_loader:
        with torch.no_grad():
            neural_visual, _, _, _ = process_neural_data_item(opt, neural_data_item)
            _, neural_response, _, _ = neural_data_item
            y_raw    = neural_response[opt.data_use].cpu().numpy().astype(np.float64)
            visual_p = model.input_process(neural_visual)
            out      = get_intermediate_outputs(model, visual_p, opt)

        y_sub = y_raw[:, voxel_indices]
        y_t   = torch.tensor(y_sub, dtype=torch.float64, device=device)

        # ★ 累積 y 總和，之後算截距用
        if y_sum_accum is None:
            y_sum_accum = y_t.sum(0)
        else:
            y_sum_accum += y_t.sum(0)
        N_train_total += y_t.shape[0]

        for key, feats in out.items():
            X_l = get_layer_X_gpu(key, feats)
            # print(f"train_X scale: mean={X_l.abs().mean():.4f}, std={X_l.std():.4f}")
            # sys.exit()
            if key not in XtX_per_layer:
                F_l = X_l.shape[1]
                XtX_per_layer[key] = torch.zeros((F_l, F_l),            dtype=torch.float64, device=device)
                XtY_per_layer[key] = torch.zeros((F_l, N_VOXEL_SAMPLE), dtype=torch.float64, device=device)
            XtX_per_layer[key] += X_l.T @ X_l
            XtY_per_layer[key] += X_l.T @ y_t
        del y_t

    # ★ 計算訓練集 Y 的均值（截距）
    y_train_mean = y_sum_accum / N_train_total   # (N_VOXEL_SAMPLE,)
    # print(f"y_train_mean: mean={y_train_mean.mean():.2f}, std={y_train_mean.std():.2f}")

    # ──────────────────────────────────────────────────────────────────────────
    # Phase 4: Collect validation features (per layer) + y
    # ──────────────────────────────────────────────────────────────────────────
    print("Collecting validation features...")
    val_X_per_layer = {m["name"]: [] for m in layer_metadata}
    all_y_val_raw   = []

    for val_item in neural_val_loader:
        with torch.no_grad():
            val_visual, _, _, _ = process_neural_data_item(opt, val_item)
            _, val_response, _, _ = val_item
            y_val    = val_response[opt.data_use].cpu().numpy().astype(np.float64)
            visual_p = model.input_process(val_visual)
            out_val  = get_intermediate_outputs(model, visual_p, opt)

        for key, feats in out_val.items():
            val_X_per_layer[key].append(get_layer_X_gpu(key, feats))
        all_y_val_raw.append(y_val[:, voxel_indices])

    # Concatenate validation tensors
    for key in val_X_per_layer:
        val_X_per_layer[key] = torch.cat(val_X_per_layer[key], dim=0)  # (N_val, F_l)

    y_val_total = np.vstack(all_y_val_raw)                          # (N_val, V) CPU
    y_val_gpu   = torch.tensor(y_val_total, dtype=torch.float64, device=device)  # raw

    # ──────────────────────────────────────────────────────────────────────────
    # Phase 5: Solve per-layer ridge regression and evaluate R²
    # ──────────────────────────────────────────────────────────────────────────
    print(f"Phase 5: Solving per-layer ridge regression (alpha={ALPHA:.1e})...")
    layer_r2 = {}
    alphas_search = np.logspace(-2, 6, 9)   # relative to trace scale

    for meta in layer_metadata:
        ALPHA = None
        name  = meta["name"]
        F_l   = XtX_per_layer[name].shape[0]
        trace = torch.trace(XtX_per_layer[name]).item()
        eye   = torch.eye(F_l, dtype=torch.float64, device=device)

        best_r2_l, best_W_l = -np.inf, None
        for a_rel in alphas_search:
            alpha_l = a_rel * trace / F_l
            W       = torch.linalg.solve(XtX_per_layer[name] + alpha_l * eye,
                                        XtY_per_layer[name])
            pred    = val_X_per_layer[name] @ W + y_train_mean   # ★ 加截距
            ss_res  = ((y_val_gpu - pred) ** 2).sum(0)
            ss_tot  = ((y_val_gpu - y_val_gpu.mean(0)) ** 2).sum(0)
            r2      = (1 - ss_res / (ss_tot + 1e-10)).mean().item()
            if r2 > best_r2_l:
                best_r2_l, best_W_l = r2, W.clone()
                ALPHA = alpha_l
            del W, pred

        layer_r2[name] = best_r2_l
        print(f"  {name}: best R² = {best_r2_l:.4f}")
        # Add this diagnostic right before the solve in Phase 5
        # eigvals = torch.linalg.eigvalsh(XtX_per_layer[name])
        # print(f"{name}: min_eig={eigvals.min():.2f}, max_eig={eigvals.max():.2f}, "
        #     f"cond={eigvals.max()/eigvals.min():.1f}, "
        #     f"alpha/max_eig ratio={ALPHA/eigvals.max().item():.4f}")
        # print(f"val_X scale: mean={val_X_per_layer[name].abs().mean():.4f}, std={val_X_per_layer[name].std():.4f}")
        # print(f"y_val scale: mean={y_val_gpu.abs().mean():.4f},  std={y_val_gpu.std():.4f}")

    # for meta in layer_metadata:
    #     name = meta["name"]
    #     F_l  = XtX_per_layer[name].shape[0]

    #     eye  = torch.eye(F_l, dtype=torch.float64, device=device)
    #     A    = XtX_per_layer[name] + ALPHA * eye
    #     W    = torch.linalg.solve(A, XtY_per_layer[name])              # (F_l, V)

    #     X_val_l  = val_X_per_layer[name]                               # (N_val, F_l)
    #     pred     = X_val_l @ W                                         # (N_val, V)  raw scale

    #     ss_res   = ((y_val_gpu - pred) ** 2).sum(0)
    #     ss_tot   = ((y_val_gpu - y_val_gpu.mean(0)) ** 2).sum(0)
    #     r2_layer = (1 - ss_res / (ss_tot + 1e-10)).mean().item()

    #     layer_r2[name] = r2_layer
    #     print(f"  {name}: R² = {r2_layer:.4f}")

        del ss_res, ss_tot, eye
        torch.cuda.empty_cache()

    # ──────────────────────────────────────────────────────────────────────────
    # Save results
    # ──────────────────────────────────────────────────────────────────────────
    import csv
    out_path = os.path.join(opt.result_path, f'raw_fmri_{opt.roi}_layer_r2_{opt.data_use}.csv')
    with open(out_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['layer_name', 'r2'])
        for name, r2 in layer_r2.items():
            writer.writerow([name, f'{r2:.6f}'])

    print(f"\nSaved per-layer R² → {out_path}")
    print(f"Layer accuracies: {layer_r2}")
    # return layer_r2

def apply_paper_pca(feat_matrix, variance_threshold=0.99):
    """
    Python implementation of the paper's PCA logic.
    """
    # 1. Standardize (z-score)
    f_mean = np.mean(feat_matrix, axis=0)
    f_std = np.std(feat_matrix, axis=0, ddof=0)
    # Avoid division by zero
    f_std[f_std < 1e-8] = 1.0
    
    X = (feat_matrix - f_mean) / f_std
    X = np.nan_to_num(X) # Assign 0 to NaNs

    N, D = X.shape

    if N > D:
        # Case: More samples than features
        R = np.dot(X.T, X) / N
        U, s, _ = np.linalg.svd(R) # s is eigenvalues/singular values
        
        # Keep components for 99% variance
        ratio = np.cumsum(s) / np.sum(s)
        nc = np.where(ratio > variance_threshold)[0][0] + 1
        
        # S_2 = diag(1./sqrt(s(1:Nc)))
        s_2 = np.diag(1.0 / np.sqrt(s[:nc]))
        
        # B = lay_feat_cont * (U(:,1:Nc) * S_2 / sqrt(N))
        projection = np.dot(U[:, :nc], s_2) / np.sqrt(N)
        B = np.dot(X, projection)
    else:
        # Case: More features than samples (Dual PCA)
        R = np.dot(X, X.T)
        U, s, _ = np.linalg.svd(R)
        
        ratio = np.cumsum(s) / np.sum(s)
        nc = np.where(ratio > variance_threshold)[0][0] + 1
        
        # B = U(:, 1:Nc)
        B = U[:, :nc]
        projection = None # In this case, B is the representation directly

    return B, projection, f_mean, f_std

def get_intermediate_outputs(model, visual_p, opt):
    if opt.network_choose == 'resnet_18':
        new_m = torchvision.models._utils.IntermediateLayerGetter(model.resnet, {'0': 'conv1', '4': 'conv5','5':'conv9','6':'conv13','7':'conv17'})
        out = new_m(visual_p)
    elif opt.network_choose == 'squeezenet':
        new_m = torchvision.models._utils.IntermediateLayerGetter(model.CNN.features,{'0': 'conv1', '6': 'Fire3', '9': 'Fire5','12': 'Fire7', '14': 'Fire9'})
        out = new_m(visual_p)
    elif opt.network_choose == 'shufflenet_v1':
        new_m = torchvision.models._utils.IntermediateLayerGetter(model.CNN,{'conv1': 'conv1', 'layer1': 'layer1', 'layer2':'layer2','layer3': 'layer3'})
        out = new_m(visual_p)
    elif opt.network_choose == 'shufflenet_v2':
        new_m = torchvision.models._utils.IntermediateLayerGetter(model.CNN,{'conv1': 'conv1', 'conv_last':'conv_last'})
        new_m_features = torchvision.models._utils.IntermediateLayerGetter(model.CNN.features,{'3': 'features3', '11': 'features11', '15': 'features15'})
        # new_m = torchvision.models._utils.IntermediateLayerGetter(model.CNN,{'conv1': 'conv1', 'features': 'features', 'conv_last':'conv_last'})
        out = new_m(visual_p)
        x = model.CNN.conv1(visual_p) 
        # Pass through maxpool
        x = model.CNN.maxpool(x)
        out_features = new_m_features(x)
        out.update(out_features)
        # order the key as 'conv1', 'features3', 'features11', 'features15', 'conv_last'
        desired_order = ['conv1', 'features3', 'features11', 'features15', 'conv_last']
        # Rebuild the dictionary in the new order
        out = OrderedDict((k, out[k]) for k in desired_order)
        # print("shufflenet_v2's out.keys():",out.keys())
    elif opt.network_choose == 'mobilenet_v1':
        new_m = torchvision.models._utils.IntermediateLayerGetter(model.CNN.features,{'0': '64channels', '1': '128channels', '3': '256channels','5': '512channels', '11': '1024channels','13':'2048channels'})
        out = new_m(visual_p)
    elif opt.network_choose == 'mobilenet_v2':
        new_m = torchvision.models._utils.IntermediateLayerGetter(model.CNN.features,{'0': '14channels', '1': '7channels', '2': '10channels', '4': '14channels', '7': '28channels', '11': '43channels', '14': '72channels', '17': '144channels','18': '1280channels'})
        # new_m = torchvision.models._utils.IntermediateLayerGetter(model.CNN.features,{'0': '14channels', '17': '144channels','18': '1280channels'})
        out = new_m(visual_p)
    return out

def get_intermediate_outputs_v0(model, visual, opt):
    if opt.network_choose != 'vaa':
        visual_p = model.input_process(visual)
    if opt.network_choose == 'resnet_18':
        new_m = torchvision.models._utils.IntermediateLayerGetter(model.resnet, {'0': 'conv1', '4': 'conv5','5':'conv9','6':'conv13','7':'conv17'})
        out = new_m(visual_p)
    elif opt.network_choose == 'squeezenet':
        new_m = torchvision.models._utils.IntermediateLayerGetter(model.CNN.features,{'0': 'conv1', '6': 'Fire3', '9': 'Fire5','12': 'Fire7', '14': 'Fire9'})
        out = new_m(visual_p)
    elif opt.network_choose == 'shufflenet_v1':
        new_m = torchvision.models._utils.IntermediateLayerGetter(model.CNN,{'conv1': 'conv1', 'layer1': 'layer1', 'layer2':'layer2','layer3': 'layer3'})
        out = new_m(visual_p)
    elif opt.network_choose == 'shufflenet_v2':
        new_m = torchvision.models._utils.IntermediateLayerGetter(model.CNN,{'conv1': 'conv1', 'features': 'features', 'conv_last':'conv_last'})
        out = new_m(visual_p)
    elif opt.network_choose == 'mobilenet_v1':
        new_m = torchvision.models._utils.IntermediateLayerGetter(model.CNN.features,{'0': '64channels', '1': '128channels', '3': '256channels','5': '512channels', '11': '1024channels','13':'2048channels'})
        out = new_m(visual_p)
    elif opt.network_choose == 'mobilenet_v2':
        new_m = torchvision.models._utils.IntermediateLayerGetter(model.CNN.features,{'0': '14channels', '17': '144channels','18': '1280channels'})
        out = new_m(visual_p)
    elif opt.network_choose == 'vit_3d':
        tokens = model.CNN.embed(visual_p)
        new_m = torchvision.models._utils.IntermediateLayerGetter(
            model.CNN.layers,
            {'encoder_layer_2': 'block3', 'encoder_layer_5': 'block6',
            'encoder_layer_8': 'block9', 'encoder_layer_11': 'block12'}
        )
        out = new_m(tokens)
    elif opt.network_choose == 'video_swin':
        tokens = model.CNN.embed(visual_p)
        new_m = torchvision.models._utils.IntermediateLayerGetter(
            model.CNN.features,
            {'0': 'stage1', '2': 'stage2', '4': 'stage3', '6': 'stage4'}
        )
        out = new_m(tokens)
    
    return out

def run_model_contribution(opt, inputs, model):
    neural_visual, voxel_select = inputs

    contribution_all = {}
    mse_all = 0

    if opt.network_choose != 'vaa':
        visual_p = model.input_process(neural_visual)
    if opt.network_choose == 'resnet_18':
        new_m = torchvision.models._utils.IntermediateLayerGetter(model.resnet, {'0': 'conv1', '4': 'conv5','5':'conv9','6':'conv13','7':'conv17'})
        out = new_m(visual_p)
    elif opt.network_choose == 'squeezenet':
        new_m = torchvision.models._utils.IntermediateLayerGetter(model.CNN.features,{'0': 'conv1', '6': 'Fire3', '9': 'Fire5','12': 'Fire7', '14': 'Fire9'})
        out = new_m(visual_p)
    elif opt.network_choose == 'shufflenet_v1':
        new_m = torchvision.models._utils.IntermediateLayerGetter(model.CNN,{'conv1': 'conv1', 'layer1': 'layer1', 'layer2':'layer2','layer3': 'layer3'})
        out = new_m(visual_p)
    elif opt.network_choose == 'shufflenet_v2':
        new_m = torchvision.models._utils.IntermediateLayerGetter(model.CNN,{'conv1': 'conv1', 'features': 'features', 'conv_last':'conv_last'})
        out = new_m(visual_p)
    elif opt.network_choose == 'mobilenet_v1':
        new_m = torchvision.models._utils.IntermediateLayerGetter(model.CNN.features,{'0': '64channels', '1': '128channels', '3': '256channels','5': '512channels', '11': '1024channels','13':'2048channels'})
        out = new_m(visual_p)
    elif opt.network_choose == 'mobilenet_v2':
        new_m = torchvision.models._utils.IntermediateLayerGetter(model.CNN.features,{'0': '14channels', '17': '144channels','18': '1280channels'})
        out = new_m(visual_p)

    output, contribution, mse, r2 = calculate_layer_contributions(voxel_select, out, opt)
    

    return output, mse, contribution

def plot_output(output, voxel_select, epoch, opt, filename=None):
    # output = output.detach().cpu().numpy()
    # voxel_select = voxel_select.detach().cpu().numpy()
    # plot output and voxel_select in a same figure
    plt.figure()
    # line width thinner
    plt.plot(output[0], linewidth=0.5)
    plt.plot(voxel_select[0], linewidth=0.5)
    plt.legend(['output', 'voxel_select'])
    plt.xlabel('voxel')
    plt.ylabel('value')
    plt.title(f'output and voxel_select {opt.roi}')
    save_path = os.path.join(opt.result_path, f'output_and_voxel_select/')
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    # plt.savefig(os.path.join(save_path, f'raw_fmri_{opt.roi}_{opt.data_use}_{opt.split}_v3.png'))
    if filename is not None:
        plt.savefig(os.path.join(save_path, filename))
    else:
        plt.savefig(os.path.join(save_path, f'raw_fmri_{opt.roi}_{opt.split}_val.png'))
    plt.close()

def run_model_iscience(opt, inputs, model, criterion=None, i=0, print_attention=False, period=30, return_attention=False,test_svm=False):
    if not test_svm:
        visual, target = inputs
        outputs = model(visual)
        y_pred, alpha, beta, gamma,fSCT = outputs
        loss = criterion(y_pred, target)
        if i % period == 0 and print_attention:
            print('====alpha====')
            print(alpha[:, 0, :])
            print('====beta====')
            print(beta[:, 0, 0:512:32])
            print('====gamma====')
            print(gamma)
        if not return_attention:
            return y_pred, loss
        else:
            return y_pred, loss, [alpha, beta, gamma]
    else:
        visual = inputs
        outputs = model(visual,test_svm=test_svm)
        y_pred, alpha, beta, gamma, fSCT = outputs
        return fSCT

def run_neural_model(opt,inputs, model,print_gamma=False):
    remove = 0
    visual,RSA_target = inputs
    # if isinstance(model, SingleClassWrapper):
    #     model = model.model
    # print("RSA_target:", RSA_target)
    if opt.network_choose != 'vaa':
        visual_p = model.input_process(visual)
    if opt.network_choose == 'resnet_18':
        new_m = torchvision.models._utils.IntermediateLayerGetter(model.resnet, {'0': 'conv1', '4': 'conv5','5':'conv9','6':'conv13','7':'conv17'})
        out = new_m(visual_p)
    elif opt.network_choose == 'squeezenet':
        new_m = torchvision.models._utils.IntermediateLayerGetter(model.CNN.features,{'0': 'conv1', '6': 'Fire3', '9': 'Fire5','12': 'Fire7', '14': 'Fire9'})
        out = new_m(visual_p)
    elif opt.network_choose == 'shufflenet_v1':
        new_m = torchvision.models._utils.IntermediateLayerGetter(model.CNN,{'conv1': 'conv1', 'layer1': 'layer1', 'layer2':'layer2','layer3': 'layer3'})
        out = new_m(visual_p)
    elif opt.network_choose == 'shufflenet_v2':
        new_m = torchvision.models._utils.IntermediateLayerGetter(model.CNN,{'conv1': 'conv1', 'features': 'features', 'conv_last':'conv_last'})
        out = new_m(visual_p)
    elif opt.network_choose == 'mobilenet_v1':
        new_m = torchvision.models._utils.IntermediateLayerGetter(model.CNN.features,{'0': '64channels', '1': '128channels', '3': '256channels','5': '512channels', '11': '1024channels','13':'2048channels'})
        out = new_m(visual_p)
    elif opt.network_choose == 'mobilenet_v2':
        new_m = torchvision.models._utils.IntermediateLayerGetter(model.CNN.features,{'0': '14channels', '17': '144channels','18': '1280channels'})
        out = new_m(visual_p)
    elif opt.network_choose == 'vaa':
        out = {}
        sa,cwa,ta = model(visual,vaa=True)
        out['spatial_attention'] = sa
        out['channel_wise_attention'] = cwa
        out['temporal_attention'] = ta
    elif opt.network_choose == 'vit_3d':
        tokens = model.CNN.embed(visual_p)
        new_m = torchvision.models._utils.IntermediateLayerGetter(
            model.CNN.layers,
            {'encoder_layer_2': 'block3', 'encoder_layer_5': 'block6',
            'encoder_layer_8': 'block9', 'encoder_layer_11': 'block12'}
        )
        out = new_m(tokens)
    elif opt.network_choose == 'video_swin':
        tokens = model.CNN.embed(visual_p)
        new_m = torchvision.models._utils.IntermediateLayerGetter(
            model.CNN.features,
            {'0': 'stage1', '2': 'stage2', '4': 'stage3', '6': 'stage4'}
        )
        out = new_m(tokens)
    S_k = []
    for k,v in out.items():
        if k != 'temporal_attention':
            if opt.network_choose == 'vit_3d':
                v = v.reshape(visual.size(1),visual.size(0),-1).contiguous()
            else:
                v = v.view(visual.size(1),visual.size(0),-1).contiguous()
            v = torch.mean(v,dim=0)
        v = v-torch.mean(v,dim=0)

        v1 = v.detach()
        v2 = v1.cuda(0)
        del v
        del v1
        torch.cuda.empty_cache()
        if opt.rho4rdm:
            # v2_numpy = v2.cpu().numpy()
            # ranks_numpy = rankdata(v2_numpy, method='average', axis=1)
            # del v2_numpy
            # v2_ranks = torch.tensor(ranks_numpy, dtype=torch.float32, device=v2.device)
            # del ranks_numpy
            # b = torch.corrcoef(v2_ranks)
            # del v2_ranks
            v2_ranks = torch.argsort(torch.argsort(v2, dim=1), dim=1).float()
            b = torch.corrcoef(v2_ranks)
            del v2_ranks
        else:
            b = torch.nn.functional.cosine_similarity(v2.unsqueeze(1), v2.unsqueeze(0), dim=-1)
        torch.cuda.empty_cache()
        b1 = b.cuda(0)
        del b
        del v2
        S_k.append(b1)
    gamma = F.softmax(model.gamma,dim=0)
    if print_gamma:
        print(gamma)
    S_cnn = 0
    for i in range(len(S_k)):
        # S_cnn = S_cnn + S_k[i] # 0923 without gamma
        S_cnn = S_cnn + gamma[i]*S_k[i]
    # S_cnn = S_cnn/len(S_k) # 0923 without gamma
    # if opt.use_lstm: # correct the nan value appears
    #     epsilon = 1e-6
    #     S_cnn_clamped = torch.clamp(S_cnn-torch.diag_embed(torch.diag(S_cnn)), min=-1 + epsilon, max=1 - epsilon)
    #     RSA_target_clamped = torch.clamp(RSA_target-torch.diag_embed(torch.diag(RSA_target)), min=-1 + epsilon, max=1 - epsilon)
    #     loss = torch.atanh(S_cnn_clamped)-torch.atanh(RSA_target_clamped)
    # else:
    #     loss = torch.atanh(S_cnn-torch.diag_embed(torch.diag(S_cnn)))-torch.atanh(RSA_target-torch.diag_embed(torch.diag(RSA_target)))
    loss = torch.atanh(S_cnn-torch.diag_embed(torch.diag(S_cnn)))-torch.atanh(RSA_target-torch.diag_embed(torch.diag(RSA_target)))
    if len(torch.where(torch.isinf(loss))[0])>0:
        print('inf appears.')
        remove = len(torch.where(torch.isinf(loss))[0])
        print('remove=',remove)
        print('position',torch.where(torch.isinf(loss)))
        cnn_f = S_cnn-torch.diag_embed(torch.diag(S_cnn))
        rsa = RSA_target-torch.diag_embed(torch.diag(RSA_target))
        print(cnn_f[torch.where(torch.isinf(loss))])
        print(rsa[torch.where(torch.isinf(loss))])
        loss = torch.where(torch.isinf(loss), torch.full_like(loss, 0), loss)

    if len(torch.where(torch.isnan(loss))[0])>0:
        print('nan appears.')
        remove = len(torch.where(torch.isnan(loss))[0])
        print('remove=',remove)
        print('position',torch.where(torch.isnan(loss)))
        cnn_f = S_cnn-torch.diag_embed(torch.diag(S_cnn))
        rsa = RSA_target-torch.diag_embed(torch.diag(RSA_target))
        print(cnn_f[torch.where(torch.isnan(loss))])
        print(rsa[torch.where(torch.isnan(loss))])
        loss = torch.where(torch.isnan(loss), torch.full_like(loss, 0), loss)

    loss = torch.pow(loss,2)
    num = (visual.size(0)*(visual.size(0)-1))/2
    loss = loss.sum()/2/(num-remove//2)
    if opt.add_mse:
        # print('-----add_mse-----')
        # Get the number of conditions (N)
        n = S_cnn.shape[0]

        # Get indices for the upper triangle, excluding the diagonal (k=1)
        # This selects all unique pairwise dissimilarities
        upper_triangle_indices = np.triu_indices(n, k=1)

        # Extract the values from the upper triangles
        S_cnn_values = S_cnn[upper_triangle_indices]
        RSA_target_values = RSA_target[upper_triangle_indices]

        # Calculate the squared error for each pair
        squared_errors = (S_cnn_values - RSA_target_values) ** 2

        # Calculate the mean of the squared errors
        mse_loss = np.mean(squared_errors.cpu().detach().numpy())
        loss = loss + mse_loss

    return  gamma, loss, S_cnn, RSA_target

def run_neural_model_dapello(opt,inputs, model,print_gamma=False):
    # only align last layer with combined rois
    remove = 0
    visual,RSA_target = inputs
    # if isinstance(model, SingleClassWrapper):
    #     model = model.model
    # print("RSA_target shape:",RSA_target.shape)
    if opt.network_choose != 'vaa':
        visual_p = model.input_process(visual)
    if opt.network_choose == 'resnet_18':
        new_m = torchvision.models._utils.IntermediateLayerGetter(model.resnet, {'7':'conv17'})
        out = new_m(visual_p)
    elif opt.network_choose == 'squeezenet':
        new_m = torchvision.models._utils.IntermediateLayerGetter(model.CNN.features,{'14': 'Fire9'})
        out = new_m(visual_p)
    elif opt.network_choose == 'shufflenet_v1':
        new_m = torchvision.models._utils.IntermediateLayerGetter(model.CNN,{'layer3': 'layer3'})
        out = new_m(visual_p)
    elif opt.network_choose == 'shufflenet_v2':
        new_m = torchvision.models._utils.IntermediateLayerGetter(model.CNN,{'conv_last':'conv_last'})
        out = new_m(visual_p)
    elif opt.network_choose == 'mobilenet_v1':
        new_m = torchvision.models._utils.IntermediateLayerGetter(model.CNN.features,{'13':'2048channels'})
        out = new_m(visual_p)
    elif opt.network_choose == 'mobilenet_v2':
        new_m = torchvision.models._utils.IntermediateLayerGetter(model.CNN.features,{'18': '1280channels'})
        out = new_m(visual_p)
    elif opt.network_choose == 'vaa':
        out = {}
        sa,cwa,ta = model(visual,vaa=True)
        out['spatial_attention'] = sa
        out['channel_wise_attention'] = cwa
        out['temporal_attention'] = ta
    elif opt.network_choose == 'vit_3d':
        tokens = model.CNN.embed(visual_p)
        new_m = torchvision.models._utils.IntermediateLayerGetter(
            model.CNN.layers, {'encoder_layer_11': 'block12'}
        )
        out = new_m(tokens)
    elif opt.network_choose == 'video_swin':
        tokens = model.CNN.embed(visual_p)
        new_m = torchvision.models._utils.IntermediateLayerGetter(
            model.CNN.features, {'6': 'stage4'}
        )
        out = new_m(tokens)
    # --- extract activations, flatten to (n_samples, n_features) ---
    X = None
    for k, v in out.items():
        if k != 'temporal_attention':
            if opt.network_choose == 'video_swin':
                v = v.mean(dim=(1, 2, 3))  # average over T', H', W' -> (seq_len*batch, 768)
                v = v.reshape(visual.size(1), visual.size(0), -1).contiguous()
            elif opt.network_choose == 'vit_3d':
                v = v.mean(dim=1)  # (seq_len*batch, 768) — average over all tokens to prevent cuda memory error
                v = v.reshape(visual.size(1),visual.size(0),-1).contiguous()
            else:
                v = v.view(visual.size(1), visual.size(0), -1).contiguous()
            v = torch.mean(v, dim=0)          # (n_samples, n_features)
        X = v.detach().cuda(0)
        del v
        torch.cuda.empty_cache()

    Y = RSA_target  # neural activations, same (n_samples, n_neurons) convention

    # --- exact Dapello et al. logCKA ---
    X = X.view(X.shape[0], -1)
    Y = Y.view(Y.shape[0], -1)
    X = X.float()
    Y = Y.float()
    X = X - X.mean(dim=0)
    Y = Y - Y.mean(dim=0)

    def frobdot(A, B):
        return torch.norm(torch.matmul(B.t(), A), p='fro')

    CKA_val = frobdot(X, Y) ** 2 / (frobdot(X, X) * frobdot(Y, Y))
    loss = torch.log(1 - CKA_val)

    return loss, X, Y

def run_neural_model_v2(opt,inputs, model,epoch,print_gamma=False):
    remove = 0
    visual,voxel_select = inputs

    # print("visual.shape",visual.shape)
    if epoch < int(opt.mixup_pct * opt.n_epochs):
        visual, perm, betas, select = mixco(visual)

    if opt.network_choose != 'vaa':
        visual_p = model.input_process(visual)

    if opt.network_choose in ['resnet_18', 'mobilenet_v1', 'mobilenet_v2']:
        if hasattr(model, 'features'):
            deep_features = model.features(visual_p)
        elif hasattr(model, 'resnet'):
            deep_features = model.resnet(visual_p)
        elif hasattr(model.CNN, 'features'): # For nested models like SqueezeNet
            deep_features = model.CNN.features(visual_p)
        else:
            raise AttributeError("Could not find the model's feature extraction backbone.")
        # print(deep_features.size())
        deep_features = F.avg_pool3d(deep_features, deep_features.data.size()[-3:])
        visual_p = deep_features.view(deep_features.size(0), -1)
        del deep_features
        torch.cuda.empty_cache()
    elif opt.network_choose in ['squeezenet']:
        if hasattr(model, 'features'):
            deep_features = model.features(visual_p)
        elif hasattr(model, 'resnet'):
            deep_features = model.resnet(visual_p)
        elif hasattr(model.CNN, 'features'): # For nested models like SqueezeNet
            deep_features = model.CNN.features(visual_p)
        else:
            raise AttributeError("Could not find the model's feature extraction backbone.")
        visual_p = deep_features
        del deep_features
        torch.cuda.empty_cache()
    elif opt.network_choose in ['shufflenet_v1']:
        deep_features = model.CNN.conv1(visual_p)
        deep_features = model.CNN.maxpool(deep_features)
        deep_features = model.CNN.layer1(deep_features)
        deep_features = model.CNN.layer2(deep_features)
        deep_features = model.CNN.layer3(deep_features)
        deep_features = F.avg_pool3d(deep_features, deep_features.data.size()[-3:])
        visual_p = deep_features.view(deep_features.size(0), -1)
        del deep_features
        torch.cuda.empty_cache()
    elif opt.network_choose in ['shufflenet_v2']:
        deep_features = model.CNN.conv1(visual_p)
        deep_features = model.CNN.maxpool(deep_features)
        deep_features = model.CNN.features(deep_features)
        deep_features = model.CNN.conv_last(deep_features)
        deep_features = F.avg_pool3d(deep_features, deep_features.data.size()[-3:])
        visual_p = deep_features.view(deep_features.size(0), -1)
        del deep_features
        torch.cuda.empty_cache()
    # elif opt.network_choose == 'resnet_18':
    #     if hasattr(model, 'resnet'): # Common for ResNet models
    #         deep_features = model.resnet(visual_p)
    
    if opt.network_choose == 'resnet_18':
        # print(model.fc)
        new_m = torchvision.models._utils.IntermediateLayerGetter(model.fc, {'0': 'projection_output'})
        out = new_m(visual_p)
    elif opt.network_choose in ['squeezenet', 'shufflenet_v1', 'shufflenet_v2', 'mobilenet_v1', 'mobilenet_v2']:
        new_m = torchvision.models._utils.IntermediateLayerGetter(model.CNN.classifier,{'1': 'projection_output'})
        out = new_m(visual_p)
    # The output 'out' is a dictionary: {'projection_output': tensor}
    visual_embeddings = out['projection_output']
    # print("visual_embeddings.shape:",visual_embeddings.shape)
    del visual_p
    torch.cuda.empty_cache()
    
    # --- 2. Process Embeddings to Match Target Shape (batch, 128) ---
    # The feature extractor might have extra dimensions (Time, H, W)
    # We average them out to get a single vector per batch item.
    if visual_embeddings.dim() > 2:
        # 1. Apply global average pooling to collapse the T, H, W dimensions
        #    The output shape will be (16, 128, 1, 1, 1)
        pooled_embeddings = F.adaptive_avg_pool3d(visual_embeddings, (1, 1, 1))

        # 2. Flatten the result to remove the trailing '1' dimensions
        #    The output shape will be (16, 128)
        visual_embeddings = torch.flatten(pooled_embeddings, start_dim=1)

    
    
    if visual_embeddings.shape[0] != opt.batch_size:
        # At this point:
        # visual_embeddings.shape is (160, 128)
        # voxel_select.shape is (16, 128)
        # Get the original batch_size from the target tensor
        batch_size = voxel_select.shape[0]
        # 1. Reshape to re-introduce the sequence dimension
        #    Shape becomes (batch_size, seq_len, 128) -> (16, 10, 128)
        #    The -1 automatically infers the seq_len (10)
        visual_embeddings = visual_embeddings.view(batch_size, -1, 2304) # 128/288/2304

        # 2. Aggregate the sequence dimension (dim=1) by averaging
        #    Shape becomes (batch_size, 128) -> (16, 128)
        visual_embeddings = visual_embeddings.mean(dim=1)
    # print("visual_embeddings.shape:",visual_embeddings.shape)
    
    # --- 3. Calculate Cosine Similarity Loss ---
    # We want to MAXIMIZE the similarity, so we MINIMIZE (1 - similarity).
    # F.cosine_similarity returns a value between -1 (opposite) and 1 (identical).
    # The loss will therefore be between 0 (perfect match) and 2 (perfectly opposite).
    
    # # Calculate similarity for each item in the batch
    # similarity = F.cosine_similarity(visual_embeddings, voxel_select, dim=1)
    # torch.cuda.empty_cache()
    
    # # Calculate the loss as the average of (1 - similarity) over the batch
    # loss = (1 - similarity).mean()
    visual_embeddings_norm = nn.functional.normalize(visual_embeddings.flatten(1), dim=-1, eps=1e-6)
    voxel_select_norm = nn.functional.normalize(voxel_select.flatten(1), dim=-1, eps=1e-6)
    if epoch < int(opt.mixup_pct * opt.n_epochs):
        loss = mixco_nce(visual_embeddings_norm, voxel_select_norm, temp=.006, perm=perm, betas=betas, select=select)
    else:
        # ----------- calculate mse loss -----------
        loss = F.mse_loss(visual_embeddings, voxel_select)
        torch.cuda.empty_cache()

    similarity = F.cosine_similarity(visual_embeddings, voxel_select, dim=1)
    torch.cuda.empty_cache()
    # ------------------------------------------

    # --- 4. Simplified Return Value ---
    # The old RSA-related return values are no longer needed.
    # We return the loss and can also return the average similarity as a metric.
    avg_similarity_metric = similarity.mean().item()

    return loss, avg_similarity_metric

def run_neural_model_v3(opt,neural_visual_inputs, RSA_inputs, model,epoch,target,print_gamma=False):
    remove = 0
    visual_evc,visual_tos,visual_ppa,visual_rsc = neural_visual_inputs
    RSA_target_evc,RSA_target_tos,RSA_target_ppa,RSA_target_rsc = RSA_inputs
    # target_dir = {'evc':target[0],'tos':target[1],'ppa':target[2],'rsc':target[3]}
    # unique_labels = torch.unique(target, sorted=True)
    # unique_labels = {'evc':torch.unique(target[0], sorted=True),
    #                  'tos':torch.unique(target[1], sorted=True),
    #                  'ppa':torch.unique(target[2], sorted=True),
    #                  'rsc':torch.unique(target[3], sorted=True),}
    samples_num = {
        'evc':visual_evc.size(0),
        'tos':visual_tos.size(0),
        'ppa':visual_ppa.size(0),
        'rsc':visual_rsc.size(0)
    }
    # print(unique_labels)
    # num_classes = unique_labels.shape[0]
    # num_classes = {'evc':unique_labels['evc'].shape[0],
    #                'tos':unique_labels['tos'].shape[0],
    #                'ppa':unique_labels['ppa'].shape[0],
    #                'rsc':unique_labels['rsc'].shape[0],}
    del target

    # def compute_label_rdm_from_features(v2, roi):
    #     """Average CNN features per label, then compute RDM. v2: [batch, features]"""
    #     feat_dim = v2.shape[1]
    #     label_means = torch.zeros(num_classes[roi], feat_dim, device=v2.device)
    #     for idx, lbl in enumerate(unique_labels[roi]):
    #         mask = (target_dir[roi] == lbl)
    #         label_means[idx] = v2[mask].mean(dim=0)
    #     label_means = label_means - torch.mean(label_means, dim=0)  # center over classes
    #     label_means = label_means + torch.randn_like(label_means) * 1e-5 # Prevent zero-norm collapse while preserving gradient flow

    #     norms = label_means.norm(dim=1)
    #     if (norms < 1e-6).all():
    #         return torch.zeros(num_classes[roi], num_classes[roi], device=v2.device)

    #     if opt.rho4rdm:
    #         label_ranks = torch.argsort(torch.argsort(label_means, dim=1), dim=1).float()
    #         label_ranks = label_ranks + torch.randn_like(label_ranks) * 1e-8
    #         rdm = torch.corrcoef(label_ranks)
    #         rdm = torch.nan_to_num(rdm, nan=0.0)
    #     else:
    #         rdm = torch.nn.functional.cosine_similarity(
    #             label_means.unsqueeze(1), label_means.unsqueeze(0), dim=-1, eps=1e-8
    #         )
    #         rdm = torch.nan_to_num(rdm, nan=0.0)
    #     return rdm
    out_evc = get_intermediate_outputs_v0(model, visual_evc, opt)
    out_tos = get_intermediate_outputs_v0(model, visual_tos, opt)
    out_ppa = get_intermediate_outputs_v0(model, visual_ppa, opt)
    out_rsc = get_intermediate_outputs_v0(model, visual_rsc, opt)

    out = {'evc': out_evc, 'tos': out_tos, 'ppa': out_ppa, 'rsc': out_rsc}
    del out_evc, out_tos, out_ppa, out_rsc
    # check visual_evc.size(1) equal to visual_tos.size(1)
    if visual_evc.size(1) != visual_tos.size(1) or visual_evc.size(1) != visual_ppa.size(1) or visual_evc.size(1) != visual_rsc.size(1):
        sys.exit(0)
    if visual_evc.size(0) != visual_tos.size(0) or visual_evc.size(0) != visual_ppa.size(0) or visual_evc.size(0) != visual_rsc.size(0):
        sys.exit(0)

    S_k = {}
    for roi in ['evc', 'tos', 'ppa', 'rsc']:
        if roi not in S_k:
            S_k[roi] = []
        for k, v in out[roi].items():
            if k != 'temporal_attention':
                if opt.network_choose == 'vit_3d':
                    v = v.reshape(visual_evc.size(1),visual_evc.size(0),-1).contiguous()
                else:
                    v = v.view(visual_evc.size(1), visual_evc.size(0), -1).contiguous()
                v = torch.mean(v, dim=0)               # [batch, features]

            v1 = v.detach()
            v2 = v1.cuda(0)
            del v, v1
            torch.cuda.empty_cache()

            # make rdm
            if opt.rho4rdm:
                # v2_numpy = v2.cpu().numpy()
                # ranks_numpy = rankdata(v2_numpy, method='average', axis=1)
                # del v2_numpy
                # v2_ranks = torch.tensor(ranks_numpy, dtype=torch.float32, device=v2.device)
                # del ranks_numpy
                # b = torch.corrcoef(v2_ranks)
                # del v2_ranks
                v2_ranks = torch.argsort(torch.argsort(v2, dim=1), dim=1).float()
                b = torch.corrcoef(v2_ranks)
                del v2_ranks
            else:
                b = torch.nn.functional.cosine_similarity(v2.unsqueeze(1), v2.unsqueeze(0), dim=-1)
            torch.cuda.empty_cache()
            b1 = b.cuda(0)

            # b1 = compute_label_rdm_from_features(v2, roi)   # [num_classes, num_classes]
            del v2
            torch.cuda.empty_cache()
            S_k[roi].append(b1)

        for i, s in enumerate(S_k[roi]):
            if torch.isnan(s).any():
                print(f"NaN in S_k[{roi}][{i}]")

    if torch.isnan(model.gamma).any():
        print("Gamma corrupted, reinitializing...")
        with torch.no_grad():
            # initialize gamma
            model.gamma.fill_(0.0)

    n_layers = len(S_k['evc'])
    # =========== initialize gamma v2 ===========
    
    # # ── compute per-layer correlations (returned for epoch-1 averaging) ────
    # def _upper_tri(mat):
    #     n = mat.shape[0]
    #     idx = torch.triu_indices(n, n, offset=1, device=mat.device)
    #     return mat[idx[0], idx[1]]

    # def _pearson_r(x, y):
    #     x = x - x.mean();  y = y - y.mean()
    #     return (x * y).sum() / (x.norm() * y.norm()).clamp(min=1e-8)

    # init_corrs = {}
    # if epoch == 1:
    #     roi_targets = {
    #         'evc': RSA_target_evc, 'tos': RSA_target_tos,
    #         'ppa': RSA_target_ppa, 'rsc': RSA_target_rsc,
    #     }
        
    #     with torch.no_grad():
    #         for roi in ['evc', 'tos', 'ppa', 'rsc']:
    #             target_flat = _upper_tri(roi_targets[roi])
    #             init_corrs[roi] = torch.stack([
    #                 _pearson_r(_upper_tri(S_k[roi][i]), target_flat)
    #                 for i in range(n_layers)
    #             ]).cpu()                                    # (n_layers,) on CPU
    # ======================================
    
    gamma = F.softmax(model.gamma, dim=0)
    # if assigned gamma v1/v2
    # gamma_grouped = model.gamma.reshape(4, n_layers)          # (4, 3)
    # gamma    = F.softmax(gamma_grouped, dim=1)          # softmax within each row
    # gamma = gamma.reshape(-1)
    if print_gamma:
        print(gamma)
    
    S_cnn_evc = sum(gamma[i]                * S_k['evc'][i] for i in range(n_layers))
    S_cnn_tos = sum(gamma[i + n_layers]     * S_k['tos'][i] for i in range(n_layers))
    S_cnn_ppa = sum(gamma[i + 2 * n_layers] * S_k['ppa'][i] for i in range(n_layers))
    S_cnn_rsc = sum(gamma[i + 3 * n_layers] * S_k['rsc'][i] for i in range(n_layers))

    S_cnn_evc = torch.nan_to_num(S_cnn_evc, nan=0.0)
    S_cnn_tos = torch.nan_to_num(S_cnn_tos, nan=0.0)
    S_cnn_ppa = torch.nan_to_num(S_cnn_ppa, nan=0.0)
    S_cnn_rsc = torch.nan_to_num(S_cnn_rsc, nan=0.0)

    # num = {'evc':(num_classes['evc'] * (num_classes['evc'] - 1)) / 2,
    #        'tos':(num_classes['tos'] * (num_classes['tos'] - 1)) / 2,
    #        'ppa':(num_classes['ppa'] * (num_classes['ppa'] - 1)) / 2,
    #        'rsc':(num_classes['rsc'] * (num_classes['rsc'] - 1)) / 2,}
    
    num = {'evc':(visual_evc.size(0) * (visual_evc.size(0) - 1)) / 2,
           'tos':(visual_tos.size(0) * (visual_tos.size(0) - 1)) / 2,
           'ppa':(visual_ppa.size(0) * (visual_ppa.size(0) - 1)) / 2,
           'rsc':(visual_rsc.size(0) * (visual_rsc.size(0) - 1)) / 2,}

    def rdm_loss(S_cnn, RSA_target, roi):
        remove = 0
        # print(f"S_cnn_{roi} range: [{S_cnn.min():.4f}, {S_cnn.max():.4f}]")
        EPS = 1e-6
        # diff = torch.atanh(S_cnn - torch.diag_embed(torch.diag(S_cnn))) \
        #         - torch.atanh(RSA_target - torch.diag_embed(torch.diag(RSA_target)))
        safe_cnn = (S_cnn - torch.diag_embed(torch.diag(S_cnn))).clamp(-1 + EPS, 1 - EPS)
        safe_target = (RSA_target - torch.diag_embed(torch.diag(RSA_target))).clamp(-1 + EPS, 1 - EPS)
        diff = torch.atanh(safe_cnn) - torch.atanh(safe_target)
        if len(torch.where(torch.isinf(diff))[0])>0:
            print('inf appears.')
            remove = len(torch.where(torch.isinf(diff))[0])
            print('remove=',remove)
            print('position',torch.where(torch.isinf(diff)))
            cnn_f = S_cnn-torch.diag_embed(torch.diag(S_cnn))
            rsa = RSA_target-torch.diag_embed(torch.diag(RSA_target))
            print(cnn_f[torch.where(torch.isinf(diff))])
            print(rsa[torch.where(torch.isinf(diff))])
            diff = torch.where(torch.isinf(diff), torch.full_like(diff, 0), diff)

        if len(torch.where(torch.isnan(diff))[0])>0:
            print('nan appears.')
            remove = len(torch.where(torch.isnan(diff))[0])
            print('remove=',remove)
            print('position',torch.where(torch.isnan(diff)))
            cnn_f = S_cnn-torch.diag_embed(torch.diag(S_cnn))
            rsa = RSA_target-torch.diag_embed(torch.diag(RSA_target))
            print(cnn_f[torch.where(torch.isnan(diff))])
            print(rsa[torch.where(torch.isnan(diff))])
            diff = torch.where(torch.isnan(diff), torch.full_like(diff, 0), diff)
        denom = max(num[roi] - remove // 2, 1)  # prevent zero/negative
        return torch.pow(diff, 2).sum() / 2 / denom

    loss = (rdm_loss(S_cnn_evc, RSA_target_evc, 'evc') \
            + rdm_loss(S_cnn_tos, RSA_target_tos, 'tos') \
            + rdm_loss(S_cnn_ppa, RSA_target_ppa, 'ppa') \
            + rdm_loss(S_cnn_rsc, RSA_target_rsc, 'rsc')) / 4

    # print('S_cnn_evc size:', S_cnn_evc.size(), "S_k['evc'][0] size:", S_k['evc'][0].size())
    # print('S_cnn_tos size:', S_cnn_tos.size(), "S_k['tos'][0] size:", S_k['tos'][0].size())
    # sys.exit(0)

    return gamma, loss, [S_cnn_evc, S_cnn_tos, S_cnn_ppa, S_cnn_rsc, S_k], samples_num
def run_neural_model_v3_add_pfc(opt,neural_visual_inputs, RSA_inputs, model,epoch,target,print_gamma=False):
    remove = 0
    visual_evc,visual_tos,visual_ppa,visual_rsc, visual_pfc = neural_visual_inputs
    RSA_target_evc,RSA_target_tos,RSA_target_ppa,RSA_target_rsc, RSA_target_pfc = RSA_inputs
    samples_num = {
        'evc':visual_evc.size(0),
        'tos':visual_tos.size(0),
        'ppa':visual_ppa.size(0),
        'rsc':visual_rsc.size(0),
        'pfc':visual_pfc.size(0),
    }
    del target

    out_evc = get_intermediate_outputs_v0(model, visual_evc, opt)
    out_tos = get_intermediate_outputs_v0(model, visual_tos, opt)
    out_ppa = get_intermediate_outputs_v0(model, visual_ppa, opt)
    out_rsc = get_intermediate_outputs_v0(model, visual_rsc, opt)
    out_pfc = get_intermediate_outputs_v0(model, visual_pfc, opt)

    out = {'evc': out_evc, 'tos': out_tos, 'ppa': out_ppa, 'rsc': out_rsc, 'pfc': out_pfc}
    del out_evc, out_tos, out_ppa, out_rsc, out_pfc
    # check visual_evc.size(1) equal to visual_tos.size(1)
    if visual_evc.size(1) != visual_tos.size(1) or visual_evc.size(1) != visual_ppa.size(1) or visual_evc.size(1) != visual_rsc.size(1) or visual_evc.size(1) != visual_pfc.size(1):
        sys.exit(0)
    if visual_evc.size(0) != visual_tos.size(0) or visual_evc.size(0) != visual_ppa.size(0) or visual_evc.size(0) != visual_rsc.size(0) or visual_evc.size(0) != visual_pfc.size(0):
        sys.exit(0)

    S_k = {}
    for roi in ['evc', 'tos', 'ppa', 'rsc', 'pfc']:
        if roi not in S_k:
            S_k[roi] = []
        for k, v in out[roi].items():
            if k != 'temporal_attention':
                if opt.network_choose == 'vit_3d':
                    v = v.reshape(visual_evc.size(1),visual_evc.size(0),-1).contiguous()
                else:
                    v = v.view(visual_evc.size(1), visual_evc.size(0), -1).contiguous()
                v = torch.mean(v, dim=0)               # [batch, features]

            v1 = v.detach()
            v2 = v1.cuda(0)
            del v, v1
            torch.cuda.empty_cache()

            # make rdm
            if opt.rho4rdm:
                # v2_numpy = v2.cpu().numpy()
                # ranks_numpy = rankdata(v2_numpy, method='average', axis=1)
                # del v2_numpy
                # v2_ranks = torch.tensor(ranks_numpy, dtype=torch.float32, device=v2.device)
                # del ranks_numpy
                # b = torch.corrcoef(v2_ranks)
                # del v2_ranks
                v2_ranks = torch.argsort(torch.argsort(v2, dim=1), dim=1).float()
                b = torch.corrcoef(v2_ranks)
                del v2_ranks
            else:
                b = torch.nn.functional.cosine_similarity(v2.unsqueeze(1), v2.unsqueeze(0), dim=-1)
            torch.cuda.empty_cache()
            b1 = b.cuda(0)

            # b1 = compute_label_rdm_from_features(v2, roi)   # [num_classes, num_classes]
            del v2
            torch.cuda.empty_cache()
            S_k[roi].append(b1)

        for i, s in enumerate(S_k[roi]):
            if torch.isnan(s).any():
                print(f"NaN in S_k[{roi}][{i}]")

    if torch.isnan(model.gamma).any():
        print("Gamma corrupted, reinitializing...")
        with torch.no_grad():
            # initialize gamma
            model.gamma.fill_(0.0)

    n_layers = len(S_k['evc'])
    
    gamma = F.softmax(model.gamma, dim=0)
    # if assigned gamma v1/v2
    # gamma_grouped = model.gamma.reshape(4, n_layers)          # (4, 3)
    # gamma    = F.softmax(gamma_grouped, dim=1)          # softmax within each row
    # gamma = gamma.reshape(-1)
    if print_gamma:
        print(gamma)
    
    S_cnn_evc = sum(gamma[i]                * S_k['evc'][i] for i in range(n_layers))
    S_cnn_tos = sum(gamma[i + n_layers]     * S_k['tos'][i] for i in range(n_layers))
    S_cnn_ppa = sum(gamma[i + 2 * n_layers] * S_k['ppa'][i] for i in range(n_layers))
    S_cnn_rsc = sum(gamma[i + 3 * n_layers] * S_k['rsc'][i] for i in range(n_layers))
    S_cnn_pfc = sum(gamma[i + 4 * n_layers] * S_k['pfc'][i] for i in range(n_layers))

    S_cnn_evc = torch.nan_to_num(S_cnn_evc, nan=0.0)
    S_cnn_tos = torch.nan_to_num(S_cnn_tos, nan=0.0)
    S_cnn_ppa = torch.nan_to_num(S_cnn_ppa, nan=0.0)
    S_cnn_rsc = torch.nan_to_num(S_cnn_rsc, nan=0.0)
    S_cnn_pfc = torch.nan_to_num(S_cnn_pfc, nan=0.0)

    
    num = {'evc':(visual_evc.size(0) * (visual_evc.size(0) - 1)) / 2,
           'tos':(visual_tos.size(0) * (visual_tos.size(0) - 1)) / 2,
           'ppa':(visual_ppa.size(0) * (visual_ppa.size(0) - 1)) / 2,
           'rsc':(visual_rsc.size(0) * (visual_rsc.size(0) - 1)) / 2,
           'pfc':(visual_pfc.size(0) * (visual_pfc.size(0) - 1)) / 2}

    def rdm_loss(S_cnn, RSA_target, roi):
        remove = 0
        # print(f"S_cnn_{roi} range: [{S_cnn.min():.4f}, {S_cnn.max():.4f}]")
        EPS = 1e-6
        # diff = torch.atanh(S_cnn - torch.diag_embed(torch.diag(S_cnn))) \
        #         - torch.atanh(RSA_target - torch.diag_embed(torch.diag(RSA_target)))
        safe_cnn = (S_cnn - torch.diag_embed(torch.diag(S_cnn))).clamp(-1 + EPS, 1 - EPS)
        safe_target = (RSA_target - torch.diag_embed(torch.diag(RSA_target))).clamp(-1 + EPS, 1 - EPS)
        diff = torch.atanh(safe_cnn) - torch.atanh(safe_target)
        if len(torch.where(torch.isinf(diff))[0])>0:
            print('inf appears.')
            remove = len(torch.where(torch.isinf(diff))[0])
            print('remove=',remove)
            print('position',torch.where(torch.isinf(diff)))
            cnn_f = S_cnn-torch.diag_embed(torch.diag(S_cnn))
            rsa = RSA_target-torch.diag_embed(torch.diag(RSA_target))
            print(cnn_f[torch.where(torch.isinf(diff))])
            print(rsa[torch.where(torch.isinf(diff))])
            diff = torch.where(torch.isinf(diff), torch.full_like(diff, 0), diff)

        if len(torch.where(torch.isnan(diff))[0])>0:
            print('nan appears.')
            remove = len(torch.where(torch.isnan(diff))[0])
            print('remove=',remove)
            print('position',torch.where(torch.isnan(diff)))
            cnn_f = S_cnn-torch.diag_embed(torch.diag(S_cnn))
            rsa = RSA_target-torch.diag_embed(torch.diag(RSA_target))
            print(cnn_f[torch.where(torch.isnan(diff))])
            print(rsa[torch.where(torch.isnan(diff))])
            diff = torch.where(torch.isnan(diff), torch.full_like(diff, 0), diff)
        denom = max(num[roi] - remove // 2, 1)  # prevent zero/negative
        return torch.pow(diff, 2).sum() / 2 / denom

    loss = (rdm_loss(S_cnn_evc, RSA_target_evc, 'evc') \
            + rdm_loss(S_cnn_tos, RSA_target_tos, 'tos') \
            + rdm_loss(S_cnn_ppa, RSA_target_ppa, 'ppa') \
            + rdm_loss(S_cnn_rsc, RSA_target_rsc, 'rsc') \
            + rdm_loss(S_cnn_pfc, RSA_target_pfc, 'pfc')) / 5

    # print('S_cnn_evc size:', S_cnn_evc.size(), "S_k['evc'][0] size:", S_k['evc'][0].size())
    # print('S_cnn_tos size:', S_cnn_tos.size(), "S_k['tos'][0] size:", S_k['tos'][0].size())
    # sys.exit(0)

    return gamma, loss, [S_cnn_evc, S_cnn_tos, S_cnn_ppa, S_cnn_rsc, S_cnn_pfc, S_k], samples_num
        
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from mpl_toolkits.axes_grid1 import make_axes_locatable
def visualize_rdms_v2(opt, S_cnn, S_k, samples_num, RSA_target, neural_data_item, epoch, save_dir="rdm_plots"):
    S_cnn = S_cnn.cpu().detach().numpy()
    RSA_target = RSA_target.cpu().detach().numpy()
    S_k = [s.cpu().detach().numpy() for s in S_k]

    visual, neural_response, visualization_item, target = neural_data_item
    target_list = target.tolist()
    # visualization_item list
    # print("visualization_item:",visualization_item)

    all_rdms = S_k + [S_cnn, RSA_target]
    n_layers = len(S_k)
    all_titles = (
        [f"Layer {i+1} RDM" for i in range(n_layers)]
        + ["S_cnn RDM (Model)", "RSA_target RDM (fMRI)"]
    )
    n_plots = len(all_rdms)

    def upper_tri(m):
        return m[np.triu_indices_from(m)]

    S_cnn_upper = upper_tri(S_cnn)
    RSA_target_upper = upper_tri(RSA_target)
    corr_cnn_rsa = np.corrcoef(S_cnn_upper, RSA_target_upper)[0, 1]
    for i in range(n_layers):
        corr_layer_rsa = np.corrcoef(upper_tri(S_k[i]), RSA_target_upper)[0, 1]
        all_titles[i] += f"\nCorr: {corr_layer_rsa:.3f}"
    all_titles[-2] += f"\nCorr: {corr_cnn_rsa:.3f}"

    cmap = "viridis"
    cb_label = "Rank Correlation" if opt.rho4rdm else "Cosine Similarity"

    fig, axes = plt.subplots(1, n_plots, figsize=(4 * n_plots, 5))
    if n_plots == 1:
        axes = [axes]

    for ax, rdm, title in zip(axes, all_rdms, all_titles):
        # Per-RDM color scale
        vmin, vmax = rdm.min(), rdm.max()
        im = ax.imshow(rdm, cmap=cmap, vmin=vmin, vmax=vmax)
        ax.set_title(title)
        ax.set_xlabel("Input Item")
        ax.set_xticks(range(samples_num))
        ax.set_xticklabels(list(range(samples_num)), rotation=45)
        ax.set_ylabel("Input Item")
        ax.set_yticks(range(samples_num))
        ax.set_yticklabels(list(range(samples_num)), rotation=45)

        # Attach a colorbar below this specific subplot
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("bottom", size="5%", pad=0.6)
        cbar = fig.colorbar(im, cax=cax, orientation="horizontal")
        cbar.set_label(cb_label, fontsize=8)
        cbar.ax.tick_params(labelsize=7)
        cbar.locator = MaxNLocator(nbins=4)   # ← max 4 ticks per bar
        cbar.update_ticks()

    fig.suptitle(
        f"RDM Comparison at Epoch {epoch}\n",
        fontsize=14
    )
    plt.tight_layout(rect=[0, 0, 1, 0.93])

    save_path = os.path.join(opt.result_path, save_dir)
    os.makedirs(save_path, exist_ok=True)
    plt.savefig(os.path.join(save_path, f"rdm_epoch_{epoch:04d}.png"), bbox_inches="tight")
    plt.close(fig)

# def visualize_rdms_v2(opt, S_cnn, S_k, samples_num, RSA_target, neural_data_item, epoch, save_dir="rdm_plots"):
#     S_cnn = S_cnn.cpu().detach().numpy()
#     RSA_target = RSA_target.cpu().detach().numpy()
#     S_k = [s.cpu().detach().numpy() for s in S_k]

#     # Build GT RDM
#     # GT_rdm = np.zeros((len(neural_data_item[3]), len(neural_data_item[3])))
#     # for i in range(len(neural_data_item[3])):
#     #     for j in range(len(neural_data_item[3])):
#     #         if neural_data_item[3][i] == neural_data_item[3][j]:
#     #             GT_rdm[i][j] = 1

#     visual, neural_response, visualization_item, target = neural_data_item
#     target_list = target.tolist()

#     # All RDMs to plot: layer RDMs + S_cnn + RSA_target
#     all_rdms = S_k + [S_cnn, RSA_target]
#     n_layers = len(S_k)
#     all_titles = (
#         [f"Layer {i+1} RDM" for i in range(n_layers)]
#         + ["S_cnn RDM (Model)", "RSA_target RDM (fMRI)"]
#         # + ["S_cnn RDM (Model)", "RSA_target RDM (fMRI)", "GT RDM"]
#     )
#     n_plots = len(all_rdms)

#     # Compute correlations for suptitle
#     def upper_tri(m):
#         return m[np.triu_indices_from(m)]

#     S_cnn_upper = upper_tri(S_cnn)
#     RSA_target_upper = upper_tri(RSA_target)
#     # GT_upper = upper_tri(GT_rdm)
#     corr_cnn_rsa = np.corrcoef(S_cnn_upper, RSA_target_upper)[0, 1]
#     for i in range(n_layers):
#         corr_layer_rsa = np.corrcoef(upper_tri(S_k[i]), RSA_target_upper)[0, 1]
#         all_titles[i] += f"\nCorr: {corr_layer_rsa:.3f}"
#     all_titles[-2] += f"\nCorr: {corr_cnn_rsa:.3f}"
#     # corr_cnn_gt  = np.corrcoef(S_cnn_upper, GT_upper)[0, 1]
#     # corr_rsa_gt  = np.corrcoef(RSA_target_upper, GT_upper)[0, 1]

#     # Global vmin/vmax for shared colorbar
#     vmin = min(m.min() for m in all_rdms)
#     vmax = max(m.max() for m in all_rdms)
#     cmap = "viridis"
#     cb_label = "Rank Correlation" if opt.rho4rdm else "Cosine Similarity"

#     fig, axes = plt.subplots(1, n_plots, figsize=(4 * n_plots, 5))
#     if n_plots == 1:
#         axes = [axes]

#     for ax, rdm, title in zip(axes, all_rdms, all_titles):
#         im = ax.imshow(rdm, cmap=cmap)
#         ax.set_title(title)
#         ax.set_xlabel("Input Item")
#         ax.set_xticks(range(samples_num))
#         ax.set_xticklabels(list(range(samples_num)), rotation=45)
#         ax.set_ylabel("Input Item")
#         ax.set_yticks(range(samples_num))
#         ax.set_yticklabels(list(range(samples_num)), rotation=45)

#     # Single shared colorbar on the right
#     # fig.colorbar(im, ax=axes, label=cb_label, orientation="vertical", fraction=0.02, pad=0.04)
#     cbar_ax = fig.add_axes([0.92, 0.15, 0.015, 0.7])  # [left, bottom, width, height]
#     fig.colorbar(im, cax=cbar_ax, label=cb_label, orientation="vertical")
#     plt.tight_layout(rect=[0, 0, 0.91, 0.93])  # leave room on the right for the colorbar


#     fig.suptitle(
#         f"RDM Comparison at Epoch {epoch}\n",
#         # f"S_cnn vs RSA_target: {corr_cnn_rsa:.3f}  |  ",
#         # f"S_cnn vs GT: {corr_cnn_gt:.3f}  |  "
#         # f"RSA_target vs GT: {corr_rsa_gt:.3f}",
#         fontsize=14
#     )
#     # plt.tight_layout(rect=[0, 0, 1, 0.93])

#     save_path = os.path.join(opt.result_path, save_dir)
#     os.makedirs(save_path, exist_ok=True)
#     plt.savefig(os.path.join(save_path, f"rdm_epoch_{epoch:04d}.png"))
#     plt.close(fig)

def visualize_rdms(opt, S_cnn, RSA_target, neural_data_item, epoch, save_dir="rdm_plots"):
    """
    Visualizes the S_cnn and RSA_target RDMs and saves the figure.

    Args:
        S_cnn (np.array): The model's similarity matrix.
        RSA_target (np.array): The target similarity matrix.
        epoch (int): The current epoch number, used for the filename.
        save_dir (str): Directory to save the plots in.
    """
    # get upper triangular elements of S_cnn and RSA_target
    S_cnn = S_cnn.cpu().detach().numpy()
    RSA_target = RSA_target.cpu().detach().numpy()
    # get GT_rdm from target, same label to 1, different to 0
    GT_rdm = np.zeros((len(neural_data_item[3]),len(neural_data_item[3])))
    for i in range(len(neural_data_item[3])):
        for j in range(len(neural_data_item[3])):
            if neural_data_item[3][i] == neural_data_item[3][j]:
                GT_rdm[i][j] = 1

    S_cnn_upper = S_cnn[np.triu_indices_from(S_cnn)]
    RSA_target_upper = RSA_target[np.triu_indices_from(RSA_target)]
    GT_rdm_upper = GT_rdm[np.triu_indices_from(GT_rdm)]

    fig, axes = plt.subplots(1, 3, figsize=(12, 6))
    fig.suptitle(f'RDM Comparison at Epoch {epoch}\nS_cnn and RSA_target Correlation {np.corrcoef(S_cnn_upper.flatten(),RSA_target_upper.flatten())[0,1]:.3f}\nS_cnn and GT Correlation {np.corrcoef(S_cnn_upper.flatten(),GT_rdm_upper.flatten())[0,1]:.3f}\nRSA target and GT Correlation {np.corrcoef(RSA_target_upper.flatten(),GT_rdm_upper.flatten())[0,1]:.3f}', fontsize=16)
    
    visual, neural_response,  visualization_item, target = neural_data_item

    # Plot S_cnn RDM
    im1 = axes[0].imshow(S_cnn, cmap='viridis')#, vmin=-1, vmax=1)
    axes[0].set_title('S_cnn RDM (Model)')
    axes[0].set_xlabel('Input Item Label')
    # set x labels as neural_data_item['label']
    axes[0].set_xticks(range(len(target)))
    axes[0].set_xticklabels(target.tolist(), rotation=45)
    axes[0].set_ylabel('Input Item Label')
    axes[0].set_yticks(range(len(target)))
    axes[0].set_yticklabels(target.tolist(), rotation=45)
    if opt.rho4rdm:
        fig.colorbar(im1, ax=axes[0], label='Rank Correlation', orientation='horizontal')
    else:
        fig.colorbar(im1, ax=axes[0], label='Cosine Similarity', orientation='horizontal')

    # Plot RSA_target RDM
    im2 = axes[1].imshow(RSA_target, cmap='viridis')#, vmin=-1, vmax=1)
    axes[1].set_title('RSA_target RDM (Target)')
    axes[1].set_xlabel('Input Item Label')
    # set x labels as neural_data_item['label']
    axes[1].set_xticks(range(len(target)))
    axes[1].set_xticklabels(target.tolist(), rotation=45)
    axes[1].set_ylabel('Input Item Label')
    axes[1].set_yticks(range(len(target)))
    axes[1].set_yticklabels(target.tolist(), rotation=45)
    if opt.rho4rdm:
        fig.colorbar(im2, ax=axes[1], label='Rank Correlation', orientation='horizontal')
    else:
        fig.colorbar(im2, ax=axes[1], label='Cosine Similarity', orientation='horizontal')

    # plot GT_rdm
    im3 = axes[2].imshow(GT_rdm, cmap='viridis')#, vmin=-1, vmax=1)
    axes[2].set_title('GT RDM')
    axes[2].set_xlabel('Input Item Label')
    # set x labels as neural_data_item['label']
    axes[2].set_xticks(range(len(target)))
    axes[2].set_xticklabels(target.tolist(), rotation=45)
    axes[2].set_ylabel('Input Item Label')
    axes[2].set_yticks(range(len(target)))
    axes[2].set_yticklabels(target.tolist(), rotation=45)
    if opt.rho4rdm:
        fig.colorbar(im3, ax=axes[2], label='Rank Correlation', orientation='horizontal')
    else:
        fig.colorbar(im3, ax=axes[2], label='Cosine Similarity', orientation='horizontal')

    plt.tight_layout(rect=[0, 0, 1, 0.96]) # Adjust layout to make room for suptitle
    
    # Save the figure with the epoch number in the filename
    save_path = os.path.join(opt.result_path, save_dir)
    # Create the save directory if it doesn't exist
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    plt.savefig(os.path.join(save_path, f"rdm_epoch_{epoch:04d}.png"))
    # print(f"✅ Saved RDM visualization to {save_path}")
    plt.close(fig) # Close the figure to free up memory


def calculate_accuracy(outputs, targets):
    batch_size = targets.size(0)
    values, indices = outputs.topk(k=1, dim=1, largest=True)
    pred = indices
    pred = pred.t()
    correct = pred.eq(targets.view(1, -1))
    n_correct_elements = correct.float()
    n_correct_elements = n_correct_elements.sum()
    n_correct_elements = n_correct_elements.item()
    return n_correct_elements / batch_size

def calculate_accuracy_annot_reg(outputs, targets):
    # from sklearn.metrics import r2_score
    from scipy.stats import pearsonr

    outputs_np = outputs.detach().cpu().numpy().flatten()
    targets_np = targets.detach().cpu().numpy().flatten()

    # # calculate r2
    # r2 = r2_score(targets_np, outputs_np)
    # return r2
    # Calculate correlation for each class, then average
    # Calculate correlation for each class, skipping constant arrays
    if np.std(targets_np) == 0:
        # Skip constant arrays or assign a default value
        print(f"constant arrays occurred, Skip this batch")
        return np.nan
    correlation, _ = pearsonr(targets_np, outputs_np)
    return correlation

from sklearn.metrics import roc_auc_score, roc_curve
def calculate_accuracy_annot(outputs, targets):
    batch_size = targets.size(0)
    # make >0.5 = 1 in targets
    # outputs = torch.sigmoid(outputs)
    # --- Original Accuracy (threshold = 0.5) ---
    preds = torch.where(outputs > 0.5, 1, 0)
    n_correct_elements = (targets == preds).sum().item()
    accuracy = n_correct_elements / (batch_size * 15) # batchsize* number of classes

    # --- Flatten for sklearn ---
    outputs_np = outputs.detach().cpu().numpy().flatten()
    targets_np = targets.detach().cpu().numpy().flatten().astype(int)

    # --- AUC ---
    auc = roc_auc_score(targets_np, outputs_np)

    # --- Youden Index & Best Threshold ---
    fpr, tpr, thresholds = roc_curve(targets_np, outputs_np)
    
    youden_scores = tpr - fpr                        # J = Sensitivity + Specificity - 1
    best_idx = np.argmax(youden_scores)
    youden_index = youden_scores[best_idx]
    best_threshold = thresholds[best_idx]

    # --- Accuracy at Best Threshold ---
    best_preds = (outputs_np >= best_threshold).astype(int)
    best_threshold_acc = (best_preds == targets_np).sum() / len(targets_np)

    return accuracy, auc, youden_index, best_threshold_acc

def calculate_accuracy_cross(outputs, targets,num):
    values, indices = outputs.topk(k=1, dim=1, largest=True)
    pred = indices
    targets = targets.detach().cpu().numpy()
    pred = pred.detach().cpu().numpy()
    correct_pred = np.zeros(num)
    total = np.zeros(num)
    for label,prediction in zip(targets,pred):
        if label == prediction:
            correct_pred[label] +=1
        total[label] +=1
    ac = correct_pred/total
    return ac

# def calculate_accuracy_cross_annot(outputs, targets,num):
#     # outputs = torch.sigmoid(outputs)
#     outputs = torch.where(outputs > 0.5, 1, 0)
#     targets = targets.detach().cpu().numpy()
#     outputs = outputs.detach().cpu().numpy()
#     # outputs and targets: (sample_size, 18)
#     # compare column-wise (per class)
#     correct_per_class = (outputs == targets).sum(axis=0)  # shape (18,)
#     total = targets.shape[0]  # batch size

#     ac = correct_per_class / total  # shape (18,)
#     return ac
def calculate_accuracy_cross_annot(outputs, targets, num):
    # --- Flatten to numpy before thresholding for sklearn ---
    outputs_prob = outputs.detach().cpu().numpy()       # keep raw probs for AUC/Youden
    targets_np   = targets.detach().cpu().numpy().astype(int)

    # --- Original Accuracy (threshold = 0.5) ---
    preds = (outputs_prob >= 0.5).astype(int)
    correct_per_class = (preds == targets_np).sum(axis=0)  # shape (15,)
    total = targets_np.shape[0]
    ac = correct_per_class / total                          # shape (15,)

    auc_per_class             = np.zeros(num)
    youden_per_class          = np.zeros(num)
    best_threshold_acc_per_class = np.zeros(num)

    for c in range(num):
        t = targets_np[:, c]
        p = outputs_prob[:, c]

        # Skip degenerate classes (only one label present)
        if len(np.unique(t)) < 2:
            # print unique labels
            # print(np.unique(t))
            auc_per_class[c]              = np.nan
            youden_per_class[c]           = np.nan
            best_threshold_acc_per_class[c] = np.nan
            continue

        # --- AUC ---
        auc_per_class[c] = roc_auc_score(t, p)

        # --- Youden Index & Best Threshold ---
        fpr, tpr, thresholds = roc_curve(t, p)
        youden_scores  = tpr - fpr
        best_idx       = np.argmax(youden_scores)
        youden_per_class[c] = youden_scores[best_idx]

        # --- Accuracy at Best Threshold ---
        best_preds = (p >= thresholds[best_idx]).astype(int)
        best_threshold_acc_per_class[c] = (best_preds == t).sum() / len(t)

    return ac, auc_per_class, youden_per_class, best_threshold_acc_per_class

def calculate_accuracy_cross_annot_reg(outputs, targets, num):
    # Convert to numpy
    outputs = outputs.detach().cpu().numpy()
    targets = targets.detach().cpu().numpy()
    
    # r2_scores = np.zeros(num)

    # for class_idx in range(num):
    #     # Get predictions and targets for this class/annotation
    #     class_outputs = outputs[:, class_idx] if outputs.ndim > 1 else outputs
    #     class_targets = targets[:, class_idx] if targets.ndim > 1 else targets
        
    #     # Calculate R² for this class
    #     ss_res = np.sum((class_targets - class_outputs) ** 2)  # Residuals
    #     ss_tot = np.sum((class_targets - class_targets.mean()) ** 2)  # Total variance
        
    #     # Handle edge case where variance is 0
    #     if ss_tot == 0:
    #         r2_scores[class_idx] = np.nan
    #     else:
    #         r2_scores[class_idx] = 1 - (ss_res / ss_tot)
    
    # return r2_scores
    from scipy.stats import pearsonr
    correlations = []
    
    for class_idx in range(num):
        if np.std(targets[:, class_idx]) == 0 or np.std(outputs[:, class_idx]) == 0:
            # Skip constant arrays or assign a default value
            print(f"constant arrays occurred, Skip this class")
            # continue  # Skip this class
            correlations.append(np.nan)  # Default to 0

        else:
            corr, _ = pearsonr(targets[:, class_idx], outputs[:, class_idx])
            correlations.append(corr)
    
    # return correlations as array or dict
    return np.array(correlations)

def plot_annot_result(outputs, targets, num, epoch, class_name, ac, opt):
    # Convert to numpy
    outputs = outputs.detach().cpu().numpy()
    targets = targets.detach().cpu().numpy()
    
    # Binarize outputs (threshold at 0.5)
    outputs_binary = (outputs > 0.5).astype(int)
    
    # Create subplots
    cols = min(3, num)  # Max 3 columns
    rows = (num + cols - 1) // cols  # Calculate rows needed
    
    fig, axes = plt.subplots(rows, cols, figsize=(5*cols, 4*rows))
    
    # Flatten axes array for easier iteration
    if num == 1:
        axes = [axes]
    else:
        axes = axes.flatten()
    
    # Plot each class
    for class_idx in range(num):
        ax = axes[class_idx]
        
        # Extract per-class outputs and targets
        if outputs_binary.ndim > 1:
            class_outputs = outputs_binary[:, class_idx]
            class_targets = targets[:, class_idx]
        else:
            class_outputs = outputs_binary
            class_targets = targets
        
        # Compute confusion matrix
        from sklearn.metrics import confusion_matrix
        import seaborn as sns
        cm = confusion_matrix(class_targets, class_outputs, labels=[0, 1])
        
        # Create heatmap
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                    xticklabels=['Pred 0', 'Pred 1'],
                    yticklabels=['True 0', 'True 1'],
                    cbar=False, annot_kws={'size': 14, 'weight': 'bold'})
        
        # Labels and title
        ax.set_xlabel('Predicted', fontsize=10)
        ax.set_ylabel('Actual', fontsize=10)
        ax.set_title(f'{class_name[class_idx]}\nAcc={ac[class_idx]:.3f}', 
                    fontsize=12, fontweight='bold')
    
    # Hide extra subplots if num is not a perfect grid
    for idx in range(num, len(axes)):
        axes[idx].axis('off')
    
    plt.tight_layout()
    save_path = os.path.join(opt.result_path, 'fig')
    if os.path.exists(save_path) == False:
        os.makedirs(save_path)
    plt.savefig(os.path.join(save_path, f'valid_threshold0.5_epoch{epoch}.png'))
    plt.close()

# def plot_regression_result(outputs, targets, num, epoch, class_name, ac, opt):
#     # Convert to numpy
#     outputs = outputs.detach().cpu().numpy()
#     targets = targets.detach().cpu().numpy()
    
#     # Create subplots
#     cols = min(3, num)  # Max 3 columns
#     rows = (num + cols - 1) // cols  # Calculate rows needed
    
#     fig, axes = plt.subplots(rows, cols, figsize=(5*cols, 4*rows))
    
#     # Flatten axes array for easier iteration
#     if num == 1:
#         axes = [axes]
#     else:
#         axes = axes.flatten()
    
#     # Plot each class
#     for class_idx in range(num):
#         ax = axes[class_idx]
        
#         # Extract per-class outputs and targets
#         if outputs.ndim > 1:
#             class_outputs = outputs[:, class_idx]
#             class_targets = targets[:, class_idx]
#         else:
#             class_outputs = outputs
#             class_targets = targets
        
#         # Scatter plot
#         ax.scatter(class_targets, class_outputs, alpha=0.6, s=50)
        
#         # Perfect prediction line (y=x)
#         min_val = min(class_targets.min(), class_outputs.min())
#         max_val = max(class_targets.max(), class_outputs.max())
#         ax.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Perfect')
        
#         # Labels and title
#         ax.set_xlabel('Targets', fontsize=10)
#         ax.set_ylabel('Predictions', fontsize=10)
#         ax.set_title(f'{class_name[class_idx]}, r2={ac[class_idx]:.3f}', fontsize=12, fontweight='bold')
#         ax.legend()
#         ax.grid(True, alpha=0.3)
    
#     # Hide extra subplots if num is not a perfect grid
#     for idx in range(num, len(axes)):
#         axes[idx].axis('off')
    
#     plt.tight_layout()
#     save_path = os.path.join(opt.result_path,'fig')
#     if os.path.exists(save_path) == False:
#         os.makedirs(save_path)
#     plt.savefig(os.path.join(save_path,f'valid_reg_epoch{epoch}.png'))
#     plt.close()
def plot_regression_result(outputs, targets, num, epoch, class_name, ac, opt):
    outputs = outputs.detach().cpu().numpy()
    targets = targets.detach().cpu().numpy()

    cols = min(3, num)
    rows = (num + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(5*cols, 4*rows))

    if num == 1:
        axes = [axes]
    else:
        axes = axes.flatten()

    for class_idx in range(num):
        ax = axes[class_idx]

        if outputs.ndim > 1:
            class_outputs = outputs[:, class_idx]
            class_targets = targets[:, class_idx]
        else:
            class_outputs = outputs
            class_targets = targets

        ax.scatter(class_targets, class_outputs, alpha=0.6, s=50)

        # Red line follows the target (x-axis) range only
        x_min, x_max = class_targets.min()-0.025, class_targets.max()+0.025
        ax.plot([x_min, x_max], [x_min, x_max], 'r--', lw=2, label='Perfect')

        # Set axis limits based on target score range
        ax.set_xlim(x_min, x_max)

        # Title only — no per-subplot axis labels
        ax.set_title(f'{class_name[class_idx]}, r={ac[class_idx]:.3f}', fontsize=12, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)

    for idx in range(num, len(axes)):
        axes[idx].axis('off')

    # Shared axis labels outside the entire figure
    fig.supxlabel('Ground Truth', fontsize=12, y=0.01)
    fig.supylabel('Predictions', fontsize=12, x=0.01)

    plt.tight_layout()
    save_path = os.path.join(opt.result_path, 'fig')
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    plt.savefig(os.path.join(save_path, f'valid_reg_epoch{epoch}.png'), bbox_inches='tight')
    plt.close()

def mixco(voxels, beta=0.15, s_thresh=0.5):
    perm = torch.randperm(voxels.shape[0])
    voxels_shuffle = voxels[perm].to(voxels.device,dtype=voxels.dtype)
    betas = torch.distributions.Beta(beta, beta).sample([voxels.shape[0]]).to(voxels.device,dtype=voxels.dtype)
    select = (torch.rand(voxels.shape[0]) <= s_thresh).to(voxels.device)
    betas_shape = [-1] + [1]*(len(voxels.shape)-1)
    voxels[select] = voxels[select] * betas[select].reshape(*betas_shape) + \
        voxels_shuffle[select] * (1 - betas[select]).reshape(*betas_shape)
    betas[~select] = 1
    return voxels, perm, betas, select

def mixco_nce(preds, targs, temp=0.1, perm=None, betas=None, select=None, distributed=False, 
              accelerator=None, local_rank=None, bidirectional=True):
    brain_clip = (preds @ targs.T)/temp
    
    if perm is not None and betas is not None and select is not None:
        probs = torch.diag(betas)
        probs[torch.arange(preds.shape[0]).to(preds.device), perm] = 1 - betas

        loss = -(brain_clip.log_softmax(-1) * probs).sum(-1).mean()
        if bidirectional:
            loss2 = -(brain_clip.T.log_softmax(-1) * probs.T).sum(-1).mean()
            loss = (loss + loss2)/2
        return loss
    else:
        loss =  F.cross_entropy(brain_clip, torch.arange(brain_clip.shape[0]).to(brain_clip.device))
        if bidirectional:
            loss2 = F.cross_entropy(brain_clip.T, torch.arange(brain_clip.shape[0]).to(brain_clip.device))
            loss = (loss + loss2)/2
        return loss
    