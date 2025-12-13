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
            if opt.co_train:
                if opt.use_model:
                    use_model = '1'
                else:
                    use_model = '0'
                co_train = 'co_train'+'_alpha='+str(opt.alpha)+'_use_model='+use_model
            else:
                co_train = 'not_co_train'

            if opt.behavior == False:
                if opt.network_choose == 'shufflenet_v1':
                    if opt.random_choice == True:
                        now = 'new_result/final_2181/sig_test/'+opt.dataset_choose+'/'+opt.network_choose+'_1.5x/'+'/result_' + opt.dataset_choose + '_split=' + str(opt.split) + '_' + co_train + '_lr=' + str(opt.learning_rate)+'_'+str(opt.sig_test_run) + '_' + str(opt.video_num)
                    else:
                        if opt.data_use == 'mean':
                            now = 'new_result/final_2181/sig_test/' + opt.dataset_choose + '/' + opt.network_choose + '_1.5x/' + '/result_' + opt.dataset_choose + '_split=' + str(opt.split) + '_' + co_train + '_lr=' + str(opt.learning_rate) + '_' + str(opt.sig_test_run)
                        else:
                            now = 'new_result/final_2181/sig_test/' + opt.dataset_choose + '/' + opt.network_choose + '_1.5x/' + '/result_' + opt.dataset_choose + '_split=' + str(opt.split) + '_' + co_train + '_lr=' + str(opt.learning_rate) + '_' + str(opt.sig_test_run) + '_' + opt.data_use
                elif opt.network_choose == 'mobilenet_v2':
                    now = 'new_result/final_2181/sig_test/' + opt.dataset_choose + '/' + opt.network_choose + '_0.45x/' + '/result_' + opt.dataset_choose + '_split=' + str(opt.split) + '_' + co_train + '_lr=' + str(opt.learning_rate) + '_' + str(opt.sig_test_run) + '_' + opt.data_use

                else:
                    if opt.random_choice == True:
                        now = 'new_result/final_2181/sig_test/' + opt.dataset_choose + '/' + opt.network_choose  + '/result_' + opt.dataset_choose + '_split=' + str(opt.split) + '_' + co_train + '_lr=' + str(opt.learning_rate) + '_' + str(opt.sig_test_run) + '_' + str(opt.video_num)
                    else:
                        if opt.data_use == 'mean':
                            now = 'new_result/final_2181/sig_test/' + opt.dataset_choose + '/' + opt.network_choose + '/result_' + opt.dataset_choose + '_split=' + str(opt.split) + '_' + co_train + '_lr=' + str(opt.learning_rate) + '_' + str(opt.sig_test_run)
                        else:
                            now = 'new_result/final_2181/sig_test/' + opt.dataset_choose + '/' + opt.network_choose + '/result_' + opt.dataset_choose + '_split=' + str(opt.split) + '_' + co_train + '_lr=' + str(opt.learning_rate) + '_' + str(opt.sig_test_run) + '_' + opt.data_use
            elif opt.behavior == True:
                if opt.network_choose == 'shufflenet_v1':
                    now = 'new_result/final_2181/sig_test/'+opt.dataset_choose+'/'+opt.network_choose+'_1.5x/result_' + opt.dataset_choose + '_split=' + str(
                    opt.split) + '_lr=' + str(opt.learning_rate)+'_'+str(opt.sig_test_run)+'_behavior_'+opt.behavior_data
                elif opt.network_choose == 'mobilenet_v2':
                    now = 'new_result/final_2181/sig_test/' + opt.dataset_choose + '/' + opt.network_choose + '_0.45x/result_' + opt.dataset_choose + '_split=' + str(opt.split) + '_lr=' + str(opt.learning_rate) + '_' + str(opt.sig_test_run) + '_behavior_' + opt.behavior_data
                else:
                    now = 'new_result/final_2181/sig_test/' + opt.dataset_choose + '/' + opt.network_choose + '/result_' + opt.dataset_choose + '_split=' + str(opt.split) + '_lr=' + str(opt.learning_rate) + '_' + str(opt.sig_test_run) + '_behavior_' + opt.behavior_data
            opt.result_path = os.path.join(opt.result_path, now)
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
    visual, target,  visualization_item = data_item
    target = target.cuda()

    visual = visual.cuda()

    batch = visual.size(0)
    return visual, target,  visualization_item, batch

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
        RSA = torch.zeros([5,batch,batch]).cuda()
        for i in range(5):
            voxel_select = neural_response['sub-'+str(i+1).zfill(2)].cuda()
            # voxel_select = neural_response['Subject'+str(i+1)].cuda()
            if opt.rho4rdm:
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
        if opt.rho4rdm:
            voxel_select_numpy = voxel_select.cpu().numpy()
            ranks_numpy = rankdata(voxel_select_numpy, method='average', axis=1)
            voxel_ranks = torch.tensor(ranks_numpy, dtype=torch.float32, device=voxel_select.device)
            RSA_output = torch.corrcoef(voxel_ranks)
        else:
            voxel_select = torch.div(voxel_select, torch.norm(voxel_select, p=2, dim=1).reshape(batch, 1))
            RSA_output = torch.mm(voxel_select,voxel_select.transpose(1,0))
    return visual,  RSA_output, batch,visualization_item

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
        visual, target = inputs
        outputs = model(visual,test_svm=test_svm)
        y_pred, alpha, beta, gamma, fSCT = outputs
        return fSCT

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
    S_k = []
    for k,v in out.items():
        if k != 'temporal_attention':
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
        visual_embeddings = visual_embeddings.view(batch_size, -1, 288)

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

import matplotlib.pyplot as plt
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
    print(f"✅ Saved RDM visualization to {save_path}")
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
    