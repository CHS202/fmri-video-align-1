from opts import parse_opts
import random
from core.model import generate_model
from core.loss import get_loss
from core.optimizer import get_optim
from core.utils import local2global_path, get_spatial_transform
from core.dataset import get_training_set, get_validation_set, get_test_set, get_data_loader,get_neural_set,get_neural_loader
from transforms.temporal import TSN
from transforms.target import ClassLabel
from train import train_epoch,co_train_epoch, co_train_epoch_lstm, train_epoch_contribution, co_train_epoch_each_roi, co_train_epoch_each_roi_add_pfc
from validation import val_epoch_class, val_epoch_contribution, val_epoch
import numpy as np
import os
import torch
from torch.optim import Adam
from models.lstm import fMRI_LSTM
import copy
from core.utils import run_model_get_contribution, run_model_get_contribution_v2, run_model_get_contribution_v3, run_model_get_contribution_per_layer, run_model_get_contribution_each_roi

from models.visual_stream import VisualStream
from models.visual_stream import CNN_3D
from datasets.ve8 import NeuralValidDataset
import torch.nn as nn

os.environ['CUDA_VISIBLE_DEVICES']='0'
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:32"
os.environ['TF_XLA_FLAGS'] = '--tf_xla_enable_xla_devices'
os.environ["CUDA_DEVICE_ORDER"]= "PCI_BUS_ID"
# os.environ['TORCH_SHOW_CPP_STACKTRACES'] = '1'
os.system("echo $CUDA_VISIBLE_DEVICES")
os.system("echo $PYTORCH_CUDA_ALLOC_CONF")

import gc
import pandas as pd
import sys

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.enabled = False
    # os.environ['PYTHONHASHSEED'] = str(seed)
    # random.seed(seed)
    # np.random.seed(seed)
    # torch.manual_seed(seed)
    # torch.cuda.manual_seed(seed)
    # torch.cuda.manual_seed_all(seed)
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False
    # torch.backends.cudnn.enabled = True

def main_sig(alpha,neural_response,sig_test_run,network,split,roi):
    opt = parse_opts()
    opt.alpha = alpha
    opt.sig_test_run = sig_test_run
    opt.network_choose = network
    opt.split = split
    opt.roi = roi
    print("ROI:",opt.roi, "SPLIT:",opt.split, "neural_response length:",len(neural_response))
    if opt.dataset_choose=='ve8':
        opt.video_path = 'VideoEmotion8--imgs'
        opt.video_raw_path = 'VideoEmotion8--raw'
        opt.annotation_path = 'video_id_ve8.csv'
        opt.n_classes = 8
    elif opt.dataset_choose=='ek6':
        opt.video_path = 'EK6--imgs'
        opt.video_raw_path = 'EK6--raw'
        opt.annotation_path = 'video_id_ek6.csv'
        opt.n_classes = 6
    elif opt.dataset_choose=='rt':
        if opt.task == 'design':
            opt.video_path = 'RT--imgs'
            opt.video_raw_path = 'RT--raw'
            opt.annotation_path = 'video_id_rt.csv'
            opt.n_classes = 4
        elif opt.task == 'space':
            opt.video_path = 'RT--imgs'
            opt.video_raw_path = 'RT--raw'
            opt.annotation_path = 'video_id_rt.csv'
            opt.n_classes = 8
        elif opt.task == 'annot' or opt.task == 'annot-reg':
            opt.video_path = 'RT--imgs'
            opt.video_raw_path = 'RT--raw'
            opt.annotation_path = 'video_id_rt_annot.csv'
            opt.n_classes = 15

    if opt.train_from_checkpoint:
        # neurostorm win4
        # if opt.network_choose == 'mobilenet_v1':
        #     opt.model_pretrained = f'BrainGuided/debug/minor/new_result/final_2181/sig_test_neurostorm_win4/rt/mobilenet_v1/result_rt_split={opt.split}_co_train_alpha=5_use_model=1_lr=0.0002_1_{opt.data_use}/checkpoints/best.pth'
        # elif opt.network_choose == 'mobilenet_v2':
        #     opt.model_pretrained = f'BrainGuided/debug/minor/new_result/final_2181/sig_test_neurostorm_win4/rt/mobilenet_v2_0.45x/result_rt_split={opt.split}_co_train_alpha=5_use_model=1_lr=0.0002_1_{opt.data_use}/checkpoints/best.pth'
        # elif opt.network_choose == 'resnet_18':
        #     opt.model_pretrained = f'BrainGuided/debug/minor/new_result/final_2181/sig_test_neurostorm_win4/rt/resnet_18/result_rt_split={opt.split}_co_train_alpha=5_use_model=1_lr=0.0002_1_{opt.data_use}/checkpoints/best.pth'
        # elif opt.network_choose == 'shufflenet_v1':
        #     opt.model_pretrained = f'BrainGuided/debug/minor/new_result/final_2181/sig_test_neurostorm_win4/rt/shufflenet_v1_1.5x/result_rt_split={opt.split}_co_train_alpha=5_use_model=1_lr=0.0002_1_{opt.data_use}/checkpoints/best.pth'
        # elif opt.network_choose == 'shufflenet_v2':
        #     opt.model_pretrained = f'BrainGuided/debug/minor/new_result/final_2181/sig_test_neurostorm_win4/rt/shufflenet_v2/result_rt_split={opt.split}_co_train_alpha=5_use_model=1_lr=0.0002_1_{opt.data_use}/checkpoints/best.pth'
        # elif opt.network_choose == 'squeezenet':
        #     opt.model_pretrained = f'BrainGuided/debug/minor/new_result/final_2181/sig_test_neurostorm_win4/rt/squeezenet/result_rt_split={opt.split}_co_train_alpha=5_use_model=1_lr=0.0002_1_{opt.data_use}/checkpoints/best.pth'
        
        # not-co-trained
        # if opt.network_choose == 'mobilenet_v1':
        #     opt.model_pretrained = f'BrainGuided/debug/minor/new_result/final_2181/sig_test/rt/mobilenet_v1/result_rt_split={opt.split}_not_co_train_lr=0.0002_1_744/checkpoints/best.pth'
        # elif opt.network_choose == 'mobilenet_v2':
        #     opt.model_pretrained = f'BrainGuided/debug/minor/new_result/final_2181/sig_test/rt/mobilenet_v2_0.45x/result_rt_split={opt.split}_not_co_train_lr=0.0002_1_744/checkpoints/best.pth'
        # elif opt.network_choose == 'resnet_18':
        #     opt.model_pretrained = f'BrainGuided/debug/minor/new_result/final_2181/sig_test/rt/resnet_18/result_rt_split={opt.split}_not_co_train_lr=0.0002_1_744/checkpoints/best.pth'
        # elif opt.network_choose == 'shufflenet_v1':
        #     opt.model_pretrained = f'BrainGuided/debug/minor/new_result/final_2181/sig_test/rt/shufflenet_v1_1.5x/result_rt_split={opt.split}_not_co_train_lr=0.0002_1_744/checkpoints/best.pth'
        # elif opt.network_choose == 'shufflenet_v2':
        #     opt.model_pretrained = f'BrainGuided/debug/minor/new_result/final_2181/sig_test/rt/shufflenet_v2/result_rt_split={opt.split}_not_co_train_lr=0.0002_1_744/checkpoints/best.pth'
        # elif opt.network_choose == 'squeezenet':
        #     opt.model_pretrained = f'BrainGuided/debug/minor/new_result/final_2181/sig_test/rt/squeezenet/result_rt_split={opt.split}_not_co_train_lr=0.0002_1_744/checkpoints/best.pth'
        
        # ce-cka-Fu
        if opt.network_choose == 'mobilenet_v1':
            opt.model_pretrained = f'BrainGuided/debug/minor/new_result/final_2181/sig_test/rt/mobilenet_v1/result_rt_split={opt.split}_co_train_alpha=5_use_model=1_lr=0.0002_1_{opt.data_use}/checkpoints/best.pth'
        elif opt.network_choose == 'mobilenet_v2':
            opt.model_pretrained = f'BrainGuided/debug/minor/new_result/final_2181/sig_test/rt/mobilenet_v2_0.45x/result_rt_split={opt.split}_co_train_alpha=5_use_model=1_lr=0.0002_1_{opt.data_use}/checkpoints/best.pth'
        elif opt.network_choose == 'resnet_18':
            opt.model_pretrained = f'BrainGuided/debug/minor/new_result/final_2181/sig_test/rt/resnet_18/result_rt_split={opt.split}_co_train_alpha=5_use_model=1_lr=0.0002_1_{opt.data_use}/checkpoints/best.pth'
        elif opt.network_choose == 'shufflenet_v1':
            opt.model_pretrained = f'BrainGuided/debug/minor/new_result/final_2181/sig_test/rt/shufflenet_v1_1.5x/result_rt_split={opt.split}_co_train_alpha=5_use_model=1_lr=0.0002_1_{opt.data_use}/checkpoints/best.pth'
        elif opt.network_choose == 'shufflenet_v2':
            opt.model_pretrained = f'BrainGuided/debug/minor/new_result/final_2181/sig_test/rt/shufflenet_v2/result_rt_split={opt.split}_co_train_alpha=5_use_model=1_lr=0.0002_1_{opt.data_use}/checkpoints/best.pth'
        elif opt.network_choose == 'squeezenet':
            opt.model_pretrained = f'BrainGuided/debug/minor/new_result/final_2181/sig_test/rt/squeezenet/result_rt_split={opt.split}_co_train_alpha=5_use_model=1_lr=0.0002_1_{opt.data_use}/checkpoints/best.pth'


    else:
        if opt.network_choose == 'mobilenet_v1':
            opt.model_pretrained = 'pretrained-models/kinetics_mobilenet_2.0x_RGB_16_best.pth'
        elif opt.network_choose == 'mobilenet_v2':
            opt.model_pretrained = 'pretrained-models/kinetics_mobilenetv2_0.45x_RGB_16_best.pth'
        elif opt.network_choose == 'resnet_18':
            opt.model_pretrained = 'pretrained-models/resnet-18-kinetics.pth'
        elif opt.network_choose == 'shufflenet_v1':
            opt.model_pretrained = 'pretrained-models/kinetics_shufflenet_1.5x_G3_RGB_16_best.pth'
        elif opt.network_choose == 'shufflenet_v2':
            opt.model_pretrained = 'pretrained-models/kinetics_shufflenetv2_2.0x_RGB_16_best.pth'
        elif opt.network_choose == 'squeezenet':
            opt.model_pretrained = 'pretrained-models/kinetics_squeezenet_RGB_16_best.pth'
        elif opt.network_choose == 'alexnet_3d':
            opt.model_pretrained = None
        elif opt.network_choose == 'vit_3d':
            opt.model_pretrained = None  # no compatible checkpoint exists; triggers inflation fallback
        elif opt.network_choose == 'video_swin':
            opt.model_pretrained = None  # None -> torchvision auto-downloads real Kinetics-400 weights
    print('Experiment information:', 'video_path=', opt.video_path, 'annotation_path=', opt.annotation_path,
          'n_classes=', opt.n_classes, 'dataset_choose=', opt.dataset_choose,'n_epoch=',opt.n_epochs)
    # print('alpha=',opt.alpha)
    # print(f"loading model from {opt.model_pretrained}")
    local2global_path(opt)
    if opt.train_from_checkpoint: # trained from 
        model = torch.load(opt.model_pretrained)
        # print(f"loading model from {opt.model_pretrained}")
        if opt.task == 'annot-reg' or opt.task == 'annot':
            if opt.network_choose in ['mobilenet_v1', 'mobilenet_v2', 'shufflenet_v1', 'shufflenet_v2', 'squeezenet']:
                model.n_classes = opt.n_classes
                model.CNN.classifier = nn.Linear(model.CNN.classifier[1].in_features, opt.n_classes)
            elif opt.network_choose == 'resnet_18':
                in_features = model.fc.in_features
                model.fc = nn.Linear(in_features, opt.n_classes)

        model = model.cuda()
        # total_params = 0

        # # --- freeze all but the last layer of the model if training from pre-cotrained model ---
        # # 1. Freeze all the parameters in the model
        # for param in model.parameters():
        #     param.requires_grad = False

        # # 2. Unfreeze the parameters of the final fully connected layer
        # #    IMPORTANT: Replace 'model.fc' with the actual name of your last layer
        # if opt.network_choose in ['mobilenet_v1', 'mobilenet_v2', 'shufflenet_v1', 'shufflenet_v2', 'squeezenet']:
        #     for param in model.CNN.classifier.parameters():
        #         param.requires_grad = True
        # elif opt.network_choose == 'resnet_18':
        #     for param in model.fc.parameters():
        #         param.requires_grad = True
        # # ---------------------------------------------------------

        # for parameter in model.parameters():
        #     if not parameter.requires_grad: continue
        #     param = parameter.numel()
        #     total_params += param

        # parameters = model.parameters()
    # elif opt.align_only_last_layer:
    #     model, _, _ = generate_model(opt)
    else:
        model, parameters,total_params = generate_model(opt)

    # ------------------------------------------------------
    if opt.freezeall:
        total_params = 0

        # --- freeze all but the last layer of the model if training from pre-cotrained model ---
        # 1. Freeze all the parameters in the model
        for param in model.parameters():
            param.requires_grad = False

        # 2. Unfreeze the parameters of the final fully connected layer
        #    IMPORTANT: Replace 'model.fc' with the actual name of your last layer
        if opt.network_choose in ['mobilenet_v1', 'mobilenet_v2', 'shufflenet_v1', 'shufflenet_v2', 'squeezenet']:
            for param in model.CNN.classifier.parameters():
                param.requires_grad = True
        elif opt.network_choose == 'resnet_18':
            for param in model.fc.parameters():
                param.requires_grad = True
            

        for parameter in model.parameters():
            if not parameter.requires_grad: continue
            param = parameter.numel()
            total_params += param

        parameters = model.parameters()
    elif opt.freezehalf:
        total_params = 0
        # --- freeze all but the last layer of the model and fc layers ---
        # 1. Freeze all the parameters in the model
        for param in model.parameters():
            param.requires_grad = False

        if opt.network_choose == 'mobilenet_v1':
            for param in model.CNN.features[-1].parameters():
                param.requires_grad = True
            for param in model.CNN.classifier.parameters():
                param.requires_grad = True
        elif opt.network_choose == 'mobilenet_v2':
            for param in model.CNN.features[-1].parameters():
                param.requires_grad = True
            for param in model.CNN.classifier.parameters():
                param.requires_grad = True
        elif opt.network_choose == 'shufflenet_v1':
            for param in model.CNN.layer3.parameters():
                param.requires_grad = True
            for param in model.CNN.classifier.parameters():
                param.requires_grad = True
        elif opt.network_choose == 'shufflenet_v2':
            for param in model.CNN.conv_last.parameters():
                param.requires_grad = True
            for param in model.CNN.classifier.parameters():
                param.requires_grad = True
        elif opt.network_choose == 'squeezenet':
            for param in model.CNN.features[-1].parameters():
                param.requires_grad = True
            for param in model.CNN.classifier.parameters():
                param.requires_grad = True
        elif opt.network_choose == 'resnet_18':
            for param in model.resnet[7].parameters():
                param.requires_grad = True
            for param in model.fc.parameters():
                param.requires_grad = True

        for parameter in model.parameters():
            if not parameter.requires_grad: continue
            param = parameter.numel()
            total_params += param

        parameters = model.parameters()
    
    # ------------------------------------

    print(f"Total Trainable Params: {total_params}")
    criterion = get_loss(opt)
    criterion = criterion.cuda()
    optimizer = get_optim(opt, parameters)

    if opt.use_lstm == True:
        # --- NEW: Instantiate the fMRI LSTM Model ---
        # The number of features is the last dimension of your fMRI data array
        fmri_input_features = 7545 
        fmri_model = fMRI_LSTM(input_features=fmri_input_features)
        fmri_model = fmri_model.cuda()
        
        # Create a separate optimizer for the fMRI model
        # You might want to add a new learning rate option to `opt` for this
        fmri_optimizer = Adam(fmri_model.parameters(), lr=opt.learning_rate)
        # use RMSprop optimizer and add clipping
        # fmri_optimizer = torch.optim.RMSprop(fmri_model.parameters(), lr=opt.learning_rate)
        print(f"Total Trainable Params (fMRI LSTM): {sum(p.numel() for p in fmri_model.parameters() if p.requires_grad)}")

    # train
    spatial_transform = get_spatial_transform(opt, 'train')
    temporal_transform = TSN(seq_len=opt.seq_len, snippet_duration=opt.snippet_duration, center=False)
    target_transform = ClassLabel()
    training_data = get_training_set(opt, spatial_transform, temporal_transform, target_transform)
    train_loader = get_data_loader(opt, training_data, shuffle=True)

    # validation
    spatial_transform = get_spatial_transform(opt, 'test')
    temporal_transform = TSN(seq_len=opt.seq_len, snippet_duration=opt.snippet_duration, center=False)
    target_transform = ClassLabel()
    validation_data = get_validation_set(opt, spatial_transform, temporal_transform, target_transform)
    val_loader = get_data_loader(opt, validation_data, shuffle=False)
    if opt.co_train == True:
        neural_data = get_neural_set(opt, spatial_transform, temporal_transform,neural_response)
        neural_loader = get_neural_loader(opt, neural_data, shuffle=True)

    
    result = np.zeros((opt.n_epochs,2))
    if opt.use_lstm:
        loss_result = np.zeros((opt.n_epochs,6))
    elif opt.train_from_checkpoint:
        loss_result = np.zeros((opt.n_epochs,3))
    elif opt.align_only_last_layer:
        loss_result = np.zeros((opt.n_epochs,6))
    else:
        loss_result = np.zeros((opt.n_epochs,5))

    if not opt.align_only_last_layer:
        if opt.network_choose == 'resnet_18':
            gamma = np.zeros((opt.n_epochs, 5))
        elif opt.network_choose == 'squeezenet':
            gamma = np.zeros((opt.n_epochs, 5))
        elif opt.network_choose == 'shufflenet_v1':
            gamma = np.zeros((opt.n_epochs, 4))
        elif opt.network_choose == 'shufflenet_v2':
            gamma = np.zeros((opt.n_epochs, 3))
        elif opt.network_choose == 'mobilenet_v1':
            gamma = np.zeros((opt.n_epochs, 6))
        elif opt.network_choose == 'mobilenet_v2':
            gamma = np.zeros((opt.n_epochs, 3))
        elif opt.network_choose == 'vaa':
            gamma = np.zeros((opt.n_epochs, 3))
        elif opt.network_choose == 'alexnet_3d':
            gamma = np.zeros((opt.n_epochs, 5))
        elif opt.network_choose == 'vit_3d':
            gamma = np.zeros((opt.n_epochs, 4))  # match self.gamma size above
        elif opt.network_choose == 'video_swin':
            gamma = np.zeros((opt.n_epochs, 4))
    # gamma_result = np.zeros((opt.n_epochs,gamma.shape[1]+1))
    if opt.single_annot_class == True:
        if opt.target_class == 'obj':
            class_accuracy = np.zeros((opt.n_epochs, 4))
        elif opt.target_class == 'subj':
            class_accuracy = np.zeros((opt.n_epochs, 11))
        else:
            class_accuracy = np.zeros((opt.n_epochs, 1))
    else:
        class_accuracy = np.zeros((opt.n_epochs,opt.n_classes))
        if opt.task == 'annot':
            class_auc = np.zeros((opt.n_epochs,opt.n_classes))
            class_youden = np.zeros((opt.n_epochs,opt.n_classes))
            class_acc_at_bestthres = np.zeros((opt.n_epochs,opt.n_classes))
    max_acc = -np.inf # set max acc to 0
    min_loss = np.inf
    # set early stopping
    if opt.task == 'annot' or opt.task == 'annot-reg':
        patience_cnt = 10
    else:
        patience_cnt = 10
    patience = patience_cnt
    for i in range(1, opt.n_epochs + 1):
        if opt.co_train == True:
            if opt.train_from_checkpoint: # train from pre-cotrained model
                train_loss = train_epoch(i, train_loader, model, criterion, optimizer, opt, training_data.class_names)
            else:
                if opt.use_lstm == True:
                    # ✅ MODIFIED: Pass the fMRI model and its optimizer to the training function
                    gamma_temp, total_loss, ce_loss, sim_loss, lstm_ce_loss = co_train_epoch_lstm(
                        i, train_loader, neural_loader, model, fmri_model, 
                        criterion, optimizer, fmri_optimizer, opt
                    )  
                    gamma[i-1] = gamma_temp.detach().cpu().numpy()
                elif opt.align_only_last_layer == True:
                    total_loss, ce_loss, sim_loss, cosine_sim = co_train_epoch(i, train_loader, neural_loader, model, criterion, optimizer, opt)
                elif opt.dapello == True:
                    print('Using Dapello')
                    total_loss, ce_loss, sim_loss = co_train_epoch(i, train_loader, neural_loader, model, criterion, optimizer, opt)
                else:
                    gamma_temp, total_loss, ce_loss, sim_loss = co_train_epoch(i, train_loader, neural_loader, model, criterion, optimizer, opt)
                    gamma[i-1] = gamma_temp.detach().cpu().numpy()
        else:
            train_loss = train_epoch(i, train_loader, model, criterion, optimizer, opt, training_data.class_names)

        if opt.task == 'annot':
            ep,acc,ac, loss, auc, youden, acc_at_bestthres = val_epoch_class(i, val_loader, model, criterion, opt, optimizer)
            # class_auc[i-1] = auc
            # class_youden[i-1] = youden
            # class_acc_at_bestthres[i-1] = acc_at_bestthres
            # np.savetxt(os.path.join(opt.result_path,'class_auc.csv'), class_auc,delimiter = ',')
            # np.savetxt(os.path.join(opt.result_path,'class_youden.csv'), class_youden,delimiter = ',')
            # np.savetxt(os.path.join(opt.result_path,'class_acc_at_bestthres.csv'), class_acc_at_bestthres,delimiter = ',')
        else:
            ep,acc,ac, loss = val_epoch_class(i, val_loader, model, criterion, opt, optimizer)

        result[i-1,0] = ep
        result[i-1,1] = acc
        class_accuracy[i-1] = ac
        acc_result = np.concatenate((result,class_accuracy),axis=1)
        
        if opt.use_lstm:
            loss_result[i-1] = [ep, total_loss, ce_loss, sim_loss, lstm_ce_loss, loss]
        elif opt.train_from_checkpoint:
            loss_result[i-1] = [ep, train_loss, loss]
        elif opt.align_only_last_layer:
            loss_result[i-1] = [ep, total_loss, ce_loss, sim_loss, cosine_sim, loss]
        else:
            loss_result[i-1] = [ep, total_loss, ce_loss, sim_loss, loss]
        # gamma_result[i-1, 0] = ep
        # gamma_result[i-1, 1:] = gamma[i-1]
        np.savetxt(os.path.join(opt.result_path,'acc_result.csv'), acc_result,delimiter = ',')
        np.savetxt(os.path.join(opt.result_path,'loss_result.csv'), loss_result,delimiter = ',')
        # np.savetxt(os.path.join(opt.result_path,'gamma_result.csv'), gamma_result,delimiter = ',')
        np.save(os.path.join(opt.result_path,'result.npy'),result)
        # save model
        # if opt.task == 'annot':
        #     acc = np.mean(auc)
        if acc > max_acc:
            max_acc = acc
            save_file_path = os.path.join(opt.ckpt_path, 'best.pth'.format(i))
            # states = {
            #     'epoch': i,
            #     'state_dict': copy.deepcopy(model.state_dict()),
            #     'optimizer': optimizer.state_dict(),
            # }
            # torch.save(states, save_file_path)
            torch.save(model, save_file_path)
            patience = patience_cnt
        else:
            patience = patience - 1
            if patience == 0:
                break
        # if i % 10 == 0:
        #     save_file_path = os.path.join(opt.ckpt_path, 'last_checkpoint.pth')
        #     states = {
        #         'epoch': i,
        #         'state_dict': copy.deepcopy(model.state_dict()),
        #         'optimizer': optimizer.state_dict(),
        #     }
        #     torch.save(states, save_file_path)
        #     print('save model at epoch {}'.format(i))
    if opt.co_train == True and opt.dapello == False:
        # np.save(os.path.join(opt.result_path,'gamma.npy'),gamma)
        if (not opt.train_from_checkpoint) and (not opt.align_only_last_layer):
            np.savetxt(os.path.join(opt.result_path,'gamma_result.csv'), gamma,delimiter = ',')
def main_sig_each_roi(alpha,neural_response_evc,neural_response_tos,neural_response_ppa,neural_response_rsc,sig_test_run,network,split):
    opt = parse_opts()
    opt.alpha = alpha
    opt.sig_test_run = sig_test_run
    opt.network_choose = network
    opt.split = split
    
    print("SPLIT:",opt.split, "opt.data_use", opt.data_use, "neural_response_evc length:",len(neural_response_evc), "neural_response_tos length:",len(neural_response_tos), "neural_response_ppa length:",len(neural_response_ppa), "neural_response_rsc length:",len(neural_response_rsc))

    if opt.task == 'design':
        opt.video_path = 'RT--imgs'
        opt.video_raw_path = 'RT--raw'
        opt.annotation_path = 'video_id_rt.csv'
        opt.n_classes = 4
    elif opt.task == 'annot' or opt.task == 'annot-reg':
        opt.video_path = 'RT--imgs'
        opt.video_raw_path = 'RT--raw'
        opt.annotation_path = 'video_id_rt_annot.csv'
        opt.n_classes = 15

    if opt.network_choose == 'mobilenet_v1':
        opt.model_pretrained = 'pretrained-models/kinetics_mobilenet_2.0x_RGB_16_best.pth'
    elif opt.network_choose == 'mobilenet_v2':
        opt.model_pretrained = 'pretrained-models/kinetics_mobilenetv2_0.45x_RGB_16_best.pth'
    elif opt.network_choose == 'resnet_18':
        opt.model_pretrained = 'pretrained-models/resnet-18-kinetics.pth'
    elif opt.network_choose == 'shufflenet_v1':
        opt.model_pretrained = 'pretrained-models/kinetics_shufflenet_1.5x_G3_RGB_16_best.pth'
    elif opt.network_choose == 'shufflenet_v2':
        opt.model_pretrained = 'pretrained-models/kinetics_shufflenetv2_2.0x_RGB_16_best.pth'
    elif opt.network_choose == 'squeezenet':
        opt.model_pretrained = 'pretrained-models/kinetics_squeezenet_RGB_16_best.pth'
    elif opt.network_choose == 'alexnet_3d':
        opt.model_pretrained = None
    elif opt.network_choose == 'vit_3d':
        opt.model_pretrained = None  # no compatible checkpoint exists; triggers inflation fallback
    elif opt.network_choose == 'video_swin':
        opt.model_pretrained = None  # None -> torchvision auto-downloads real Kinetics-400 weights

    print('Experiment information:', 'video_path=', opt.video_path, 'annotation_path=', opt.annotation_path,
          'n_classes=', opt.n_classes, 'dataset_choose=', opt.dataset_choose,'n_epoch=',opt.n_epochs)
    local2global_path(opt)
    
    model, parameters,total_params = generate_model(opt)
    print(f"Total Trainable Params: {total_params}")
    criterion = get_loss(opt)
    criterion = criterion.cuda()
    optimizer = get_optim(opt, parameters)

    # train
    spatial_transform = get_spatial_transform(opt, 'train')
    temporal_transform = TSN(seq_len=opt.seq_len, snippet_duration=opt.snippet_duration, center=False)
    target_transform = ClassLabel()
    training_data = get_training_set(opt, spatial_transform, temporal_transform, target_transform)
    train_loader = get_data_loader(opt, training_data, shuffle=True)

    # validation
    spatial_transform = get_spatial_transform(opt, 'test')
    temporal_transform = TSN(seq_len=opt.seq_len, snippet_duration=opt.snippet_duration, center=False)
    target_transform = ClassLabel()
    validation_data = get_validation_set(opt, spatial_transform, temporal_transform, target_transform)
    val_loader = get_data_loader(opt, validation_data, shuffle=False)
    if opt.co_train == True:
        neural_data_evc = get_neural_set(opt, spatial_transform, temporal_transform,neural_response_evc)
        neural_loader_evc = get_neural_loader(opt, neural_data_evc, shuffle=True)
        neural_data_tos = get_neural_set(opt, spatial_transform, temporal_transform,neural_response_tos)
        neural_loader_tos = get_neural_loader(opt, neural_data_tos, shuffle=True)
        neural_data_ppa = get_neural_set(opt, spatial_transform, temporal_transform,neural_response_ppa)
        neural_loader_ppa = get_neural_loader(opt, neural_data_ppa, shuffle=True)
        neural_data_rsc = get_neural_set(opt, spatial_transform, temporal_transform,neural_response_rsc)
        neural_loader_rsc = get_neural_loader(opt, neural_data_rsc, shuffle=True)

    result = np.zeros((opt.n_epochs,2))
    loss_result = np.zeros((opt.n_epochs,5))
    if opt.network_choose == 'resnet_18':
        gamma = np.zeros((opt.n_epochs, 20))
    elif opt.network_choose == 'squeezenet':
        gamma = np.zeros((opt.n_epochs, 20))
    elif opt.network_choose == 'shufflenet_v1':
        gamma = np.zeros((opt.n_epochs, 16))
    elif opt.network_choose == 'shufflenet_v2':
        gamma = np.zeros((opt.n_epochs, 12))
    elif opt.network_choose == 'mobilenet_v1':
        gamma = np.zeros((opt.n_epochs, 24))
    elif opt.network_choose == 'mobilenet_v2':
        gamma = np.zeros((opt.n_epochs, 12))
    elif opt.network_choose == 'alexnet_3d':
        gamma = np.zeros((opt.n_epochs, 20))
    elif opt.network_choose == 'vit_3d':
        gamma = np.zeros((opt.n_epochs, 16))  # match self.gamma size above
    elif opt.network_choose == 'video_swin':
        gamma = np.zeros((opt.n_epochs, 16))

    if opt.single_annot_class == True:
        if opt.target_class == 'obj':
            class_accuracy = np.zeros((opt.n_epochs, 4))
        elif opt.target_class == 'subj':
            class_accuracy = np.zeros((opt.n_epochs, 11))
        else:
            class_accuracy = np.zeros((opt.n_epochs, 1))
    else:
        class_accuracy = np.zeros((opt.n_epochs,opt.n_classes))
        if opt.task == 'annot':
            class_auc = np.zeros((opt.n_epochs,opt.n_classes))
            class_youden = np.zeros((opt.n_epochs,opt.n_classes))
            class_acc_at_bestthres = np.zeros((opt.n_epochs,opt.n_classes))
    max_acc = -np.inf # set max acc to 0

    if opt.task == 'annot' or opt.task == 'annot-reg':
        patience_cnt = 10
    else:
        patience_cnt = 10
    patience = patience_cnt
    for i in range(1, opt.n_epochs + 1):
        if opt.co_train == True:
            gamma_temp, total_loss, ce_loss, sim_loss = co_train_epoch_each_roi(i, train_loader, neural_loader_evc, neural_loader_tos, neural_loader_ppa, neural_loader_rsc, model, criterion, optimizer, opt)
            gamma[i-1] = gamma_temp.detach().cpu().numpy()
        else:
            train_loss = train_epoch(i, train_loader, model, criterion, optimizer, opt, training_data.class_names)
        if opt.task == 'annot':
            ep,acc,ac, loss, auc, youden, acc_at_bestthres = val_epoch_class(i, val_loader, model, criterion, opt, optimizer)
            class_auc[i-1] = auc
            class_youden[i-1] = youden
            class_acc_at_bestthres[i-1] = acc_at_bestthres
            np.savetxt(os.path.join(opt.result_path,'class_auc.csv'), class_auc,delimiter = ',')
            np.savetxt(os.path.join(opt.result_path,'class_youden.csv'), class_youden,delimiter = ',')
            np.savetxt(os.path.join(opt.result_path,'class_acc_at_bestthres.csv'), class_acc_at_bestthres,delimiter = ',')
        else:
            ep,acc,ac, loss = val_epoch_class(i, val_loader, model, criterion, opt, optimizer)
        result[i-1,0] = ep
        result[i-1,1] = acc
        class_accuracy[i-1] = ac
        acc_result = np.concatenate((result,class_accuracy),axis=1)
        
        loss_result[i-1] = [ep, total_loss, ce_loss, sim_loss, loss]
        # gamma_result[i-1, 0] = ep
        # gamma_result[i-1, 1:] = gamma[i-1]
        np.savetxt(os.path.join(opt.result_path,'acc_result.csv'), acc_result,delimiter = ',')
        np.savetxt(os.path.join(opt.result_path,'loss_result.csv'), loss_result,delimiter = ',')
        # np.savetxt(os.path.join(opt.result_path,'gamma_result.csv'), gamma_result,delimiter = ',')
        np.save(os.path.join(opt.result_path,'result.npy'),result)
        # if loss == nan break
        # if np.isnan(total_loss):
        #     break
        # save model
        # if opt.task == 'annot':
        #     acc = np.mean(auc)
        if acc > max_acc:
            max_acc = acc
            save_file_path = os.path.join(opt.ckpt_path, 'best.pth'.format(i))
            # states = {
            #     'epoch': i,
            #     'state_dict': copy.deepcopy(model.state_dict()),
            #     'optimizer': optimizer.state_dict(),
            # }
            # torch.save(states, save_file_path)
            torch.save(model, save_file_path)
            patience = patience_cnt
        else:
            patience = patience - 1
            if patience == 0:
                break
        # if i % 10 == 0:
        #     save_file_path = os.path.join(opt.ckpt_path, 'last_checkpoint.pth')
        #     states = {
        #         'epoch': i,
        #         'state_dict': copy.deepcopy(model.state_dict()),
        #         'optimizer': optimizer.state_dict(),
        #     }
        #     torch.save(states, save_file_path)
        #     print('save model at epoch {}'.format(i))
    if opt.co_train == True:
        layer_names_map = {
            'resnet_18':     ['conv1', 'conv5', 'conv9', 'conv13', 'conv17'],
            'squeezenet':    ['conv1', 'Fire3', 'Fire5', 'Fire7', 'Fire9'],
            'shufflenet_v1': ['conv1', 'layer1', 'layer2', 'layer3'],
            'shufflenet_v2': ['conv1', 'features', 'conv_last'],
            'mobilenet_v1':  ['64channels', '128channels', '256channels', '512channels', '1024channels', '2048channels'],
            'mobilenet_v2':  ['14channels', '144channels', '1280channels'],
            'vit_3d':        ['block3', 'block6', 'block9', 'block12'],
            'video_swin':    ['stage1', 'stage2', 'stage3', 'stage4'],
        }

        layer_names = layer_names_map[opt.network_choose]
        suffixes = ['_evc', '_opa', '_ppa', '_rsc']
        header = ','.join(f"{name}{suffix}" for suffix in suffixes for name in layer_names)
        # np.save(os.path.join(opt.result_path,'gamma.npy'),gamma)
        if (not opt.train_from_checkpoint) and (not opt.align_only_last_layer):
            np.savetxt(os.path.join(opt.result_path,'gamma_result.csv'), gamma,delimiter = ',', header=header, comments='')

def main_sig_each_roi_add_pfc(alpha,neural_response_evc,neural_response_tos,neural_response_ppa,neural_response_rsc,neural_response_pfc,sig_test_run,network,split):
    opt = parse_opts()
    opt.alpha = alpha
    opt.sig_test_run = sig_test_run
    opt.network_choose = network
    opt.split = split
    
    print("SPLIT:",opt.split, "opt.data_use", opt.data_use, "neural_response_evc length:",len(neural_response_evc), "neural_response_tos length:",len(neural_response_tos), 
          "neural_response_ppa length:",len(neural_response_ppa), "neural_response_rsc length:",len(neural_response_rsc), "neural_response_pfc length:",len(neural_response_pfc))

    if opt.task == 'design':
        opt.video_path = 'RT--imgs'
        opt.video_raw_path = 'RT--raw'
        opt.annotation_path = 'video_id_rt.csv'
        opt.n_classes = 4
    elif opt.task == 'annot' or opt.task == 'annot-reg':
        opt.video_path = 'RT--imgs'
        opt.video_raw_path = 'RT--raw'
        opt.annotation_path = 'video_id_rt_annot.csv'
        opt.n_classes = 15

    if opt.network_choose == 'mobilenet_v1':
        opt.model_pretrained = 'pretrained-models/kinetics_mobilenet_2.0x_RGB_16_best.pth'
    elif opt.network_choose == 'mobilenet_v2':
        opt.model_pretrained = 'pretrained-models/kinetics_mobilenetv2_0.45x_RGB_16_best.pth'
    elif opt.network_choose == 'resnet_18':
        opt.model_pretrained = 'pretrained-models/resnet-18-kinetics.pth'
    elif opt.network_choose == 'shufflenet_v1':
        opt.model_pretrained = 'pretrained-models/kinetics_shufflenet_1.5x_G3_RGB_16_best.pth'
    elif opt.network_choose == 'shufflenet_v2':
        opt.model_pretrained = 'pretrained-models/kinetics_shufflenetv2_2.0x_RGB_16_best.pth'
    elif opt.network_choose == 'squeezenet':
        opt.model_pretrained = 'pretrained-models/kinetics_squeezenet_RGB_16_best.pth'
    elif opt.network_choose == 'alexnet_3d':
        opt.model_pretrained = None
    elif opt.network_choose == 'vit_3d':
        opt.model_pretrained = None  # no compatible checkpoint exists; triggers inflation fallback
    elif opt.network_choose == 'video_swin':
        opt.model_pretrained = None  # None -> torchvision auto-downloads real Kinetics-400 weights

    print('Experiment information:', 'video_path=', opt.video_path, 'annotation_path=', opt.annotation_path,
          'n_classes=', opt.n_classes, 'dataset_choose=', opt.dataset_choose,'n_epoch=',opt.n_epochs)
    local2global_path(opt)
    
    model, parameters,total_params = generate_model(opt)
    print(f"Total Trainable Params: {total_params}")
    criterion = get_loss(opt)
    criterion = criterion.cuda()
    optimizer = get_optim(opt, parameters)

    # train
    spatial_transform = get_spatial_transform(opt, 'train')
    temporal_transform = TSN(seq_len=opt.seq_len, snippet_duration=opt.snippet_duration, center=False)
    target_transform = ClassLabel()
    training_data = get_training_set(opt, spatial_transform, temporal_transform, target_transform)
    train_loader = get_data_loader(opt, training_data, shuffle=True)

    # validation
    spatial_transform = get_spatial_transform(opt, 'test')
    temporal_transform = TSN(seq_len=opt.seq_len, snippet_duration=opt.snippet_duration, center=False)
    target_transform = ClassLabel()
    validation_data = get_validation_set(opt, spatial_transform, temporal_transform, target_transform)
    val_loader = get_data_loader(opt, validation_data, shuffle=False)
    if opt.co_train == True:
        neural_data_evc = get_neural_set(opt, spatial_transform, temporal_transform,neural_response_evc)
        neural_loader_evc = get_neural_loader(opt, neural_data_evc, shuffle=True)
        neural_data_tos = get_neural_set(opt, spatial_transform, temporal_transform,neural_response_tos)
        neural_loader_tos = get_neural_loader(opt, neural_data_tos, shuffle=True)
        neural_data_ppa = get_neural_set(opt, spatial_transform, temporal_transform,neural_response_ppa)
        neural_loader_ppa = get_neural_loader(opt, neural_data_ppa, shuffle=True)
        neural_data_rsc = get_neural_set(opt, spatial_transform, temporal_transform,neural_response_rsc)
        neural_loader_rsc = get_neural_loader(opt, neural_data_rsc, shuffle=True)
        neural_data_pfc = get_neural_set(opt, spatial_transform, temporal_transform,neural_response_rsc)
        neural_loader_pfc = get_neural_loader(opt, neural_data_pfc, shuffle=True)

    result = np.zeros((opt.n_epochs,2))
    loss_result = np.zeros((opt.n_epochs,5))
    if opt.network_choose == 'resnet_18':
        gamma = np.zeros((opt.n_epochs, 25))
    elif opt.network_choose == 'squeezenet':
        gamma = np.zeros((opt.n_epochs, 25))
    elif opt.network_choose == 'shufflenet_v1':
        gamma = np.zeros((opt.n_epochs, 20))
    elif opt.network_choose == 'shufflenet_v2':
        gamma = np.zeros((opt.n_epochs, 15))
    elif opt.network_choose == 'mobilenet_v1':
        gamma = np.zeros((opt.n_epochs, 30))
    elif opt.network_choose == 'mobilenet_v2':
        gamma = np.zeros((opt.n_epochs, 15))
    elif opt.network_choose == 'alexnet_3d':
        gamma = np.zeros((opt.n_epochs, 25))
    elif opt.network_choose == 'vit_3d':
        gamma = np.zeros((opt.n_epochs, 20))  # match self.gamma size above
    elif opt.network_choose == 'video_swin':
        gamma = np.zeros((opt.n_epochs, 20))

    if opt.single_annot_class == True:
        if opt.target_class == 'obj':
            class_accuracy = np.zeros((opt.n_epochs, 4))
        elif opt.target_class == 'subj':
            class_accuracy = np.zeros((opt.n_epochs, 11))
        else:
            class_accuracy = np.zeros((opt.n_epochs, 1))
    else:
        class_accuracy = np.zeros((opt.n_epochs,opt.n_classes))
        if opt.task == 'annot':
            class_auc = np.zeros((opt.n_epochs,opt.n_classes))
            class_youden = np.zeros((opt.n_epochs,opt.n_classes))
            class_acc_at_bestthres = np.zeros((opt.n_epochs,opt.n_classes))
    max_acc = -np.inf # set max acc to 0

    if opt.task == 'annot' or opt.task == 'annot-reg':
        patience_cnt = 10
    else:
        patience_cnt = 10
    patience = patience_cnt
    for i in range(1, opt.n_epochs + 1):
        if opt.co_train == True:
            gamma_temp, total_loss, ce_loss, sim_loss = co_train_epoch_each_roi_add_pfc(i, train_loader, neural_loader_evc, neural_loader_tos, neural_loader_ppa, neural_loader_rsc, neural_loader_pfc, model, criterion, optimizer, opt)
            gamma[i-1] = gamma_temp.detach().cpu().numpy()
        else:
            train_loss = train_epoch(i, train_loader, model, criterion, optimizer, opt, training_data.class_names)
        if opt.task == 'annot':
            ep,acc,ac, loss, auc, youden, acc_at_bestthres = val_epoch_class(i, val_loader, model, criterion, opt, optimizer)
            class_auc[i-1] = auc
            class_youden[i-1] = youden
            class_acc_at_bestthres[i-1] = acc_at_bestthres
            np.savetxt(os.path.join(opt.result_path,'class_auc.csv'), class_auc,delimiter = ',')
            np.savetxt(os.path.join(opt.result_path,'class_youden.csv'), class_youden,delimiter = ',')
            np.savetxt(os.path.join(opt.result_path,'class_acc_at_bestthres.csv'), class_acc_at_bestthres,delimiter = ',')
        else:
            ep,acc,ac, loss = val_epoch_class(i, val_loader, model, criterion, opt, optimizer)
        result[i-1,0] = ep
        result[i-1,1] = acc
        class_accuracy[i-1] = ac
        acc_result = np.concatenate((result,class_accuracy),axis=1)
        
        loss_result[i-1] = [ep, total_loss, ce_loss, sim_loss, loss]
        # gamma_result[i-1, 0] = ep
        # gamma_result[i-1, 1:] = gamma[i-1]
        np.savetxt(os.path.join(opt.result_path,'acc_result.csv'), acc_result,delimiter = ',')
        np.savetxt(os.path.join(opt.result_path,'loss_result.csv'), loss_result,delimiter = ',')
        # np.savetxt(os.path.join(opt.result_path,'gamma_result.csv'), gamma_result,delimiter = ',')
        np.save(os.path.join(opt.result_path,'result.npy'),result)
        # if loss == nan break
        # if np.isnan(total_loss):
        #     break
        # save model
        # if opt.task == 'annot':
        #     acc = np.mean(auc)
        if acc > max_acc:
            max_acc = acc
            save_file_path = os.path.join(opt.ckpt_path, 'best.pth'.format(i))
            # states = {
            #     'epoch': i,
            #     'state_dict': copy.deepcopy(model.state_dict()),
            #     'optimizer': optimizer.state_dict(),
            # }
            # torch.save(states, save_file_path)
            torch.save(model, save_file_path)
            patience = patience_cnt
        else:
            patience = patience - 1
            if patience == 0:
                break
        # if i % 10 == 0:
        #     save_file_path = os.path.join(opt.ckpt_path, 'last_checkpoint.pth')
        #     states = {
        #         'epoch': i,
        #         'state_dict': copy.deepcopy(model.state_dict()),
        #         'optimizer': optimizer.state_dict(),
        #     }
        #     torch.save(states, save_file_path)
        #     print('save model at epoch {}'.format(i))
    if opt.co_train == True:
        layer_names_map = {
            'resnet_18':     ['conv1', 'conv5', 'conv9', 'conv13', 'conv17'],
            'squeezenet':    ['conv1', 'Fire3', 'Fire5', 'Fire7', 'Fire9'],
            'shufflenet_v1': ['conv1', 'layer1', 'layer2', 'layer3'],
            'shufflenet_v2': ['conv1', 'features', 'conv_last'],
            'mobilenet_v1':  ['64channels', '128channels', '256channels', '512channels', '1024channels', '2048channels'],
            'mobilenet_v2':  ['14channels', '144channels', '1280channels'],
            'vit_3d':        ['block3', 'block6', 'block9', 'block12'],
            'video_swin':    ['stage1', 'stage2', 'stage3', 'stage4'],
        }

        layer_names = layer_names_map[opt.network_choose]
        suffixes = ['_evc', '_opa', '_ppa', '_rsc', '_pfc']
        header = ','.join(f"{name}{suffix}" for suffix in suffixes for name in layer_names)
        # np.save(os.path.join(opt.result_path,'gamma.npy'),gamma)
        if (not opt.train_from_checkpoint) and (not opt.align_only_last_layer):
            np.savetxt(os.path.join(opt.result_path,'gamma_result.csv'), gamma,delimiter = ',', header=header, comments='')
def main_contribution(alpha,neural_response, neural_response_valid,sig_test_run,network,split,roi):
    opt = parse_opts()
    opt.alpha = 1
    opt.sig_test_run = sig_test_run
    opt.network_choose = network
    opt.split = split
    opt.roi = roi
    if opt.dataset_choose=='rt':
        if opt.task == 'design':
            opt.video_path = 'RT--imgs'
            opt.video_raw_path = 'RT--raw'
            opt.annotation_path = 'video_id_rt.csv'
            opt.n_classes = 4
        elif opt.task == 'annot' or opt.task == 'annot-reg':
            opt.video_path = 'RT--imgs'
            opt.video_raw_path = 'RT--raw'
            opt.annotation_path = 'video_id_rt_annot.csv'
            opt.n_classes = 15
    if opt.get_layer_contribution:
        # not-co-train
        # if opt.network_choose == 'mobilenet_v1':
        #     opt.model_pretrained = f'BrainGuided/debug/minor/new_result/final_2181_annot/sig_test/rt/mobilenet_v1/result_rt_split={opt.split}_not_co_train_lr=0.0002_1_/checkpoints/best.pth'
        # elif opt.network_choose == 'mobilenet_v2':
        #     opt.model_pretrained = f'BrainGuided/debug/minor/new_result/final_2181_annot/sig_test/rt/mobilenet_v2_0.45x/result_rt_split={opt.split}_not_co_train_lr=0.0002_1_/checkpoints/best.pth'
        # elif opt.network_choose == 'resnet_18':
        #     opt.model_pretrained = f'BrainGuided/debug/minor/new_result/final_2181_annot/sig_test/rt/resnet_18/result_rt_split={opt.split}_not_co_train_lr=0.0002_1_/checkpoints/best.pth'
        # elif opt.network_choose == 'shufflenet_v1':
        #     opt.model_pretrained = f'BrainGuided/debug/minor/new_result/final_2181_annot/sig_test/rt/shufflenet_v1_1.5x/result_rt_split={opt.split}_not_co_train_lr=0.0002_1_/checkpoints/best.pth'
        # elif opt.network_choose == 'shufflenet_v2':
        #     opt.model_pretrained = f'BrainGuided/debug/minor/new_result/final_2181_annot/sig_test/rt/shufflenet_v2/result_rt_split={opt.split}_not_co_train_lr=0.0002_1_/checkpoints/best.pth'
        # elif opt.network_choose == 'squeezenet':
        #     opt.model_pretrained = f'BrainGuided/debug/minor/new_result/final_2181_annot/sig_test/rt/squeezenet/result_rt_split={opt.split}_not_co_train_lr=0.0002_1_/checkpoints/best.pth'

        # ce_cka_each_roi
        if opt.network_choose == 'mobilenet_v1':
            opt.model_pretrained = f'BrainGuided/debug/minor/new_result/final_2181_annot/sig_test_each_roi/rt/mobilenet_v1/result_rt_split={opt.split}_co_train_alpha=5_use_model=1_lr=0.0002_1_{opt.data_use}/checkpoints/best.pth'
        elif opt.network_choose == 'mobilenet_v2':
            opt.model_pretrained = f'BrainGuided/debug/minor/new_result/final_2181_annot/sig_test_each_roi/rt/mobilenet_v2_0.45x/result_rt_split={opt.split}_co_train_alpha=5_use_model=1_lr=0.0002_1_{opt.data_use}/checkpoints/best.pth'
        elif opt.network_choose == 'resnet_18':
            opt.model_pretrained = f'BrainGuided/debug/minor/new_result/final_2181_annot/sig_test_each_roi/rt/resnet_18/result_rt_split={opt.split}_co_train_alpha=5_use_model=1_lr=0.0002_1_{opt.data_use}/checkpoints/best.pth'
        elif opt.network_choose == 'shufflenet_v1':
            opt.model_pretrained = f'BrainGuided/debug/minor/new_result/final_2181_annot/sig_test_each_roi/rt/shufflenet_v1_1.5x/result_rt_split={opt.split}_co_train_alpha=5_use_model=1_lr=0.0002_1_{opt.data_use}/checkpoints/best.pth'
        elif opt.network_choose == 'shufflenet_v2':
            opt.model_pretrained = f'BrainGuided/debug/minor/new_result/final_2181_annot/sig_test_each_roi/rt/shufflenet_v2/result_rt_split={opt.split}_co_train_alpha=5_use_model=1_lr=0.0002_1_{opt.data_use}/checkpoints/best.pth'
        elif opt.network_choose == 'squeezenet':
            opt.model_pretrained = f'BrainGuided/debug/minor/new_result/final_2181_annot/sig_test_each_roi/rt/squeezenet/result_rt_split={opt.split}_co_train_alpha=5_use_model=1_lr=0.0002_1_{opt.data_use}/checkpoints/best.pth'
        
        # ce_cka_Fu
        # if opt.network_choose == 'mobilenet_v1':
        #     opt.model_pretrained = f'BrainGuided/debug/minor/new_result/final_2181/sig_test/rt/mobilenet_v1/result_rt_split={opt.split}_co_train_alpha=5_use_model=1_lr=0.0002_1_{opt.data_use}/checkpoints/best.pth'
        # elif opt.network_choose == 'mobilenet_v2':
        #     opt.model_pretrained = f'BrainGuided/debug/minor/new_result/final_2181/sig_test/rt/mobilenet_v2_0.45x/result_rt_split={opt.split}_co_train_alpha=5_use_model=1_lr=0.0002_1_{opt.data_use}/checkpoints/best.pth'
        # elif opt.network_choose == 'resnet_18':
        #     opt.model_pretrained = f'BrainGuided/debug/minor/new_result/final_2181/sig_test/rt/resnet_18/result_rt_split={opt.split}_co_train_alpha=5_use_model=1_lr=0.0002_1_{opt.data_use}/checkpoints/best.pth'
        # elif opt.network_choose == 'shufflenet_v1':
        #     opt.model_pretrained = f'BrainGuided/debug/minor/new_result/final_2181/sig_test/rt/shufflenet_v1_1.5x/result_rt_split={opt.split}_co_train_alpha=5_use_model=1_lr=0.0002_1_{opt.data_use}/checkpoints/best.pth'
        # elif opt.network_choose == 'shufflenet_v2':
        #     opt.model_pretrained = f'BrainGuided/debug/minor/new_result/final_2181/sig_test/rt/shufflenet_v2/result_rt_split={opt.split}_co_train_alpha=5_use_model=1_lr=0.0002_1_{opt.data_use}/checkpoints/best.pth'
        # elif opt.network_choose == 'squeezenet':
        #     opt.model_pretrained = f'BrainGuided/debug/minor/new_result/final_2181/sig_test/rt/squeezenet/result_rt_split={opt.split}_co_train_alpha=5_use_model=1_lr=0.0002_1_{opt.data_use}/checkpoints/best.pth'
        
        # ce_cka_neurostorm_pretrained
        # if opt.network_choose == 'mobilenet_v1':
        #     opt.model_pretrained = f'BrainGuided/debug/minor/new_result/final_2181/sig_test_neurostorm_feature_pretrain_mae0.5/rt/mobilenet_v1/result_rt_split={opt.split}_co_train_alpha=5_use_model=1_lr=0.0002_1_{opt.data_use}/checkpoints/best.pth'
        # elif opt.network_choose == 'mobilenet_v2':
        #     opt.model_pretrained = f'BrainGuided/debug/minor/new_result/final_2181/sig_test_neurostorm_feature_pretrain_mae0.5/rt/mobilenet_v2_0.45x/result_rt_split={opt.split}_co_train_alpha=5_use_model=1_lr=0.0002_1_{opt.data_use}/checkpoints/best.pth'
        # elif opt.network_choose == 'resnet_18':
        #     opt.model_pretrained = f'BrainGuided/debug/minor/new_result/final_2181/sig_test_neurostorm_feature_pretrain_mae0.5/rt/resnet_18/result_rt_split={opt.split}_co_train_alpha=5_use_model=1_lr=0.0002_1_{opt.data_use}/checkpoints/best.pth'
        # elif opt.network_choose == 'shufflenet_v1':
        #     opt.model_pretrained = f'BrainGuided/debug/minor/new_result/final_2181/sig_test_neurostorm_feature_pretrain_mae0.5/rt/shufflenet_v1_1.5x/result_rt_split={opt.split}_co_train_alpha=5_use_model=1_lr=0.0002_1_{opt.data_use}/checkpoints/best.pth'
        # elif opt.network_choose == 'shufflenet_v2':
        #     opt.model_pretrained = f'BrainGuided/debug/minor/new_result/final_2181/sig_test_neurostorm_feature_pretrain_mae0.5/rt/shufflenet_v2/result_rt_split={opt.split}_co_train_alpha=5_use_model=1_lr=0.0002_1_{opt.data_use}/checkpoints/best.pth'
        # elif opt.network_choose == 'squeezenet':
        #     opt.model_pretrained = f'BrainGuided/debug/minor/new_result/final_2181/sig_test_neurostorm_feature_pretrain_mae0.5/rt/squeezenet/result_rt_split={opt.split}_co_train_alpha=5_use_model=1_lr=0.0002_1_{opt.data_use}/checkpoints/best.pth'

        # ce_cka_lstm
        # if opt.network_choose == 'mobilenet_v1':
        #     opt.model_pretrained = f'BrainGuided/debug/minor/new_result/final_2181/sig_test_lstm_feature/rt/mobilenet_v1/result_rt_split={opt.split}_co_train_alpha=5_use_model=1_lr=0.0002_1_{opt.data_use}/checkpoints/best.pth'
        # elif opt.network_choose == 'mobilenet_v2':
        #     opt.model_pretrained = f'BrainGuided/debug/minor/new_result/final_2181/sig_test_lstm_feature/rt/mobilenet_v2_0.45x/result_rt_split={opt.split}_co_train_alpha=5_use_model=1_lr=0.0002_1_{opt.data_use}/checkpoints/best.pth'
        # elif opt.network_choose == 'resnet_18':
        #     opt.model_pretrained = f'BrainGuided/debug/minor/new_result/final_2181/sig_test_lstm_feature/rt/resnet_18/result_rt_split={opt.split}_co_train_alpha=5_use_model=1_lr=0.0002_1_{opt.data_use}/checkpoints/best.pth'
        # elif opt.network_choose == 'shufflenet_v1':
        #     opt.model_pretrained = f'BrainGuided/debug/minor/new_result/final_2181/sig_test_lstm_feature/rt/shufflenet_v1_1.5x/result_rt_split={opt.split}_co_train_alpha=5_use_model=1_lr=0.0002_1_{opt.data_use}/checkpoints/best.pth'
        # elif opt.network_choose == 'shufflenet_v2':
        #     opt.model_pretrained = f'BrainGuided/debug/minor/new_result/final_2181/sig_test_lstm_feature/rt/shufflenet_v2/result_rt_split={opt.split}_co_train_alpha=5_use_model=1_lr=0.0002_1_{opt.data_use}/checkpoints/best.pth'
        # elif opt.network_choose == 'squeezenet':
        #     opt.model_pretrained = f'BrainGuided/debug/minor/new_result/final_2181/sig_test_lstm_feature/rt/squeezenet/result_rt_split={opt.split}_co_train_alpha=5_use_model=1_lr=0.0002_1_{opt.data_use}/checkpoints/best.pth'

        # ce_mse_lstm
        # if opt.network_choose == 'mobilenet_v1':
        #     opt.model_pretrained = f'BrainGuided/debug/minor/new_result/final_2181/sig_test_add_project_layer_mse/rt/mobilenet_v1/result_rt_split={opt.split}_co_train_alpha=5_use_model=1_lr=0.0002_1_{opt.data_use}/checkpoints/best.pth'
        # elif opt.network_choose == 'mobilenet_v2':
        #     opt.model_pretrained = f'BrainGuided/debug/minor/new_result/final_2181/sig_test_add_project_layer_mse/rt/mobilenet_v2_0.45x/result_rt_split={opt.split}_co_train_alpha=5_use_model=1_lr=0.0002_1_{opt.data_use}/checkpoints/best.pth'
        # elif opt.network_choose == 'resnet_18':
        #     opt.model_pretrained = f'BrainGuided/debug/minor/new_result/final_2181/sig_test_add_project_layer_mse/rt/resnet_18/result_rt_split={opt.split}_co_train_alpha=5_use_model=1_lr=0.0002_1_{opt.data_use}/checkpoints/best.pth'
        # elif opt.network_choose == 'shufflenet_v1':
        #     opt.model_pretrained = f'BrainGuided/debug/minor/new_result/final_2181/sig_test_add_project_layer_mse/rt/shufflenet_v1_1.5x/result_rt_split={opt.split}_co_train_alpha=5_use_model=1_lr=0.0002_1_{opt.data_use}/checkpoints/best.pth'
        # elif opt.network_choose == 'shufflenet_v2':
        #     opt.model_pretrained = f'BrainGuided/debug/minor/new_result/final_2181/sig_test_add_project_layer_mse/rt/shufflenet_v2/result_rt_split={opt.split}_co_train_alpha=5_use_model=1_lr=0.0002_1_{opt.data_use}/checkpoints/best.pth'
        # elif opt.network_choose == 'squeezenet':
        #     opt.model_pretrained = f'BrainGuided/debug/minor/new_result/final_2181/sig_test_add_project_layer_mse/rt/squeezenet/result_rt_split={opt.split}_co_train_alpha=5_use_model=1_lr=0.0002_1_{opt.data_use}/checkpoints/best.pth'

        # ce_mse_neurostorm_pretrained
        # if opt.network_choose == 'mobilenet_v1':
        #     opt.model_pretrained = f'BrainGuided/debug/minor/new_result/final_2181/sig_test_add_project_layer_mse_neurostorm_pretrain_mae0.5/rt/mobilenet_v1/result_rt_split={opt.split}_co_train_alpha=5_use_model=1_lr=0.0002_1_{opt.data_use}/checkpoints/best.pth'
        # elif opt.network_choose == 'mobilenet_v2':
        #     opt.model_pretrained = f'BrainGuided/debug/minor/new_result/final_2181/sig_test_add_project_layer_mse_neurostorm_pretrain_mae0.5/rt/mobilenet_v2_0.45x/result_rt_split={opt.split}_co_train_alpha=5_use_model=1_lr=0.0002_1_{opt.data_use}/checkpoints/best.pth'
        # elif opt.network_choose == 'resnet_18':
        #     opt.model_pretrained = f'BrainGuided/debug/minor/new_result/final_2181/sig_test_add_project_layer_mse_neurostorm_pretrain_mae0.5/rt/resnet_18/result_rt_split={opt.split}_co_train_alpha=5_use_model=1_lr=0.0002_1_{opt.data_use}/checkpoints/best.pth'
        # elif opt.network_choose == 'shufflenet_v1':
        #     opt.model_pretrained = f'BrainGuided/debug/minor/new_result/final_2181/sig_test_add_project_layer_mse_neurostorm_pretrain_mae0.5/rt/shufflenet_v1_1.5x/result_rt_split={opt.split}_co_train_alpha=5_use_model=1_lr=0.0002_1_{opt.data_use}/checkpoints/best.pth'
        # elif opt.network_choose == 'shufflenet_v2':
        #     opt.model_pretrained = f'BrainGuided/debug/minor/new_result/final_2181/sig_test_add_project_layer_mse_neurostorm_pretrain_mae0.5/rt/shufflenet_v2/result_rt_split={opt.split}_co_train_alpha=5_use_model=1_lr=0.0002_1_{opt.data_use}/checkpoints/best.pth'
        # elif opt.network_choose == 'squeezenet':
        #     opt.model_pretrained = f'BrainGuided/debug/minor/new_result/final_2181/sig_test_add_project_layer_mse_neurostorm_pretrain_mae0.5/rt/squeezenet/result_rt_split={opt.split}_co_train_alpha=5_use_model=1_lr=0.0002_1_{opt.data_use}/checkpoints/best.pth'

    else:
        if opt.network_choose == 'mobilenet_v1':
            opt.model_pretrained = 'pretrained-models/kinetics_mobilenet_2.0x_RGB_16_best.pth'
        elif opt.network_choose == 'mobilenet_v2':
            opt.model_pretrained = 'pretrained-models/kinetics_mobilenetv2_0.45x_RGB_16_best.pth'
        elif opt.network_choose == 'resnet_18':
            opt.model_pretrained = 'pretrained-models/resnet-18-kinetics.pth'
        elif opt.network_choose == 'shufflenet_v1':
            opt.model_pretrained = 'pretrained-models/kinetics_shufflenet_1.5x_G3_RGB_16_best.pth'
        elif opt.network_choose == 'shufflenet_v2':
            opt.model_pretrained = 'pretrained-models/kinetics_shufflenetv2_2.0x_RGB_16_best.pth'
        elif opt.network_choose == 'squeezenet':
            opt.model_pretrained = 'pretrained-models/kinetics_squeezenet_RGB_16_best.pth'
    print('Experiment information:', 'video_path=', opt.video_path, 'annotation_path=', opt.annotation_path,
          'n_classes=', opt.n_classes, 'dataset_choose=', opt.dataset_choose,'n_epoch=',opt.n_epochs, 'split=',opt.split)
    print("length of neural_response:",len(neural_response), "length of neural_response_valid:",len(neural_response_valid))
    
    local2global_path(opt)
    if opt.get_layer_contribution:
        model = torch.load(opt.model_pretrained)
        model = model.cuda()
        total_params = 0
        for parameter in model.parameters():
            if not parameter.requires_grad: continue
            param = parameter.numel()
            total_params += param
        parameters = model.parameters()
    else:
        model, parameters,total_params = generate_model(opt)
    print(f"Total Trainable Params: {total_params}")
    criterion = nn.MSELoss()
    criterion = criterion.cuda()
    optimizer = get_optim(opt, parameters)

    # validation
    spatial_transform = get_spatial_transform(opt, 'test')
    temporal_transform = TSN(seq_len=opt.seq_len, snippet_duration=opt.snippet_duration, center=False)
    neural_train_data = get_neural_set(opt, spatial_transform, temporal_transform,neural_response)
    neural_train_loader = get_neural_loader(opt, neural_train_data, shuffle=True)
    neural_valid_data = NeuralValidDataset(opt,opt.neural_video_path,neural_response_valid,opt.fps,spatial_transform,temporal_transform)
    if opt.contribution_method == 'rdm_corr':
        neural_valid_loader = get_neural_loader(opt, neural_valid_data, shuffle=True)
    else:
        neural_valid_loader = get_neural_loader(opt, neural_valid_data, shuffle=False)
    del neural_train_data, neural_valid_data

    if opt.get_layer_contribution:
        if opt.contribution_method == 'ridge':
            # run_model_get_contribution_v3(opt, neural_train_loader, neural_valid_loader, model)
            run_model_get_contribution_per_layer(opt, neural_train_loader, neural_valid_loader, model)
        elif opt.contribution_method == 'rdm_corr':
            # train
            # spatial_transform = get_spatial_transform(opt, 'train')
            # temporal_transform = TSN(seq_len=opt.seq_len, snippet_duration=opt.snippet_duration, center=False)
            # target_transform = ClassLabel()
            # training_data = get_training_set(opt, spatial_transform, temporal_transform, target_transform)
            # train_loader = get_data_loader(opt, training_data, shuffle=True)
            # valid
            spatial_transform = get_spatial_transform(opt, 'test')
            temporal_transform = TSN(seq_len=opt.seq_len, snippet_duration=opt.snippet_duration, center=False)
            target_transform = ClassLabel()
            validation_data = get_validation_set(opt, spatial_transform, temporal_transform, target_transform)
            val_loader = get_data_loader(opt, validation_data, shuffle=False)
            run_model_get_contribution(opt, val_loader,neural_valid_loader, model)
        return

    result = []
    patience = 50
    min_loss = float('inf')
    for i in range(1, opt.n_epochs + 1):
        mse_loss, contribution = train_epoch_contribution(i, neural_train_loader, model, criterion, optimizer, opt)
        mse_loss_valid, contribution_valid = val_epoch_contribution(i, neural_valid_loader, model, criterion, optimizer, opt)
        row_data = {
            'epoch': i,
            'mse_loss': mse_loss,
            'mse_loss_valid': mse_loss_valid
        }
        # 3. Merge the contribution dictionary into this row
        # This adds key1, key2, etc., to the same dictionary
        row_data.update(contribution_valid)
        
        # 4. Append to history
        result.append(row_data)

        if mse_loss_valid < min_loss:
            min_loss = mse_loss_valid
            patience = 50
            save_file_path = os.path.join(opt.ckpt_path, 'best.pth')
            torch.save(model, save_file_path)
            print('save model at epoch {}'.format(i))
        else:
            patience -= 1
            if patience == 0:
                break

    # 5. Convert to DataFrame and Save
    df = pd.DataFrame(result)
    cols = ['epoch', 'mse_loss', 'mse_loss_valid'] + [c for c in df.columns if c not in ['epoch', 'mse_loss', 'mse_loss_valid']]
    df = df[cols]
    df.to_csv(os.path.join(opt.result_path,'result.csv'), index=False)

def main_contribution_each_roi(alpha, neural_response_evc, neural_response_tos,
                                        neural_response_ppa, neural_response_rsc, neural_response_pfc,
                                        sig_test_run, network, split, roi):
    """
    Per-ROI counterpart to main_contribution, mirroring how main_sig_each_roi_add_pfc
    sets up training. neural_response_pfc may be [] / None when pfc data isn't available;
    the pfc loader is then simply not built and run_model_get_contribution_each_roi_add_pfc
    skips it.
    """
    opt = parse_opts()
    opt.alpha = 1
    opt.sig_test_run = sig_test_run
    opt.network_choose = network
    opt.split = split
    opt.roi = roi
 
    if opt.dataset_choose == 'rt':
        if opt.task == 'design':
            opt.video_path = 'RT--imgs'
            opt.video_raw_path = 'RT--raw'
            opt.annotation_path = 'video_id_rt.csv'
            opt.n_classes = 4
        elif opt.task == 'annot' or opt.task == 'annot-reg':
            opt.video_path = 'RT--imgs'
            opt.video_raw_path = 'RT--raw'
            opt.annotation_path = 'video_id_rt_annot.csv'
            opt.n_classes = 15
 
    # NOTE: same checkpoint-path resolution as main_contribution's
    # "ce_cka_each_roi" branch -- copy the rest of that if/elif chain in if you use
    # other network choices / other pretrained variants.
    if opt.network_choose == 'mobilenet_v1':
        opt.model_pretrained = f'BrainGuided/debug/minor/new_result/{opt.task}/sig_test_each_roi_neurostorm_win4_add_pfc/rt/mobilenet_v1/result_rt_split={opt.split}_co_train_alpha=5_use_model=1_lr=0.0002_99_{opt.data_use}/checkpoints/best.pth'
    elif opt.network_choose == 'mobilenet_v2':
        opt.model_pretrained = f'BrainGuided/debug/minor/new_result/{opt.task}/sig_test_each_roi_neurostorm_win4_add_pfc/rt/mobilenet_v2_0.45x/result_rt_split={opt.split}_co_train_alpha=5_use_model=1_lr=0.0002_99_{opt.data_use}/checkpoints/best.pth'
    elif opt.network_choose == 'resnet_18':
        opt.model_pretrained = f'BrainGuided/debug/minor/new_result/{opt.task}/sig_test_each_roi_neurostorm_win4_add_pfc/rt/resnet_18/result_rt_split={opt.split}_co_train_alpha=5_use_model=1_lr=0.0002_99_{opt.data_use}/checkpoints/best.pth'
    elif opt.network_choose == 'shufflenet_v1':
        opt.model_pretrained = f'BrainGuided/debug/minor/new_result/{opt.task}/sig_test_each_roi_neurostorm_win4_add_pfc/rt/shufflenet_v1_1.5x/result_rt_split={opt.split}_co_train_alpha=5_use_model=1_lr=0.0002_99_{opt.data_use}/checkpoints/best.pth'
    elif opt.network_choose == 'shufflenet_v2':
        opt.model_pretrained = f'BrainGuided/debug/minor/new_result/{opt.task}/sig_test_each_roi_neurostorm_win4_add_pfc/rt/shufflenet_v2/result_rt_split={opt.split}_co_train_alpha=5_use_model=1_lr=0.0002_99_{opt.data_use}/checkpoints/best.pth'
    elif opt.network_choose == 'squeezenet':
        opt.model_pretrained = f'BrainGuided/debug/minor/new_result/{opt.task}/sig_test_each_roi_neurostorm_win4_add_pfc/rt/squeezenet/result_rt_split={opt.split}_co_train_alpha=5_use_model=1_lr=0.0002_99_{opt.data_use}/checkpoints/best.pth'
    elif opt.network_choose == 'vit_3d':
        opt.model_pretrained = f'BrainGuided/debug/minor/new_result/{opt.task}/sig_test_each_roi_neurostorm_win4_add_pfc/rt/vit_3d/result_rt_split={opt.split}_co_train_alpha=5_use_model=1_lr=0.0002_99_{opt.data_use}/checkpoints/best.pth'
    elif opt.network_choose == 'video_swin':
        opt.model_pretrained = f'BrainGuided/debug/minor/new_result/{opt.task}/sig_test_each_roi_neurostorm_win4_add_pfc/rt/video_swin/result_rt_split={opt.split}_co_train_alpha=5_use_model=1_lr=0.0002_99_{opt.data_use}/checkpoints/best.pth'

    print('Experiment information:', 'video_path=', opt.video_path, 'annotation_path=', opt.annotation_path,
          'n_classes=', opt.n_classes, 'dataset_choose=', opt.dataset_choose, 'n_epoch=', opt.n_epochs, 'split=', opt.split)
 
    local2global_path(opt)
 
    model = torch.load(opt.model_pretrained)
    model = model.cuda()
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total Trainable Params: {total_params}")
 
    spatial_transform = get_spatial_transform(opt, 'test')
    temporal_transform = TSN(seq_len=opt.seq_len, snippet_duration=opt.snippet_duration, center=False)
 
    def build_loader(neural_response_list):
        if not neural_response_list:
            return None
        neural_valid_data = NeuralValidDataset(
            opt, opt.neural_video_path, neural_response_list, opt.fps,
            spatial_transform, temporal_transform
        )
        # opt.batch_size_neural = len(neural_valid_data)
        return get_neural_loader(opt, neural_valid_data, shuffle=False)  # matches rdm_corr branch in main_contribution
 
    neural_loader_evc = build_loader(neural_response_evc)
    neural_loader_tos = build_loader(neural_response_tos)
    neural_loader_ppa = build_loader(neural_response_ppa)
    neural_loader_rsc = build_loader(neural_response_rsc)
    neural_loader_pfc = build_loader(neural_response_pfc)  # None-safe: OK if pfc missing
 
    target_transform = ClassLabel()
    validation_data = get_validation_set(opt, spatial_transform, temporal_transform, target_transform)
    # opt.batch_size = len(validation_data)
    val_loader = get_data_loader(opt, validation_data, shuffle=False)
 
    run_model_get_contribution_each_roi(
        opt, val_loader,
        neural_loader_evc, neural_loader_tos, neural_loader_ppa, neural_loader_rsc, neural_loader_pfc,
        model
    )

def main_vanilla(sig_test_run,network,split):
    opt = parse_opts()
    opt.sig_test_run = sig_test_run
    opt.network_choose = network
    opt.split = split
    print('!!!co_train:',opt.co_train)
    if opt.dataset_choose=='ve8':
        opt.video_path = 'VideoEmotion8--imgs'
        opt.video_raw_path = 'VideoEmotion8--raw'
        opt.annotation_path = 'video_id_ve8.csv'
        opt.n_classes = 8
    elif opt.dataset_choose=='ek6':
        opt.video_path = 'EK6--imgs'
        opt.video_raw_path = 'EK6--raw'
        opt.annotation_path = 'video_id_ek6.csv'
        opt.n_classes = 6
    elif opt.dataset_choose=='rt':
        if opt.task == 'design':
            opt.video_path = 'RT--imgs'
            opt.video_raw_path = 'RT--raw'
            opt.annotation_path = 'video_id_rt.csv'
            opt.n_classes = 4
        elif opt.task == 'space':
            opt.video_path = 'RT--imgs'
            opt.video_raw_path = 'RT--raw'
            opt.annotation_path = 'video_id_rt.csv'
            opt.n_classes = 8
        elif opt.task == 'annot' or opt.task == 'annot-reg':
            opt.video_path = 'RT--imgs'
            opt.video_raw_path = 'RT--raw'
            opt.annotation_path = 'video_id_rt_annot.csv'
            opt.n_classes = 15
    if opt.network_choose == 'mobilenet_v1':
        opt.model_pretrained = 'pretrained-models/kinetics_mobilenet_2.0x_RGB_16_best.pth'
    elif opt.network_choose == 'mobilenet_v2':
        opt.model_pretrained = 'pretrained-models/kinetics_mobilenetv2_0.45x_RGB_16_best.pth'
    elif opt.network_choose == 'resnet_18':
        opt.model_pretrained = 'pretrained-models/resnet-18-kinetics.pth'
    elif opt.network_choose == 'shufflenet_v1':
        opt.model_pretrained = 'pretrained-models/kinetics_shufflenet_1.5x_G3_RGB_16_best.pth'
    elif opt.network_choose == 'shufflenet_v2':
        opt.model_pretrained = 'pretrained-models/kinetics_shufflenetv2_2.0x_RGB_16_best.pth'
    elif opt.network_choose == 'squeezenet':
        opt.model_pretrained = 'pretrained-models/kinetics_squeezenet_RGB_16_best.pth'
    elif opt.network_choose == 'alexnet_3d':
        opt.model_pretrained = None
    elif opt.network_choose == 'vit_3d':
        opt.model_pretrained = None
    elif opt.network_choose == 'video_swin':
        opt.model_pretrained = None  # None -> torchvision auto-downloads real Kinetics-400 weights
    print('Experiment information:', 'video_path=', opt.video_path, 'annotation_path=', opt.annotation_path,
          'n_classes=', opt.n_classes, 'dataset_choose=', opt.dataset_choose)
    local2global_path(opt)
    model, parameters,total_params = generate_model(opt)
    print(f"Total Trainable Params: {total_params}")
    sys.exit(0)
    criterion = get_loss(opt)
    criterion = criterion.cuda()
    optimizer = get_optim(opt, parameters)


    # train
    spatial_transform = get_spatial_transform(opt, 'train')
    temporal_transform = TSN(seq_len=opt.seq_len, snippet_duration=opt.snippet_duration, center=False)
    target_transform = ClassLabel()
    training_data = get_training_set(opt, spatial_transform, temporal_transform, target_transform)
    train_loader = get_data_loader(opt, training_data, shuffle=True)

    # validation
    spatial_transform = get_spatial_transform(opt, 'test')
    temporal_transform = TSN(seq_len=opt.seq_len, snippet_duration=opt.snippet_duration, center=False)
    target_transform = ClassLabel()
    validation_data = get_validation_set(opt, spatial_transform, temporal_transform, target_transform)
    val_loader = get_data_loader(opt, validation_data, shuffle=False)

    result = np.zeros((opt.n_epochs,2))
    if opt.single_annot_class == True:
        class_accuracy = np.zeros((opt.n_epochs, 1))
    else:
        class_accuracy = np.zeros((opt.n_epochs, opt.n_classes))
        if opt.task == 'annot':
            class_auc = np.zeros((opt.n_epochs,opt.n_classes))
            class_youden = np.zeros((opt.n_epochs,opt.n_classes))
            class_acc_at_bestthres = np.zeros((opt.n_epochs,opt.n_classes))
    max_acc = -np.inf
    min_loss = np.inf
    if opt.task == 'annot' or opt.task == 'annot-reg':
        patience_cnt = 10
    else:
        patience_cnt = 10
    patience = patience_cnt
    for i in range(1, opt.n_epochs + 1):
        train_loss = train_epoch(i, train_loader, model, criterion, optimizer, opt, training_data.class_names)
        if opt.task == 'annot':
            ep,acc,ac, loss, auc, youden, acc_at_bestthres = val_epoch_class(i, val_loader, model, criterion, opt, optimizer)
            # class_auc[i-1] = auc
            # class_youden[i-1] = youden
            # class_acc_at_bestthres[i-1] = acc_at_bestthres
            # np.savetxt(os.path.join(opt.result_path,'class_auc.csv'), class_auc,delimiter = ',')
            # np.savetxt(os.path.join(opt.result_path,'class_youden.csv'), class_youden,delimiter = ',')
            # np.savetxt(os.path.join(opt.result_path,'class_acc_at_bestthres.csv'), class_acc_at_bestthres,delimiter = ',')
        else:
            ep,acc,ac, loss = val_epoch_class(i, val_loader, model, criterion, opt, optimizer)
        result[i-1,0] = ep
        result[i-1,1] = acc
        class_accuracy[i-1] = ac
        acc_result = np.concatenate((result,class_accuracy),axis=1)
        np.savetxt(os.path.join(opt.result_path,'acc_result.csv'), acc_result,delimiter = ',')
        np.save(os.path.join(opt.result_path,'result.npy'),result)

        # save model
        # if opt.task == 'annot':
        #     acc = np.mean(auc)
        if acc > max_acc:
            max_acc = acc
            save_file_path = os.path.join(opt.ckpt_path, 'best.pth'.format(i))
            # states = {
            #     'epoch': i,
            #     'state_dict': copy.deepcopy(model.state_dict()),
            #     'optimizer': optimizer.state_dict(),
            # }
            # torch.save(states, save_file_path)
            torch.save(model, save_file_path)
            print("saved in result_path", opt.result_path)
            patience = patience_cnt

        else:
            patience = patience - 1
            if patience == 0:
                break

if __name__ == "__main__":

    gc.collect()
    torch.cuda.empty_cache()
    # setup_seed(42)
    
    opt = parse_opts()
    if opt.co_train == True:
        
        rois = ['ALL']
        for roi in rois:
            opt.roi = roi
            split_all = [1,2,3,4]
            for split in split_all:
                opt.split = split
                neural_response =[]
                neural_response_valid = []
                if opt.align_each_roi:
                    neural_response_evc = []
                    neural_response_tos = []
                    neural_response_ppa = []
                    neural_response_rsc = []
                    neural_response_pfc = []
                print(opt.use_model)
                if opt.use_model:
                    print('use_model')
                    # for Subject in range(1,6):
                    # for Subject in range(1,2):
                    print(opt.data_use)
                    if opt.use_lstm:
                        neural_response.append(np.load('Neural_data/emotion_encoding_results/'+opt.data_use+f'/voxel_select_remain_time_{opt.split}.npy'))
                    elif opt.align_each_roi:
                        if opt.get_layer_contribution:
                            if opt.task == 'annot' or opt.task == 'annot-reg':
                                neural_response_evc.append(np.load('Neural_data/emotion_encoding_results/'+opt.data_use+f'/neurostorm_pretrained_win4_annot_valid_EVC_{opt.split}.npy'))
                                neural_response_tos.append(np.load('Neural_data/emotion_encoding_results/'+opt.data_use+f'/neurostorm_pretrained_win4_annot_valid_TOS_{opt.split}.npy'))
                                neural_response_ppa.append(np.load('Neural_data/emotion_encoding_results/'+opt.data_use+f'/neurostorm_pretrained_win4_annot_valid_PPA_{opt.split}.npy'))
                                neural_response_rsc.append(np.load('Neural_data/emotion_encoding_results/'+opt.data_use+f'/neurostorm_pretrained_win4_annot_valid_RSC_{opt.split}.npy'))
                                if opt.add_pfc == True:
                                    neural_response_pfc.append(np.load('Neural_data/emotion_encoding_results/'+opt.data_use+f'/neurostorm_pretrained_win4_annot_valid_PFC_{opt.split}.npy'))
                            else:
                                neural_response_evc.append(np.load('Neural_data/emotion_encoding_results/'+opt.data_use+f'/neurostorm_pretrained_win4_valid_EVC_{opt.split}.npy'))
                                neural_response_tos.append(np.load('Neural_data/emotion_encoding_results/'+opt.data_use+f'/neurostorm_pretrained_win4_valid_TOS_{opt.split}.npy'))
                                neural_response_ppa.append(np.load('Neural_data/emotion_encoding_results/'+opt.data_use+f'/neurostorm_pretrained_win4_valid_PPA_{opt.split}.npy'))
                                neural_response_rsc.append(np.load('Neural_data/emotion_encoding_results/'+opt.data_use+f'/neurostorm_pretrained_win4_valid_RSC_{opt.split}.npy'))
                                if opt.add_pfc == True:
                                    neural_response_pfc.append(np.load('Neural_data/emotion_encoding_results/'+opt.data_use+f'/neurostorm_pretrained_win4_valid_PFC_{opt.split}.npy'))
                        else:
                            if opt.task == 'annot' or opt.task == 'annot-reg':
                                neural_response_evc.append(np.load('Neural_data/emotion_encoding_results/'+opt.data_use+f'/neurostorm_pretrained_win4_annot_EVC_{opt.split}.npy'))
                                neural_response_tos.append(np.load('Neural_data/emotion_encoding_results/'+opt.data_use+f'/neurostorm_pretrained_win4_annot_TOS_{opt.split}.npy'))
                                neural_response_ppa.append(np.load('Neural_data/emotion_encoding_results/'+opt.data_use+f'/neurostorm_pretrained_win4_annot_PPA_{opt.split}.npy'))
                                neural_response_rsc.append(np.load('Neural_data/emotion_encoding_results/'+opt.data_use+f'/neurostorm_pretrained_win4_annot_RSC_{opt.split}.npy'))
                                if opt.add_pfc == True:
                                    neural_response_pfc.append(np.load('Neural_data/emotion_encoding_results/'+opt.data_use+f'/neurostorm_pretrained_win4_annot_PFC_{opt.split}.npy'))
                            else:
                                neural_response_evc.append(np.load('Neural_data/emotion_encoding_results/'+opt.data_use+f'/neurostorm_pretrained_win4_EVC_{opt.split}.npy'))
                                neural_response_tos.append(np.load('Neural_data/emotion_encoding_results/'+opt.data_use+f'/neurostorm_pretrained_win4_TOS_{opt.split}.npy'))
                                neural_response_ppa.append(np.load('Neural_data/emotion_encoding_results/'+opt.data_use+f'/neurostorm_pretrained_win4_PPA_{opt.split}.npy'))
                                neural_response_rsc.append(np.load('Neural_data/emotion_encoding_results/'+opt.data_use+f'/neurostorm_pretrained_win4_RSC_{opt.split}.npy'))
                                if opt.add_pfc == True:
                                    neural_response_pfc.append(np.load('Neural_data/emotion_encoding_results/'+opt.data_use+f'/neurostorm_pretrained_win4_PFC_{opt.split}.npy'))
                    
                    elif opt.train_only_layer_contribution or opt.get_layer_contribution:
                        if opt.task == 'annot':
                            neural_response_valid.append(np.load('Neural_data/emotion_encoding_results/'+opt.data_use+f'/voxel_select_new_annot_valid_{opt.roi}_{opt.split}.npy'))
                        else:
                            neural_response_valid.append(np.load('Neural_data/emotion_encoding_results/'+opt.data_use+f'/voxel_select_new_valid_{opt.roi}_{opt.split}.npy'))
                        # neural_response_valid.append(np.load('Neural_data/emotion_encoding_results/'+opt.data_use+f'/neurostorm_embeddings_pretrain_mae0.5_{opt.roi}_valid_{opt.split}.npy'))

                    else:
                        if opt.task == 'annot' or opt.task == 'annot-reg':
                            # neural_response.append(np.load('Neural_data/emotion_encoding_results/'+opt.data_use+f'/neurostorm_pretrained_win4_annot_{opt.split}.npy'))
                            neural_response.append(np.load('Neural_data/emotion_encoding_results/'+opt.data_use+f'/voxel_select_new_annot_{opt.roi}_{opt.split}.npy'))
                        else:
                            neural_response.append(np.load('Neural_data/emotion_encoding_results/'+opt.data_use+f'/voxel_select_new_{opt.roi}_{opt.split}.npy'))
                        # neural_response.append(np.load('Neural_data/emotion_encoding_results/'+opt.data_use+f'/voxel_select_lstm_feature_{opt.split}.npy')) # _lstm_feature_{opt.split}
                            # neural_response.append(np.load('Neural_data/emotion_encoding_results/'+opt.data_use+f'/neurostorm_pretrained_win4_{opt.split}.npy')) # neurostorm_embeddings_cls_{opt.split}
                            # neural_response.append(np.load('Neural_data/emotion_encoding_results/'+opt.data_use+f'/voxel_select_new_annot_{opt.roi}_{opt.split}.npy')) # neurostorm_embeddings_cls_{opt.split}
                        print(f"!!!!!!!!!!!!!!!!!!!!!!split {opt.split} neural_response shape {neural_response[-1].shape}")

                    # check if input has nan
                    if opt.align_each_roi:
                        if np.isnan(neural_response_evc[-1]).any():
                            print('nan appears in neural_response_evc')
                            neural_response_evc[-1][np.isnan(neural_response_evc[-1])] = 0
                        if np.isnan(neural_response_tos[-1]).any():
                            print('nan appears in neural_response_tos')
                            neural_response_tos[-1][np.isnan(neural_response_tos[-1])] = 0
                        if np.isnan(neural_response_ppa[-1]).any():
                            print('nan appears in neural_response_ppa')
                            neural_response_ppa[-1][np.isnan(neural_response_ppa[-1])] = 0
                        if np.isnan(neural_response_rsc[-1]).any():
                            print('nan appears in neural_response_rsc')
                            neural_response_rsc[-1][np.isnan(neural_response_rsc[-1])] = 0
                        if opt.add_pfc == True:
                            if np.isnan(neural_response_pfc[-1]).any():
                                print('nan appears in neural_response_pfc')
                                neural_response_pfc[-1][np.isnan(neural_response_pfc[-1])] = 0
                    else:
                        if np.isnan(neural_response[-1]).any():
                            print('nan appears in neural_response')
                            neural_response[-1][np.isnan(neural_response[-1])] = 0

                        # neural_response.append(np.load('Neural_data/emotion_encoding_results/Subject'+str(Subject)+'/voxel_select.npy'))
                # print('neural_finish')

                # alpha_all = [1]
                alpha_all = [5]
                
                # network_all = ['resnet_18', 'mobilenet_v1', 'shufflenet_v1', 'squeezenet', 'vit_3d', 'video_swin']
                network_all = ['shufflenet_v1']
                for network in network_all:
                    for alpha in alpha_all:
                        for sig_test_run in range(99,100):
                            # print(f"split {opt.split} neural_response shape {neural_response[-1].shape}")
                            if opt.train_only_layer_contribution or opt.get_layer_contribution:
                                if opt.align_each_roi:
                                    print(f"split {opt.split} neural_response_evc shape {neural_response_evc[-1].shape}")
                                    print(f"split {opt.split} neural_response_tos shape {neural_response_tos[-1].shape}")
                                    print(f"split {opt.split} neural_response_ppa shape {neural_response_ppa[-1].shape}")
                                    print(f"split {opt.split} neural_response_rsc shape {neural_response_rsc[-1].shape}")
                                    if opt.add_pfc == True:
                                        print(f"split {opt.split} neural_response_pfc shape {neural_response_pfc[-1].shape}")
                                    main_contribution_each_roi(alpha, neural_response_evc, neural_response_tos, neural_response_ppa, neural_response_rsc, neural_response_pfc, sig_test_run, network, split, opt.roi)
                                else:
                                    main_contribution(alpha,neural_response,neural_response_valid,sig_test_run,network,split, opt.roi)
                            elif opt.align_each_roi:
                                print(f"split {opt.split} neural_response_evc shape {neural_response_evc[-1].shape}")
                                print(f"split {opt.split} neural_response_tos shape {neural_response_tos[-1].shape}")
                                print(f"split {opt.split} neural_response_ppa shape {neural_response_ppa[-1].shape}")
                                print(f"split {opt.split} neural_response_rsc shape {neural_response_rsc[-1].shape}")
                                if opt.add_pfc == True:
                                    print(f"split {opt.split} neural_response_pfc shape {neural_response_pfc[-1].shape}")
                                    main_sig_each_roi_add_pfc(alpha,neural_response_evc,neural_response_tos,neural_response_ppa,neural_response_rsc,neural_response_pfc,sig_test_run,network,split)
                                else:
                                    main_sig_each_roi(alpha,neural_response_evc,neural_response_tos,neural_response_ppa,neural_response_rsc,sig_test_run,network,split)
                            else:
                                main_sig(alpha,neural_response,sig_test_run,network,split, opt.roi)
                            torch.cuda.empty_cache()

    else:
        # network_all = ['resnet_18', 'mobilenet_v1'] #['vaa']
        network_all = ['video_swin'] #['vaa']
        split_all = [1,2,3,4]
        for network in network_all:
            for split in split_all:
                for sig_test_run in range(0,7):
                    main_vanilla(sig_test_run+1,network,split)
                    torch.cuda.empty_cache()

    import smtplib
    smtp=smtplib.SMTP('smtp.gmail.com', 587)
    smtp.ehlo()
    smtp.starttls()
    smtp.login('qianhuisu@gmail.com','kjza aiie bbkk yiqp')
    from_addr='qianhuisu@gmail.com'
    to_addr="qianhuisu@gmail.com"
    msg=f"Subject:Gmail sent by Python scripts\nComplete {opt.data_use}"
    status=smtp.sendmail(from_addr, to_addr, msg)#加密文件，避免私密信息被截取
    if status=={}:
        print("郵件傳送成功!")
    else:
        print("郵件傳送失敗!")
    smtp.quit()

"""
python main.py --expr_name demo
"""