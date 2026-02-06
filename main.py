from opts import parse_opts
import random
from core.model import generate_model
from core.loss import get_loss
from core.optimizer import get_optim
from core.utils import local2global_path, get_spatial_transform
from core.dataset import get_training_set, get_validation_set, get_test_set, get_data_loader,get_neural_set,get_neural_loader
from transforms.temporal import TSN
from transforms.target import ClassLabel
from train import train_epoch,co_train_epoch, co_train_epoch_lstm, train_epoch_contribution
from validation import val_epoch_class, val_epoch_contribution
import numpy as np
import os
import torch
from torch.optim import Adam
from models.lstm import fMRI_LSTM
import copy
from core.utils import run_model_get_contribution

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

    if opt.train_from_checkpoint:
        if opt.network_choose == 'mobilenet_v1':
            opt.model_pretrained = 'BrainGuided/debug/minor/new_result/final_2181/sig_test_lstm_feature/rt/mobilenet_v1/result_rt_split=1_co_train_alpha=5_use_model=1_lr=0.0002_2_744/checkpoints/best.pth'
        elif opt.network_choose == 'mobilenet_v2':
            opt.model_pretrained = 'BrainGuided/debug/minor/new_result/final_2181/sig_test_lstm_feature/rt/mobilenet_v2_0.45x/result_rt_split=1_co_train_alpha=5_use_model=1_lr=0.0002_2/checkpoints/best.pth'
        elif opt.network_choose == 'resnet_18':
            opt.model_pretrained = 'BrainGuided/debug/minor/new_result/final_2181/sig_test_lstm_feature/rt/resnet_18/result_rt_split=1_co_train_alpha=5_use_model=1_lr=0.0002_2_744/checkpoints/best.pth'
        elif opt.network_choose == 'shufflenet_v1':
            opt.model_pretrained = 'BrainGuided/debug/minor/new_result/final_2181/sig_test_lstm_feature/rt/shufflenet_v1_1.5x/result_rt_split=1_co_train_alpha=5_use_model=1_lr=0.0002_2_744/checkpoints/best.pth'
        elif opt.network_choose == 'shufflenet_v2':
            opt.model_pretrained = 'BrainGuided/debug/minor/new_result/final_2181/sig_test_lstm_feature/rt/shufflenet_v2/result_rt_split=1_co_train_alpha=5_use_model=1_lr=0.0002_2_744/checkpoints/best.pth'
        elif opt.network_choose == 'squeezenet':
            opt.model_pretrained = 'BrainGuided/debug/minor/new_result/final_2181/sig_test_lstm_feature/rt/squeezenet/result_rt_split=1_co_train_alpha=5_use_model=1_lr=0.0002_2_744/checkpoints/best.pth'
    
    elif opt.get_layer_contribution:
        if opt.network_choose == 'mobilenet_v1':
            opt.model_pretrained = f'BrainGuided/debug/minor/new_result/final_2181/sig_test/rt/mobilenet_v1/result_rt_split={opt.split}_not_co_train_lr=0.0002_1_744/checkpoints/best.pth'
        elif opt.network_choose == 'mobilenet_v2':
            opt.model_pretrained = f'BrainGuided/debug/minor/new_result/final_2181/sig_test/rt/mobilenet_v2_0.45x/result_rt_split={opt.split}_not_co_train_lr=0.0002_1/checkpoints/best.pth'
        elif opt.network_choose == 'resnet_18':
            opt.model_pretrained = f'BrainGuided/debug/minor/new_result/final_2181/sig_test/rt/resnet_18/result_rt_split={opt.split}_not_co_train_lr=0.0002_1_744/checkpoints/best.pth'
        elif opt.network_choose == 'shufflenet_v1':
            opt.model_pretrained = f'BrainGuided/debug/minor/new_result/final_2181/sig_test/rt/shufflenet_v1_1.5x/result_rt_split={opt.split}_not_co_train_lr=0.0002_1_744/checkpoints/best.pth'
        elif opt.network_choose == 'shufflenet_v2':
            opt.model_pretrained = f'BrainGuided/debug/minor/new_result/final_2181/sig_test/rt/shufflenet_v2/result_rt_split={opt.split}_not_co_train_lr=0.0002_1_744/checkpoints/best.pth'
        elif opt.network_choose == 'squeezenet':
            opt.model_pretrained = f'BrainGuided/debug/minor/new_result/final_2181/sig_test/rt/squeezenet/result_rt_split={opt.split}_not_co_train_lr=0.0002_1_744/checkpoints/best.pth'

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
          'n_classes=', opt.n_classes, 'dataset_choose=', opt.dataset_choose,'n_epoch=',opt.n_epochs)
    # print('alpha=',opt.alpha)
    local2global_path(opt)
    if opt.train_from_checkpoint: # trained from 
        model = torch.load(opt.model_pretrained)
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
    elif opt.get_layer_contribution:
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

    if opt.get_layer_contribution:
        run_model_get_contribution(opt, train_loader, neural_loader, model)
        return
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
    # gamma_result = np.zeros((opt.n_epochs,gamma.shape[1]+1))
    class_accuracy = np.zeros((opt.n_epochs,opt.n_classes))
    max_acc = 0 # set max acc to 0
    min_loss = np.inf
    # set early stopping
    patience = 50
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
                elif opt.align_only_last_layer:
                    total_loss, ce_loss, sim_loss, cosine_sim = co_train_epoch(i, train_loader, neural_loader, model, criterion, optimizer, opt)
                else:
                    gamma_temp, total_loss, ce_loss, sim_loss = co_train_epoch(i, train_loader, neural_loader, model, criterion, optimizer, opt)
                    gamma[i-1] = gamma_temp.detach().cpu().numpy()
        else:
            train_loss = train_epoch(i, train_loader, model, criterion, optimizer, opt, training_data.class_names)
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
            patience = 50
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
        # np.save(os.path.join(opt.result_path,'gamma.npy'),gamma)
        if (not opt.train_from_checkpoint) and (not opt.align_only_last_layer):
            np.savetxt(os.path.join(opt.result_path,'gamma_result.csv'), gamma,delimiter = ',')
    
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
          'n_classes=', opt.n_classes, 'dataset_choose=', opt.dataset_choose,'n_epoch=',opt.n_epochs)
    
    local2global_path(opt)
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
    neural_valid_data = NeuralValidDataset(opt,opt.neural_video_path,neural_response,opt.fps,spatial_transform,temporal_transform)
    neural_valid_loader = get_neural_loader(opt, neural_valid_data, shuffle=True)

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


def main_vanilla(sig_test_run,network,split):
    opt = parse_opts()
    opt.sig_test_run = sig_test_run
    opt.network_choose = network
    opt.split = split
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
          'n_classes=', opt.n_classes, 'dataset_choose=', opt.dataset_choose)
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

    result = np.zeros((opt.n_epochs,2))
    class_accuracy = np.zeros((opt.n_epochs, opt.n_classes))
    max_acc = 0
    min_loss = np.inf
    patience = 50
    for i in range(1, opt.n_epochs + 1):
        train_loss = train_epoch(i, train_loader, model, criterion, optimizer, opt, training_data.class_names)
        ep,acc,ac, loss = val_epoch_class(i, val_loader, model, criterion, opt, optimizer)
        result[i-1,0] = ep
        result[i-1,1] = acc
        class_accuracy[i-1] = ac
        acc_result = np.concatenate((result,class_accuracy),axis=1)
        np.savetxt(os.path.join(opt.result_path,'acc_result.csv'), acc_result,delimiter = ',')
        np.save(os.path.join(opt.result_path,'result.npy'),result)

        # save model
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
            patience = 50

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
        
        rois = ['ALL', 'RSC', 'PPA', 'EVC', 'TOS']
        for roi in rois:
            opt.roi = roi
            split_all = [1,2,3,4]
            for split in split_all:
                opt.split = split
                neural_response =[]
                neural_response_valid = []
                print(opt.use_model)
                if opt.use_model:
                    print('use_model')
                    # for Subject in range(1,6):
                    for Subject in range(1,2):
                        print(Subject)
                        if opt.use_lstm:
                            neural_response.append(np.load('Neural_data/emotion_encoding_results/sub-'+str(Subject).zfill(2)+f'/voxel_select_remain_time_{opt.split}.npy'))
                        else:
                            neural_response.append(np.load('Neural_data/emotion_encoding_results/sub-'+str(Subject).zfill(2)+f'/voxel_select_new_{opt.roi}_{opt.split}.npy'))
                            # neural_response.append(np.load('Neural_data/emotion_encoding_results/sub-'+str(Subject).zfill(2)+f'/voxel_select_lstm_feature_{opt.split}.npy')) # _lstm_feature_{opt.split}
                            # neural_response.append(np.load('Neural_data/emotion_encoding_results/sub-'+str(Subject).zfill(2)+f'/neurostorm_embeddings_pretrain_mae0.5_{opt.split}.npy')) # neurostorm_embeddings_cls_{opt.split}
                            # print(f"split {opt.split} neural_response shape {neural_response[-1].shape}")
                        if opt.train_only_layer_contribution:
                            neural_response_valid.append(np.load('Neural_data/emotion_encoding_results/sub-'+str(Subject).zfill(2)+f'/neurostorm_embeddings_pretrain_mae0.5_{opt.roi}_valid_{opt.split}.npy'))

                        # check if input has nan
                        if np.isnan(neural_response[-1]).any():
                            print('nan appears in neural_response')
                            neural_response[-1][np.isnan(neural_response[-1])] = 0

                        # neural_response.append(np.load('Neural_data/emotion_encoding_results/Subject'+str(Subject)+'/voxel_select.npy'))
                # print('neural_finish')

                # alpha_all = [1]
                alpha_all = [5]
                
                network_all = ['resnet_18', 'mobilenet_v1', 'mobilenet_v2', 'shufflenet_v1', 'shufflenet_v2', 'squeezenet']
                # network_all = ['shufflenet_v2']
                for network in network_all:
                    for alpha in alpha_all:
                        for sig_test_run in range(1,2):
                            print(f"split {opt.split} neural_response shape {neural_response[-1].shape}")
                            if opt.train_only_layer_contribution:
                                main_contribution(alpha,neural_response,neural_response_valid,sig_test_run,network,split, opt.roi)
                            else:
                                main_sig(alpha,neural_response,sig_test_run,network,split, opt.roi)
                            torch.cuda.empty_cache()

    else:
        network_all = ['resnet_18', 'mobilenet_v1', 'mobilenet_v2', 'shufflenet_v1', 'shufflenet_v2', 'squeezenet'] #['vaa']
        split_all = [3,4]
        for network in network_all:
            for split in split_all:
                for sig_test_run in range(1):
                    main_vanilla(sig_test_run+1,network,split)
                    torch.cuda.empty_cache()

"""
python main.py --expr_name demo
"""