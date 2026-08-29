import numpy as np

from core.utils import AverageMeter, process_data_item, run_model, calculate_accuracy, calculate_accuracy_annot, calculate_accuracy_annot_reg, process_neural_data_item,process_neural_data_item_v2,run_neural_model,run_neural_model_v2,process_behavior_data_item,run_model_contribution, run_neural_model_v3, run_neural_model_v3_add_pfc, run_neural_model_dapello

import time
import torch
import torch.nn.functional as F
from itertools import cycle
from core.utils import visualize_rdms, visualize_rdms_v2

def train_epoch(epoch, data_loader, model, criterion, optimizer, opt, class_names):
    print("# ---------------------------------------------------------------------- #")
    print('Training at epoch {}'.format(epoch))
    model.train()

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    accuracies = AverageMeter()
    aucs, youden_indexes, best_threshold_accs = AverageMeter(), AverageMeter(), AverageMeter()

    end_time = time.time()

    for i, data_item in enumerate(data_loader):
        visual, target, visualization_item, batch_size, annot = process_data_item(opt, data_item)
        if opt.single_annot_class == True:
            annot = annot[:,int(opt.target_class):int(opt.target_class)+1]
        data_time.update(time.time() - end_time)

        if opt.task == 'annot' or opt.task == 'annot-reg':
            output, loss = run_model(opt, [visual, annot], model, criterion, i, print_attention=False)
        else:
            output, loss = run_model(opt, [visual, target], model, criterion, i, print_attention=False)

        if opt.task == 'annot':
            acc, auc, youden_index, best_threshold_acc = calculate_accuracy_annot(output, annot)
            aucs.update(auc, batch_size)
            youden_indexes.update(youden_index, batch_size)
            best_threshold_accs.update(best_threshold_acc, batch_size)
        elif opt.task == 'annot-reg':
            acc = calculate_accuracy_annot_reg(output, annot)
        else:
            acc = calculate_accuracy(output, target)

        losses.update(loss.item(), batch_size)
        accuracies.update(acc, batch_size)

        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        batch_time.update(time.time() - end_time)
        end_time = time.time()

        Iter = (epoch - 1) * len(data_loader) + (i + 1)

        # if opt.debug:
        #     print('Epoch: [{0}][{1}/{2}]\t'
        #           'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
        #           'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
        #           'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
        #           'Acc {acc.val:.3f} ({acc.avg:.3f})'.format(
        #         epoch, i + 1, len(data_loader), batch_time=batch_time, data_time=data_time, loss=losses, acc=accuracies))

        torch.cuda.empty_cache()
        
    # ---------------------------------------------------------------------- #
    print("Epoch Time: {:.2f}min".format(batch_time.avg * len(data_loader) / 60))
    print("Train loss: {:.4f}".format(losses.avg))
    print("Train acc: {:.4f}".format(accuracies.avg))
    if opt.task == 'annot':
        print("Train auc: {:.4f}".format(aucs.avg))
        print("Train youden index: {:.4f}".format(youden_indexes.avg))
        print("Train best threshold acc: {:.4f}".format(best_threshold_accs.avg))

    return losses.avg

def co_train_epoch_behavior(epoch, train_loader,behavior_loader, model, criterion, optimizer, opt):
    print("# ---------------------------------------------------------------------- #")
    print('Training at epoch {}'.format(epoch))
    model.train()

    batch_time = AverageMeter()
    data_time = AverageMeter()
    accuracies = AverageMeter()
    end_time = time.time()
    total_losses_sum = []
    similarity_losses_sum = []
    ce_losses_sum = []
    for i, data_item in enumerate(zip(train_loader,behavior_loader)):
        train_data_item,behaivor_data_item = data_item
        visual, target, visualization_item, batch_size = process_data_item(opt, train_data_item)
        neural_visual, RSA_output, behavior_batch_size,visual_item = process_behavior_data_item(opt, behaivor_data_item)
        print(visual_item)
        data_time.update(time.time() - end_time)
        if i ==len(train_loader)-1:
            print_gamma=True
        else:
            print_gamma = False
        output, ce_loss = run_model(opt, [visual, target], model, criterion, i, print_attention=False)
        gamma,similarity_loss = run_neural_model(opt,[neural_visual, RSA_output],model,print_gamma)
        acc = calculate_accuracy(output, target)
        total_loss = ce_loss + opt.alpha * similarity_loss
        accuracies.update(acc, batch_size)

        # Backward and optimize
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

        batch_time.update(time.time() - end_time)
        end_time = time.time()

        total_losses_sum.append(total_loss.item())
        similarity_losses_sum.append(similarity_loss.item())
        ce_losses_sum.append(ce_loss.item())
        total_losses_avg = torch.mean(torch.tensor(total_losses_sum))
        similarity_losses_avg = torch.mean(torch.tensor(similarity_losses_sum))
        ce_losses_avg = torch.mean(torch.tensor(ce_losses_sum))

        if opt.debug:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Total_Loss {total_loss:.4f} ({total_loss_avg:.4f})\t'
                  'SIM_Loss {similarity_loss:.4f} ({similarity_loss_avg:.4f})\t'
                  'CE_Loss {ce_loss:.4f} ({ce_loss_avg:.4f})\t'
                  'Acc {acc.val:.3f} ({acc.avg:.3f})'.format(
                epoch, i + 1, len(train_loader), batch_time=batch_time, data_time=data_time, total_loss=total_loss, total_loss_avg=total_losses_avg,similarity_loss=similarity_loss,similarity_loss_avg=similarity_losses_avg,ce_loss=ce_loss,ce_loss_avg=ce_losses_avg,acc=accuracies))

    # ---------------------------------------------------------------------- #
    print("Epoch Time: {:.2f}min".format(batch_time.avg * len(train_loader) / 60))
    print("Train total loss: {:.4f}".format(torch.mean(torch.tensor(total_losses_sum))))
    print("Train ce loss: {:.4f}".format(torch.mean(torch.tensor(ce_losses_sum))))
    print("Train sim loss: {:.4f}".format(torch.mean(torch.tensor(similarity_losses_sum))))
    print("Train acc: {:.4f}".format(accuracies.avg))

    return gamma

def train_epoch_contribution(epoch, train_loader, model, criterion, optimizer, opt):
    print("# ---------------------------------------------------------------------- #")
    print('Training at epoch {}'.format(epoch))
    model.train()

    batch_time = AverageMeter()
    data_time = AverageMeter()
    end_time = time.time()
    mse_losses_sum = []
    contribution_all = {}
    for i, train_data_item in enumerate(train_loader):
        neural_visual, RSA_output, neural_batch_size,visual_item = process_neural_data_item(opt, train_data_item)
        _, neural_response,  _, _ = train_data_item
        voxel_select = neural_response[opt.data_use].cuda()
        data_time.update(time.time() - end_time)
        output, _, contribution = run_model_contribution(opt, [neural_visual, voxel_select], model)
        # print("output.shape",output.shape,"\nvoxel_select.shape",voxel_select.shape)
        output = torch.from_numpy(output).float().cuda()
        output.requires_grad_()
        mse_loss = criterion(output, voxel_select)

        # Backward and optimize
        optimizer.zero_grad()
        mse_loss.backward()
        optimizer.step()

        batch_time.update(time.time() - end_time)
        end_time = time.time()
        mse_losses_sum.append(mse_loss.item())
        # sum contribution to contribution_all and average
        for k in contribution:
            if k not in contribution_all.keys():
                contribution_all[k] = contribution[k]
            else:
                contribution_all[k] += contribution[k]

    # average contribution
    for k in contribution_all:
        contribution_all[k] = contribution_all[k] / len(train_loader)
    print("Epoch Time: {:.2f}min".format(batch_time.avg * len(train_loader) / 60))
    print("Train total loss: {:.4f}".format(torch.mean(torch.tensor(mse_losses_sum))))

    return torch.mean(torch.tensor(mse_losses_sum)), contribution_all

def co_train_epoch_each_roi(epoch, train_loader, neural_loader_evc, neural_loader_tos, neural_loader_ppa, neural_loader_rsc, model, criterion, optimizer, opt):
    print("# ---------------------------------------------------------------------- #")
    print('Training at epoch {}'.format(epoch))
    model.train()

    batch_time = AverageMeter()
    data_time = AverageMeter()
    accuracies = AverageMeter()
    aucs, youden_indexes, best_threshold_accs = AverageMeter(), AverageMeter(), AverageMeter()
    end_time = time.time()
    total_losses_sum = []
    similarity_losses_sum = []
    ce_losses_sum = []
    dataloader_iterator_evc = iter(neural_loader_evc)
    dataloader_iterator_tos = iter(neural_loader_tos)
    dataloader_iterator_ppa = iter(neural_loader_ppa)
    dataloader_iterator_rsc = iter(neural_loader_rsc)

    corr_accum = {'evc': [], 'tos': [], 'ppa': [], 'rsc': []}

    for i, train_data_item in enumerate(train_loader):
        try:
            neural_data_item_evc = next(dataloader_iterator_evc)
            neural_data_item_tos = next(dataloader_iterator_tos)
            neural_data_item_ppa = next(dataloader_iterator_ppa)
            neural_data_item_rsc = next(dataloader_iterator_rsc)
        except StopIteration:
            dataloader_iterator_evc = iter(neural_loader_evc)
            neural_data_item_evc = next(dataloader_iterator_evc)
            dataloader_iterator_tos = iter(neural_loader_tos)
            neural_data_item_tos = next(dataloader_iterator_tos)
            dataloader_iterator_ppa = iter(neural_loader_ppa)
            neural_data_item_ppa = next(dataloader_iterator_ppa)
            dataloader_iterator_rsc = iter(neural_loader_rsc)
            neural_data_item_rsc = next(dataloader_iterator_rsc)
        visual, target, _, batch_size, annot = process_data_item(opt, train_data_item)
        if opt.single_annot_class == True:
            if opt.target_class == 'obj':
                # annot[1,2,3,4]
                annot = annot[:,1:5]
            elif opt.target_class == 'subj':
                # annot[0, 5-14]
                annot_1 = annot[:,0:1]
                annot_2 = annot[:,5:15]
                annot = torch.cat([annot_1, annot_2], dim=1)
            else:
                annot = annot[:,int(opt.target_class):int(opt.target_class)+1]
        neural_visual_evc, RSA_output_evc, _, _, neural_target_evc = process_neural_data_item(opt, neural_data_item_evc) # if want 4*4 rdm use v2
        neural_visual_tos, RSA_output_tos, _, _, neural_target_tos = process_neural_data_item(opt, neural_data_item_tos)
        neural_visual_ppa, RSA_output_ppa, _, _, neural_target_ppa = process_neural_data_item(opt, neural_data_item_ppa)
        neural_visual_rsc, RSA_output_rsc, _, _, neural_target_rsc = process_neural_data_item(opt, neural_data_item_rsc)

        data_time.update(time.time() - end_time)
        if i ==len(train_loader)-1:
            print_gamma=True
        else:
            print_gamma = False

        if opt.task == 'annot' or opt.task == 'annot-reg':
            output, ce_loss = run_model(opt, [visual, annot], model, criterion, i, print_attention=False)
        else:
            output, ce_loss = run_model(opt, [visual, target], model, criterion, i, print_attention=False)
        gamma, similarity_loss, S_cnn, samples_num = run_neural_model_v3(
            opt,
            [neural_visual_evc, neural_visual_tos, neural_visual_ppa, neural_visual_rsc], [RSA_output_evc, RSA_output_tos, RSA_output_ppa, RSA_output_rsc],
            model, epoch, [neural_target_evc,neural_target_tos,neural_target_ppa,neural_target_rsc], print_gamma   # <-- added neural_target
        )
        # if epoch == 1:
        #     for roi in ['evc', 'tos', 'ppa', 'rsc']:
        #         corr_accum[roi].append(init_corrs[roi])

        if ((epoch%1) == 0) and (i == 0):
            visualize_rdms_v2(opt, S_cnn[0], S_cnn[4]['evc'], samples_num['evc'], RSA_output_evc, neural_data_item_evc, epoch=epoch, save_dir="rdm_plots_evc")
            visualize_rdms_v2(opt, S_cnn[1], S_cnn[4]['tos'], samples_num['tos'], RSA_output_tos, neural_data_item_tos, epoch=epoch, save_dir="rdm_plots_tos")
            visualize_rdms_v2(opt, S_cnn[2], S_cnn[4]['ppa'], samples_num['ppa'], RSA_output_ppa, neural_data_item_ppa, epoch=epoch, save_dir="rdm_plots_ppa")
            visualize_rdms_v2(opt, S_cnn[3], S_cnn[4]['rsc'], samples_num['rsc'], RSA_output_rsc, neural_data_item_rsc, epoch=epoch, save_dir="rdm_plots_rsc")

        if opt.task == 'annot':
            acc, auc, youden_index, best_threshold_acc = calculate_accuracy_annot(output, annot)
            aucs.update(auc, batch_size)
            youden_indexes.update(youden_index, batch_size)
            best_threshold_accs.update(best_threshold_acc, batch_size)
        elif opt.task == 'annot-reg':
            acc = calculate_accuracy_annot_reg(output, annot)
        else:
            acc = calculate_accuracy(output, target)
        total_loss = ce_loss + opt.alpha * similarity_loss
        accuracies.update(acc, batch_size)
        

        # Backward and optimize
        optimizer.zero_grad()
        total_loss.backward()
        # gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        if model.gamma.grad is not None:
            if torch.isnan(model.gamma.grad).any():
                print("NaN in gamma.grad")
                print("gamma.grad:", model.gamma.grad)

        batch_time.update(time.time() - end_time)
        end_time = time.time()

        total_losses_sum.append(total_loss.item())
        if similarity_loss is not None: similarity_losses_sum.append(similarity_loss.item())
        ce_losses_sum.append(ce_loss.item())

    # if epoch == 1:
    #     with torch.no_grad():
    #         log_weights = []
    #         for roi in ['evc', 'tos', 'ppa', 'rsc']:
    #             mean_corr = torch.stack(corr_accum[roi]).mean(dim=0)  # (n_layers,)
    #             w = F.softmax(mean_corr, dim=0)
    #             print(f"[init] {roi} mean corrs: {mean_corr.numpy()}")
    #             print(f"[init] {roi} mean weights: {w.numpy()}")
    #             log_weights.append(torch.log(w.clamp(min=1e-8)))
    #         model.gamma.data = torch.cat(log_weights).to(model.gamma.device)
    #         print("[init] Gamma initialized from epoch-1 averaged correlations.")
    # ---------------------------------------------------------------------- #
    print("Epoch Time: {:.2f}min".format(batch_time.avg * len(train_loader) / 60))
    print("Train total loss: {:.4f}".format(torch.mean(torch.tensor(total_losses_sum))))
    print("Train ce loss: {:.4f}".format(torch.mean(torch.tensor(ce_losses_sum))))
    print("Train sim loss: {:.4f}".format(torch.mean(torch.tensor(similarity_losses_sum))))
    print("Train acc: {:.4f}".format(accuracies.avg))
    if opt.task == 'annot':
        print("Train auc: {:.4f}".format(aucs.avg))
        print("Train youden index: {:.4f}".format(youden_indexes.avg))
        print("Train best threshold acc: {:.4f}".format(best_threshold_accs.avg))

    return gamma, torch.mean(torch.tensor(total_losses_sum)), torch.mean(torch.tensor(ce_losses_sum)), torch.mean(torch.tensor(similarity_losses_sum))

def co_train_epoch_each_roi_add_pfc(epoch, train_loader, neural_loader_evc, neural_loader_tos, neural_loader_ppa, neural_loader_rsc, neural_loader_pfc, model, criterion, optimizer, opt):
    print("# ---------------------------------------------------------------------- #")
    print('Training at epoch {}'.format(epoch))
    model.train()

    batch_time = AverageMeter()
    data_time = AverageMeter()
    accuracies = AverageMeter()
    aucs, youden_indexes, best_threshold_accs = AverageMeter(), AverageMeter(), AverageMeter()
    end_time = time.time()
    total_losses_sum = []
    similarity_losses_sum = []
    ce_losses_sum = []
    dataloader_iterator_evc = iter(neural_loader_evc)
    dataloader_iterator_tos = iter(neural_loader_tos)
    dataloader_iterator_ppa = iter(neural_loader_ppa)
    dataloader_iterator_rsc = iter(neural_loader_rsc)
    dataloader_iterator_pfc = iter(neural_loader_pfc)

    corr_accum = {'evc': [], 'tos': [], 'ppa': [], 'rsc': []}

    for i, train_data_item in enumerate(train_loader):
        try:
            neural_data_item_evc = next(dataloader_iterator_evc)
            neural_data_item_tos = next(dataloader_iterator_tos)
            neural_data_item_ppa = next(dataloader_iterator_ppa)
            neural_data_item_rsc = next(dataloader_iterator_rsc)
            neural_data_item_pfc = next(dataloader_iterator_pfc)
        except StopIteration:
            dataloader_iterator_evc = iter(neural_loader_evc)
            neural_data_item_evc = next(dataloader_iterator_evc)
            dataloader_iterator_tos = iter(neural_loader_tos)
            neural_data_item_tos = next(dataloader_iterator_tos)
            dataloader_iterator_ppa = iter(neural_loader_ppa)
            neural_data_item_ppa = next(dataloader_iterator_ppa)
            dataloader_iterator_rsc = iter(neural_loader_rsc)
            neural_data_item_rsc = next(dataloader_iterator_rsc)
            dataloader_iterator_pfc = iter(neural_loader_pfc)
            neural_data_item_pfc = next(dataloader_iterator_pfc)
        visual, target, _, batch_size, annot = process_data_item(opt, train_data_item)
        if opt.single_annot_class == True:
            if opt.target_class == 'obj':
                # annot[1,2,3,4]
                annot = annot[:,1:5]
            elif opt.target_class == 'subj':
                # annot[0, 5-14]
                annot_1 = annot[:,0:1]
                annot_2 = annot[:,5:15]
                annot = torch.cat([annot_1, annot_2], dim=1)
            else:
                annot = annot[:,int(opt.target_class):int(opt.target_class)+1]
        neural_visual_evc, RSA_output_evc, _, _, neural_target_evc = process_neural_data_item(opt, neural_data_item_evc) # if want 4*4 rdm use v2
        neural_visual_tos, RSA_output_tos, _, _, neural_target_tos = process_neural_data_item(opt, neural_data_item_tos)
        neural_visual_ppa, RSA_output_ppa, _, _, neural_target_ppa = process_neural_data_item(opt, neural_data_item_ppa)
        neural_visual_rsc, RSA_output_rsc, _, _, neural_target_rsc = process_neural_data_item(opt, neural_data_item_rsc)
        neural_visual_pfc, RSA_output_pfc, _, _, neural_target_pfc = process_neural_data_item(opt, neural_data_item_pfc)

        data_time.update(time.time() - end_time)
        if i ==len(train_loader)-1:
            print_gamma=True
        else:
            print_gamma = False

        if opt.task == 'annot' or opt.task == 'annot-reg':
            output, ce_loss = run_model(opt, [visual, annot], model, criterion, i, print_attention=False)
        else:
            output, ce_loss = run_model(opt, [visual, target], model, criterion, i, print_attention=False)
        gamma, similarity_loss, S_cnn, samples_num = run_neural_model_v3_add_pfc(
            opt,
            [neural_visual_evc, neural_visual_tos, neural_visual_ppa, neural_visual_rsc, neural_visual_pfc], [RSA_output_evc, RSA_output_tos, RSA_output_ppa, RSA_output_rsc, RSA_output_pfc],
            model, epoch, [neural_target_evc,neural_target_tos,neural_target_ppa,neural_target_rsc,neural_target_pfc], print_gamma   # <-- added neural_target
        )
        # if epoch == 1:
        #     for roi in ['evc', 'tos', 'ppa', 'rsc']:
        #         corr_accum[roi].append(init_corrs[roi])

        if ((epoch%1) == 0) and (i == 0):
            visualize_rdms_v2(opt, S_cnn[0], S_cnn[5]['evc'], samples_num['evc'], RSA_output_evc, neural_data_item_evc, epoch=epoch, save_dir="rdm_plots_evc")
            visualize_rdms_v2(opt, S_cnn[1], S_cnn[5]['tos'], samples_num['tos'], RSA_output_tos, neural_data_item_tos, epoch=epoch, save_dir="rdm_plots_tos")
            visualize_rdms_v2(opt, S_cnn[2], S_cnn[5]['ppa'], samples_num['ppa'], RSA_output_ppa, neural_data_item_ppa, epoch=epoch, save_dir="rdm_plots_ppa")
            visualize_rdms_v2(opt, S_cnn[3], S_cnn[5]['rsc'], samples_num['rsc'], RSA_output_rsc, neural_data_item_rsc, epoch=epoch, save_dir="rdm_plots_rsc")
            visualize_rdms_v2(opt, S_cnn[4], S_cnn[5]['pfc'], samples_num['pfc'], RSA_output_pfc, neural_data_item_pfc, epoch=epoch, save_dir="rdm_plots_pfc")

        if opt.task == 'annot':
            acc, auc, youden_index, best_threshold_acc = calculate_accuracy_annot(output, annot)
            aucs.update(auc, batch_size)
            youden_indexes.update(youden_index, batch_size)
            best_threshold_accs.update(best_threshold_acc, batch_size)
        elif opt.task == 'annot-reg':
            acc = calculate_accuracy_annot_reg(output, annot)
        else:
            acc = calculate_accuracy(output, target)
        total_loss = ce_loss + opt.alpha * similarity_loss
        accuracies.update(acc, batch_size)
        

        # Backward and optimize
        optimizer.zero_grad()
        total_loss.backward()
        # gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        if model.gamma.grad is not None:
            if torch.isnan(model.gamma.grad).any():
                print("NaN in gamma.grad")
                print("gamma.grad:", model.gamma.grad)

        batch_time.update(time.time() - end_time)
        end_time = time.time()

        total_losses_sum.append(total_loss.item())
        if similarity_loss is not None: similarity_losses_sum.append(similarity_loss.item())
        ce_losses_sum.append(ce_loss.item())

    # ---------------------------------------------------------------------- #
    print("Epoch Time: {:.2f}min".format(batch_time.avg * len(train_loader) / 60))
    print("Train total loss: {:.4f}".format(torch.mean(torch.tensor(total_losses_sum))))
    print("Train ce loss: {:.4f}".format(torch.mean(torch.tensor(ce_losses_sum))))
    print("Train sim loss: {:.4f}".format(torch.mean(torch.tensor(similarity_losses_sum))))
    print("Train acc: {:.4f}".format(accuracies.avg))
    if opt.task == 'annot':
        print("Train auc: {:.4f}".format(aucs.avg))
        print("Train youden index: {:.4f}".format(youden_indexes.avg))
        print("Train best threshold acc: {:.4f}".format(best_threshold_accs.avg))

    return gamma, torch.mean(torch.tensor(total_losses_sum)), torch.mean(torch.tensor(ce_losses_sum)), torch.mean(torch.tensor(similarity_losses_sum))

def co_train_epoch(epoch, train_loader,neural_loader, model, criterion, optimizer, opt):
    print("# ---------------------------------------------------------------------- #")
    print('Training at epoch {}'.format(epoch))
    model.train()

    batch_time = AverageMeter()
    data_time = AverageMeter()
    accuracies = AverageMeter()
    aucs, youden_indexes, best_threshold_accs = AverageMeter(), AverageMeter(), AverageMeter()
    end_time = time.time()
    total_losses_sum = []
    similarity_losses_sum = []
    ce_losses_sum = []
    dataloader_iterator1 = iter(neural_loader)
    for i, train_data_item in enumerate(train_loader):
        try:
            neural_data_item = next(dataloader_iterator1)
        except StopIteration:
            dataloader_iterator1 = iter(neural_loader)
            neural_data_item = next(dataloader_iterator1)
        visual, target, visualization_item, batch_size, annot = process_data_item(opt, train_data_item)
        if opt.single_annot_class == True:
            if opt.target_class == 'obj':
                # annot[1,2,3,4]
                annot = annot[:,1:5]
            elif opt.target_class == 'subj':
                # annot[0, 5-14]
                annot_1 = annot[:,0:1]
                annot_2 = annot[:,5:15]
                annot = torch.cat([annot_1, annot_2], dim=1)
            else:
                annot = annot[:,int(opt.target_class):int(opt.target_class)+1]
        neural_visual, RSA_output, neural_batch_size,visual_item = process_neural_data_item(opt, neural_data_item)
        _, neural_response,  _, _ = neural_data_item
        voxel_select = neural_response[opt.data_use].cuda()
        # print("voxel_select.shape:",voxel_select.shape)
        # print("visual_item:", visual_item)
        data_time.update(time.time() - end_time)
        if i ==len(train_loader)-1:
            print_gamma=True
        else:
            print_gamma = False

        if opt.task == 'annot' or opt.task == 'annot-reg':
            output, ce_loss = run_model(opt, [visual, annot], model, criterion, i, print_attention=False)
        else:
            output, ce_loss = run_model(opt, [visual, target], model, criterion, i, print_attention=False)

        if opt.align_only_last_layer:
            similarity_loss, cosine_sim = run_neural_model_v2(opt,[neural_visual, voxel_select],model,epoch,print_gamma)
        elif opt.dapello == True:
            similarity_loss, S_cnn, RSA_target = run_neural_model_dapello(opt,[neural_visual, RSA_output],model,print_gamma)
            # if i == 0:
            #     visualize_rdms(opt, S_cnn, RSA_target, neural_data_item, epoch=epoch)
        else:
            gamma,similarity_loss, S_cnn, RSA_target = run_neural_model(opt,[neural_visual, RSA_output],model,print_gamma)
            if i == 0:
                visualize_rdms(opt, S_cnn, RSA_target, neural_data_item, epoch=epoch)
        # gamma,similarity_loss = run_neural_model(opt,[neural_visual, RSA_output],model,print_gamma)
        if opt.task == 'annot':
            acc, auc, youden_index, best_threshold_acc = calculate_accuracy_annot(output, annot)
            aucs.update(auc, batch_size)
            youden_indexes.update(youden_index, batch_size)
            best_threshold_accs.update(best_threshold_acc, batch_size)
        elif opt.task == 'annot-reg':
            acc = calculate_accuracy_annot_reg(output, annot)
        else:
            acc = calculate_accuracy(output, target)
        # total_loss = similarity_loss # 251007 only_sim
        if similarity_loss is None:
            total_loss = ce_loss
        else:
            total_loss = ce_loss + opt.alpha * similarity_loss
        accuracies.update(acc, batch_size)

        # Backward and optimize
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

        batch_time.update(time.time() - end_time)
        end_time = time.time()

        total_losses_sum.append(total_loss.item())
        if similarity_loss is not None: similarity_losses_sum.append(similarity_loss.item())
        ce_losses_sum.append(ce_loss.item())
        # total_losses_avg = torch.mean(torch.tensor(total_losses_sum))
        # similarity_losses_avg = torch.mean(torch.tensor(similarity_losses_sum))
        # ce_losses_avg = torch.mean(torch.tensor(ce_losses_sum))

        # if opt.debug:
        #     if epoch < int(opt.mixup_pct * opt.n_epochs):
        #         print('Epoch: [{0}][{1}/{2}]\t'
        #           'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
        #           'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
        #           'Total_Loss {total_loss:.4f} ({total_loss_avg:.4f})\t'
        #           'MIXCO_Loss {similarity_loss:.4f} ({similarity_loss_avg:.4f})\t'
        #           'CE_Loss {ce_loss:.4f} ({ce_loss_avg:.4f})\t'
        #           'Acc {acc.val:.3f} ({acc.avg:.3f})'.format(
        #         epoch, i + 1, len(train_loader), batch_time=batch_time, data_time=data_time, total_loss=total_loss, total_loss_avg=total_losses_avg,similarity_loss=similarity_loss,similarity_loss_avg=similarity_losses_avg,ce_loss=ce_loss,ce_loss_avg=ce_losses_avg,acc=accuracies))
        #     else:
        #         print('Epoch: [{0}][{1}/{2}]\t'
        #             'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
        #             'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
        #             'Total_Loss {total_loss:.4f} ({total_loss_avg:.4f})\t'
        #             'SIM_Loss {similarity_loss:.4f} ({similarity_loss_avg:.4f})\t'
        #             'CE_Loss {ce_loss:.4f} ({ce_loss_avg:.4f})\t'
        #             'Acc {acc.val:.3f} ({acc.avg:.3f})'.format(
        #             epoch, i + 1, len(train_loader), batch_time=batch_time, data_time=data_time, total_loss=total_loss, total_loss_avg=total_losses_avg,similarity_loss=similarity_loss,similarity_loss_avg=similarity_losses_avg,ce_loss=ce_loss,ce_loss_avg=ce_losses_avg,acc=accuracies))
        
        torch.cuda.empty_cache()

    # ---------------------------------------------------------------------- #
    print("Epoch Time: {:.2f}min".format(batch_time.avg * len(train_loader) / 60))
    print("Train total loss: {:.4f}".format(torch.mean(torch.tensor(total_losses_sum))))
    print("Train ce loss: {:.4f}".format(torch.mean(torch.tensor(ce_losses_sum))))
    print("Train sim loss: {:.4f}".format(torch.mean(torch.tensor(similarity_losses_sum))))
    print("Train acc: {:.4f}".format(accuracies.avg))
    if opt.task == 'annot':
        print("Train auc: {:.4f}".format(aucs.avg))
        print("Train youden index: {:.4f}".format(youden_indexes.avg))
        print("Train best threshold acc: {:.4f}".format(best_threshold_accs.avg))

    if opt.align_only_last_layer:
        return torch.mean(torch.tensor(total_losses_sum)), torch.mean(torch.tensor(ce_losses_sum)), torch.mean(torch.tensor(similarity_losses_sum)), cosine_sim
    elif opt.dapello == True:
        return torch.mean(torch.tensor(total_losses_sum)), torch.mean(torch.tensor(ce_losses_sum)), torch.mean(torch.tensor(similarity_losses_sum))
    else:
        return gamma, torch.mean(torch.tensor(total_losses_sum)), torch.mean(torch.tensor(ce_losses_sum)), torch.mean(torch.tensor(similarity_losses_sum))

# ✅ 1. Update the function signature to accept the new models and optimizers
def co_train_epoch_lstm(epoch, train_loader, neural_loader, cnn_model, fmri_model, criterion, cnn_optimizer, fmri_optimizer, opt):
    torch.autograd.set_detect_anomaly(True)
    print("# ---------------------------------------------------------------------- #")
    print('Training at epoch {}'.format(epoch))
    fmri_model.train()
    batch_time = AverageMeter()
    data_time = AverageMeter()
    accuracies = AverageMeter()
    end_time = time.time()
    lstm_ce_losses_sum = []
    dataloader_iterator1 = iter(neural_loader)
    
    for i, train_data_item in enumerate(train_loader):
        try:
            neural_data_item = next(dataloader_iterator1)
        except StopIteration:
            dataloader_iterator1 = iter(neural_loader)
            neural_data_item = next(dataloader_iterator1)
        print("Debugging neural_data_item:", type(neural_data_item), "Length:", len(neural_data_item))
            
        data_time.update(time.time() - end_time)

        # --- NEW DATAFLOW for fMRI ---
        
        # ✅ 1. Unpack raw data from the loader.
        # We assume `neural_data_item` is a tuple like (video_data, fmri_sequence_data).
        visual_1, neural_response,  visualization_item_1, target_1 = neural_data_item
        raw_fmri_sequence = neural_response[opt.data_use].cuda().float()
        target_1 = target_1.cuda()
        # raw_fmri_sequence = raw_fmri_sequence.cuda() # Move raw data to GPU
        print("raw_fmri_sequence:", raw_fmri_sequence.shape)
        print("lstm target:", target_1.shape)

        # ✅ 2. Apply the LSTM model FIRST.
        processed_fmri_features = {}
        # The LSTM processes the raw time-series data.
        lstm_output, processed_fmri_features[opt.data_use] = fmri_model(raw_fmri_sequence)
        print("processed_fmri_features:", processed_fmri_features[opt.data_use].shape)
        lstm_ce_loss = criterion(lstm_output, target_1)
            
        acc = calculate_accuracy(lstm_output, target_1)
        accuracies.update(acc, opt.batch_size)

        fmri_optimizer.zero_grad()

        lstm_ce_loss.backward()

        torch.nn.utils.clip_grad_norm_(fmri_model.parameters(), max_norm=1.0)

        fmri_optimizer.step()

        # (The rest of the logging and return logic remains the same)
        batch_time.update(time.time() - end_time)
        end_time = time.time()
        lstm_ce_losses_sum.append(lstm_ce_loss.item())
        lstm_ce_losses_avg = torch.mean(torch.tensor(lstm_ce_losses_sum))

        if opt.debug:
            print('LSTM Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'LSTM_CE_Loss {lstm_ce_loss:.4f} ({lstm_ce_losses_avg:.4f})\t'
                  'Acc {acc.val:.3f} ({acc.avg:.3f})'.format(
                epoch, i + 1, len(train_loader), batch_time=batch_time, data_time=data_time,lstm_ce_loss=lstm_ce_loss,lstm_ce_losses_avg=lstm_ce_losses_avg,acc=accuracies))


    # (Variable initializations remain the same)
    cnn_model.train()
    batch_time_1 = AverageMeter()
    data_time_1 = AverageMeter()
    accuracies_1 = AverageMeter()
    end_time = time.time()
    total_losses_sum = []
    similarity_losses_sum = []
    ce_losses_sum = []
    dataloader_iterator1 = iter(neural_loader)
    
    for i, train_data_item in enumerate(train_loader):
        try:
            neural_data_item = next(dataloader_iterator1)
        except StopIteration:
            dataloader_iterator1 = iter(neural_loader)
            neural_data_item = next(dataloader_iterator1)
        print("Debugging neural_data_item:", type(neural_data_item), "Length:", len(neural_data_item))
            
        visual, target, visualization_item, batch_size, annot = process_data_item(opt, train_data_item)
        data_time_1.update(time.time() - end_time)

        # --- NEW DATAFLOW for fMRI ---
        
        # ✅ 1. Unpack raw data from the loader.
        # We assume `neural_data_item` is a tuple like (video_data, fmri_sequence_data).
        visual_1, neural_response,  visualization_item_1, target_1 = neural_data_item
        raw_fmri_sequence = neural_response[opt.data_use].cuda().float()
        visual = visual.cuda()
        target = target.cuda()
        # raw_fmri_sequence = raw_fmri_sequence.cuda() # Move raw data to GPU
        print("raw_fmri_sequence:", raw_fmri_sequence.shape)

        # ✅ 2. Apply the LSTM model FIRST.
        processed_fmri_features = {}
        # The LSTM processes the raw time-series data.
        lstm_output, processed_fmri_features[opt.data_use] = fmri_model(raw_fmri_sequence)
        print("processed_fmri_features:", processed_fmri_features[opt.data_use].shape)

        # ✅ 3. Pass the LSTM's OUTPUT to your processing function.
        # We create a new tuple to pass to your processing function.
        # Your `process_neural_data_item` function will now receive the (batch, 128) feature tensor.
        processed_neural_data_item = (visual_1, processed_fmri_features, visualization_item_1, target_1)
        neural_visual, similarity_target, neural_batch_size, visual_item = process_neural_data_item(opt, processed_neural_data_item)
        
        print("visual_item:", visual_item)
        # --- END of new dataflow ---

        if i == len(train_loader) - 1:
            print_gamma = True
        else:
            print_gamma = False
            
        # Forward pass for the classification task
        output, ce_loss = run_model(opt, [visual, target], cnn_model, criterion, i, print_attention=False)
        
        # Forward pass for the similarity task using the processed data
        gamma, similarity_loss, S_cnn, RSA_target = run_neural_model(
            opt, [neural_visual, similarity_target], cnn_model, print_gamma
        )
        
        if i == 0:
            visualize_rdms(opt, S_cnn, RSA_target, epoch=epoch)
            
        acc = calculate_accuracy(output, target)
        total_loss = ce_loss + opt.alpha * similarity_loss
        accuracies_1.update(acc, batch_size)

        # Backward and optimize for BOTH models
        cnn_optimizer.zero_grad()
        
        total_loss.backward()

        cnn_optimizer.step()

        # (The rest of the logging and return logic remains the same)
        batch_time_1.update(time.time() - end_time)
        end_time = time.time()
        total_losses_sum.append(total_loss.item())
        similarity_losses_sum.append(similarity_loss.item())
        ce_losses_sum.append(ce_loss.item())
        total_losses_avg = torch.mean(torch.tensor(total_losses_sum))
        similarity_losses_avg = torch.mean(torch.tensor(similarity_losses_sum))
        ce_losses_avg = torch.mean(torch.tensor(ce_losses_sum))

        if opt.debug:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time_1.val:.3f} ({batch_time_1.avg:.3f})\t'
                  'Data {data_time_1.val:.3f} ({data_time_1.avg:.3f})\t'
                  'Total_Loss {total_loss:.4f} ({total_loss_avg:.4f})\t'
                  'SIM_Loss {similarity_loss:.4f} ({similarity_loss_avg:.4f})\t'
                  'CE_Loss {ce_loss:.4f} ({ce_loss_avg:.4f})\t'
                  'Acc {acc.val:.3f} ({acc.avg:.3f})'.format(
                epoch, i + 1, len(train_loader), batch_time_1=batch_time_1, data_time_1=data_time_1, total_loss=total_loss, total_loss_avg=total_losses_avg,similarity_loss=similarity_loss,similarity_loss_avg=similarity_losses_avg,ce_loss=ce_loss,ce_loss_avg=ce_losses_avg,acc=accuracies_1))

    print("# ---------------------------------------------------------------------- #")
    print("Epoch Time: {:.2f}min".format(batch_time_1.avg * len(train_loader) / 60))
    print("Train total loss: {:.4f}".format(torch.mean(torch.tensor(total_losses_sum))))
    print("Train ce loss: {:.4f}".format(torch.mean(torch.tensor(ce_losses_sum))))
    print("Train sim loss: {:.4f}".format(torch.mean(torch.tensor(similarity_losses_sum))))
    print("Train LSTM ce loss: {:.4f}".format(torch.mean(torch.tensor(lstm_ce_losses_sum))))
    print("Train acc: {:.4f}".format(accuracies_1.avg))

    return gamma, torch.mean(torch.tensor(total_losses_sum)), torch.mean(torch.tensor(ce_losses_sum)), torch.mean(torch.tensor(similarity_losses_sum)), torch.mean(torch.tensor(lstm_ce_losses_sum))