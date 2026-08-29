from core.utils import AverageMeter, process_data_item, process_neural_data_item, run_model, calculate_accuracy,calculate_accuracy_cross, run_model_contribution, plot_output,calculate_accuracy_annot, calculate_accuracy_cross_annot, calculate_accuracy_annot_reg, calculate_accuracy_cross_annot_reg, plot_regression_result, plot_annot_result

import os
import time
import torch
import numpy as np

def val_epoch_contribution(epoch, data_loader, model, criterion, optimizer, opt):
    print("# ---------------------------------------------------------------------- #")
    print('Validation at epoch {}'.format(epoch))
    model.eval()

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    end_time = time.time()

    contribution_all = {}

    for i, data_item in enumerate(data_loader):
        neural_visual, RSA_output, neural_batch_size,visual_item = process_neural_data_item(opt, data_item)
        _, neural_response,  _, _ = data_item
        voxel_select = neural_response[opt.data_use].cuda()
        data_time.update(time.time() - end_time)
        with torch.no_grad():
            output, mse_loss, contribution = run_model_contribution(opt, [neural_visual, voxel_select], model)
            # plot output and voxel select (sample, voxels)
            plot_output(output, voxel_select, epoch, opt)

        losses.update(mse_loss.item(), neural_batch_size)
        batch_time.update(time.time() - end_time)
        end_time = time.time()

        for k in contribution:
            if k not in contribution_all.keys():
                contribution_all[k] = contribution[k]
            else:
                contribution_all[k] += contribution[k]

    # average contribution
    for k in contribution_all:
        contribution_all[k] = contribution_all[k] / len(data_loader)

    print("Val loss: {:.4f}".format(losses.avg))
    return losses.avg, contribution

def val_epoch(epoch, data_loader, model, criterion, opt, optimizer):
    print("# ---------------------------------------------------------------------- #")
    print('Validation at epoch {}'.format(epoch))
    model.eval()

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    accuracies = AverageMeter()

    end_time = time.time()

    for i, data_item in enumerate(data_loader):
        visual, target,  visualization_item, batch_size, annot = process_data_item(opt, data_item)
        data_time.update(time.time() - end_time)
        with torch.no_grad():
            if opt.task == 'annot':
                output, loss = run_model(opt, [visual, annot], model, criterion, i,print_attention=False)
            else:
                output, loss = run_model(opt, [visual, target], model, criterion, i,print_attention=False)

        if opt.task == 'annot':
            acc = calculate_accuracy_annot(output, annot)
        else:
            acc = calculate_accuracy(output, target)

        losses.update(loss.item(), batch_size)
        accuracies.update(acc, batch_size)
        batch_time.update(time.time() - end_time)
        end_time = time.time()

    print("Val loss: {:.4f}".format(losses.avg))
    print("Val acc: {:.4f}".format(accuracies.avg))

    save_file_path = os.path.join(opt.ckpt_path, 'save_{}.pth'.format(epoch))
    states = {
        'epoch': epoch + 1,
        'state_dict': model.state_dict(),
        'optimizer': optimizer.state_dict(),
    }
    # if epoch % 1 ==0:
    #     torch.save(states, save_file_path)
    return epoch,accuracies.avg,losses.avg

def val_epoch_class(epoch, data_loader, model, criterion, opt, optimizer):
    print("# ---------------------------------------------------------------------- #")
    print('Validation at epoch {}'.format(epoch))
    model.eval()

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    accuracies = AverageMeter()
    aucs, youden_indexes, best_threshold_accs = AverageMeter(), AverageMeter(), AverageMeter()
    if opt.dataset_choose == 've8':
        class_name = ['Anger', 'Anticipation', 'Disgust', 'Fear', 'Joy', 'Sadness', 'Surprise', 'Trust']
    elif opt.dataset_choose == 'ek6':
        class_name = ['Anger', 'Disgust', 'Fear', 'Joy', 'Sadness', 'Surprise']
    elif opt.dataset_choose == 'rt':
        if opt.task == 'design':
            class_name = ['MODN', 'MUJI', 'SCAN', 'WABI']
        elif opt.task == 'space':
            class_name = ['SPACE_05', 'SPACE_06', 'SPACE_07', 'SPACE_08', 'SPACE_09', 'SPACE_10', 'SPACE_11', 'SPACE_12']
        elif opt.task == 'annot' or opt.task == 'annot-reg':
            class_name = ['color_comfort','light_association','complexity','organization','naturalness','interest','valence','stimulation',
                          'vitality','comfort','relaxation','hominess','uplift','approachability','explorability']
            if opt.single_annot_class == True:
                if opt.target_class == 'obj':
                    class_name = ['light_association','complexity','organization','naturalness']
                elif opt.target_class == 'subj':
                    class_name = ['color_comfort','interest','valence','stimulation', 'vitality','comfort','relaxation','hominess','uplift','approachability','explorability']
                else:
                    class_name = [class_name[int(opt.target_class)]]

    end_time = time.time()
    test_label = []
    test_annot = []
    output_all = []
    t = time.time()
    for i, data_item in enumerate(data_loader):
        visual, target,  visualization_item, batch_size, annot = process_data_item(opt, data_item)
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
        data_time.update(time.time() - end_time)
        with torch.no_grad():
            if opt.task == 'annot' or opt.task == 'annot-reg':
                output, loss = run_model(opt, [visual, annot], model, criterion, i,print_attention=False)
            else:
                output, loss = run_model(opt, [visual, target], model, criterion, i,print_attention=False)
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
        if not np.isnan(acc):
            accuracies.update(acc, batch_size)

        batch_time.update(time.time() - end_time)
        end_time = time.time()
        test_label.append(target)
        test_annot.append(annot)
        output_all.append(output)
    print("Val loss: {:.4f}".format(losses.avg))
    print("Val acc: {:.4f}".format(accuracies.avg))
    if opt.task == 'annot':
        ac, auc_per_class, youden_per_class, best_threshold_acc_per_class = calculate_accuracy_cross_annot(torch.cat(output_all,dim=0),torch.cat(test_annot,dim=0),len(class_name))
        plot_annot_result(torch.cat(output_all,dim=0),torch.cat(test_annot,dim=0),len(class_name),epoch, class_name,ac,opt)
    elif opt.task == 'annot-reg':
        ac = calculate_accuracy_cross_annot_reg(torch.cat(output_all,dim=0),torch.cat(test_annot,dim=0),len(class_name))
        plot_regression_result(torch.cat(output_all,dim=0),torch.cat(test_annot,dim=0),len(class_name),epoch, class_name,ac,opt)
    else:
        ac = calculate_accuracy_cross(torch.cat(output_all,dim=0),torch.cat(test_label,dim=0),len(class_name))
    for i in range(len(class_name)):
        if opt.task == 'annot':
            print(f"{class_name[i]} | Acc@0.5: {ac[i]:.4f} | AUC: {auc_per_class[i]:.4f} | "
                    f"Youden: {youden_per_class[i]:.4f} | Acc@best: {best_threshold_acc_per_class[i]:.4f}")
        else:
            print(class_name[i],'=',ac[i])
        
    # save_file_path = os.path.join(opt.ckpt_path, 'save_{}.pth'.format(epoch))
    # states = {
    #     'epoch': epoch + 1,
    #     'state_dict': model.state_dict(),
    #     'optimizer': optimizer.state_dict(),
    # }
    # if epoch % 1 ==0:
    #     torch.save(states, save_file_path)
    print('validation_time=',(time.time()-t)/60,' min')
    if opt.task == 'annot':
        return epoch,accuracies.avg, ac, losses.avg, auc_per_class, youden_per_class, best_threshold_acc_per_class
    else:
        return epoch,accuracies.avg,ac, losses.avg

def val_total(epoch, data_loader, model, criterion, opt):
    print("# ---------------------------------------------------------------------- #")
    print('Validation at epoch {}'.format(epoch))
    model.eval()

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    accuracies = AverageMeter()

    end_time = time.time()
    test_label = []
    output_all = []
    for i, data_item in enumerate(data_loader):
        print(i)
        visual, target,  visualization_item, batch_size = process_data_item(opt, data_item)
        data_time.update(time.time() - end_time)
        with torch.no_grad():
            output, loss = run_model(opt, [visual, target], model, criterion, i,print_attention=False)

        acc = calculate_accuracy(output, target)

        losses.update(loss.item(), batch_size)
        accuracies.update(acc, batch_size)
        batch_time.update(time.time() - end_time)
        end_time = time.time()
        test_label.append(target)
        output_all.append(output)
    print("Val loss: {:.4f}".format(losses.avg))
    print("Val acc: {:.4f}".format(accuracies.avg))


    return epoch,accuracies.avg,test_label,output_all
