from core.utils import AverageMeter, process_data_item, run_model, calculate_accuracy,calculate_accuracy_cross

import os
import time
import torch


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
        visual, target,  visualization_item, batch_size = process_data_item(opt, data_item)
        data_time.update(time.time() - end_time)
        with torch.no_grad():
            output, loss = run_model(opt, [visual, target], model, criterion, i,print_attention=False)

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
    return epoch,accuracies.avg

def val_epoch_class(epoch, data_loader, model, criterion, opt, optimizer):
    print("# ---------------------------------------------------------------------- #")
    print('Validation at epoch {}'.format(epoch))
    model.eval()

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    accuracies = AverageMeter()
    if opt.dataset_choose == 've8':
        class_name = ['Anger', 'Anticipation', 'Disgust', 'Fear', 'Joy', 'Sadness', 'Surprise', 'Trust']
    elif opt.dataset_choose == 'ek6':
        class_name = ['Anger', 'Disgust', 'Fear', 'Joy', 'Sadness', 'Surprise']
    elif opt.dataset_choose == 'rt':
        if opt.task == 'design':
            class_name = ['MODN', 'MUJI', 'SCAN', 'WABI']
        elif opt.task == 'space':
            class_name = ['SPACE_05', 'SPACE_06', 'SPACE_07', 'SPACE_08', 'SPACE_09', 'SPACE_10', 'SPACE_11', 'SPACE_12']
    end_time = time.time()
    test_label = []
    output_all = []
    t = time.time()
    for i, data_item in enumerate(data_loader):
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
    ac = calculate_accuracy_cross(torch.cat(output_all,dim=0),torch.cat(test_label,dim=0),len(class_name))
    for i in range(len(class_name)):
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
