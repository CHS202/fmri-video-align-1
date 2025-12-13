import torch.nn as nn
# from models.vaanet import VAANet
from models.visual_stream import VisualStream
from models.visual_stream import CNN_3D
from models.visual_stream import VisualStream_VAA
# def generate_model(opt):
#     model = VAANet(
#         snippet_duration=opt.snippet_duration,
#         sample_size=opt.sample_size,
#         n_classes=opt.n_classes,
#         seq_len=opt.seq_len,
#         audio_embed_size=opt.audio_embed_size,
#         audio_n_segments=opt.audio_n_segments,
#         pretrained_resnet101_path=opt.resnet101_pretrained,
#     )
#     model = model.cuda()
#     return model, model.parameters()
# def generate_model(opt):
#     model = VisualStream(
#         snippet_duration=opt.snippet_duration,
#         sample_size=opt.sample_size,
#         n_classes=opt.n_classes,
#         seq_len=opt.seq_len,
#         pretrained_model_path=opt.model_pretrained,
#     )
#     model = model.cuda()
#     total_params = 0
#     for parameter in model.parameters():
#         if not parameter.requires_grad: continue
#         param = parameter.numel()
#         total_params += param
#     return model, model.parameters(),total_params

def generate_model(opt):
    if opt.network_choose == 'resnet_18':
        model = VisualStream(
                snippet_duration=opt.snippet_duration,
                sample_size=opt.sample_size,
                n_classes=opt.n_classes,
                seq_len=opt.seq_len,
                pretrained_model_path=opt.model_pretrained,
            )
    elif opt.network_choose == 'vaa':
        model = VisualStream_VAA(
            snippet_duration=opt.snippet_duration,
            sample_size=opt.sample_size,
            n_classes=opt.n_classes,
            seq_len=opt.seq_len,
            pretrained_model_path='/data/home/fukaicheng/pythonProject/ICLR2023/VAANet_copy/resnet-101-kinetics.pth',
        )
    else:
        model = CNN_3D(
            snippet_duration=opt.snippet_duration,
            sample_size=opt.sample_size,
            n_classes=opt.n_classes,
            seq_len=opt.seq_len,
            pretrained_model_path=opt.model_pretrained,
            network_choose = opt.network_choose
        )
    print('Trained network is: ',opt.network_choose)
    model = model.cuda()
    total_params = 0
    for parameter in model.parameters():
        if not parameter.requires_grad: continue
        param = parameter.numel()
        total_params += param
    return model, model.parameters(),total_params