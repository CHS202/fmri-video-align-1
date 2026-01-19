'''MobileNet in PyTorch.

See the paper "MobileNets: Efficient Convolutional Neural Networks for Mobile Vision Applications"
for more details.
'''
import torch
import torch.nn as nn
import torch.nn.functional as F


def conv_bn(inp, oup, stride):
    return nn.Sequential(
        nn.Conv3d(inp, oup, kernel_size=3, stride=stride, padding=(1,1,1), bias=False),
        nn.BatchNorm3d(oup),
        nn.ReLU(inplace=True)
    )


class Block(nn.Module):
    '''Depthwise conv + Pointwise conv'''
    def __init__(self, in_planes, out_planes, stride=1):
        super(Block, self).__init__()
        self.conv1 = nn.Conv3d(in_planes, in_planes, kernel_size=3, stride=stride, padding=1, groups=in_planes, bias=False)
        self.bn1 = nn.BatchNorm3d(in_planes)
        self.conv2 = nn.Conv3d(in_planes, out_planes, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn2 = nn.BatchNorm3d(out_planes)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        return out


class MobileNet(nn.Module):
    def __init__(self, num_classes=600, sample_size=224, width_mult=1.):
        super(MobileNet, self).__init__()

        input_channel = 32
        last_channel = 1024
        input_channel = int(input_channel * width_mult)
        last_channel = int(last_channel * width_mult)
        cfg = [
        # c, n, s
        [64,   1, (2,2,2)],
        [128,  2, (2,2,2)],
        [256,  2, (2,2,2)],
        [512,  6, (2,2,2)],
        [1024, 2, (1,1,1)],
        ]

        self.features = [conv_bn(3, input_channel, (1,2,2))]
        # building inverted residual blocks
        for c, n, s in cfg:
            output_channel = int(c * width_mult)
            for i in range(n):
                stride = s if i == 0 else 1
                self.features.append(Block(input_channel, output_channel, stride))
                input_channel = output_channel
        # make it nn.Sequential
        self.features = nn.Sequential(*self.features)

        # building classifier
        self.classifier = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(last_channel, num_classes),
        )


    def forward(self, x,test_svm=False):
        x = self.features(x)
        x = F.avg_pool3d(x, x.data.size()[-3:])
        x = x.view(x.size(0), -1)
        if test_svm == False:
            x = self.classifier(x)
        return x


def get_fine_tuning_parameters(model, ft_portion):
    if ft_portion == "complete":
        return model.parameters()

    elif ft_portion == "last_layer":
        ft_module_names = []
        ft_module_names.append('classifier')

        parameters = []
        for k, v in model.named_parameters():
            for ft_module in ft_module_names:
                if ft_module in k:
                    parameters.append({'params': v})
                    break
            else:
                parameters.append({'params': v, 'lr': 0.0})
        return parameters

    else:
        raise ValueError("Unsupported ft_portion: 'complete' or 'last_layer' expected")
    

def get_model(**kwargs):
    """
    Returns the model.
    """
    model = MobileNet(**kwargs)
    return model

def pretrained_mobilenet_v1(snippet_duration: int,
                         sample_size: int,
                         n_classes,
                         pretrained_model_path):
    n_finetune_classes = 600
    model = get_model(num_classes=600, sample_size = sample_size, width_mult=2.0)
    model = model.cuda()
    print('Loading pretrained 3D mobilenet_v1 {}'.format(pretrained_model_path))
    pretrain = torch.load(pretrained_model_path)
    from collections import OrderedDict
    new_state_dict = OrderedDict()
    old_state_dict = pretrain['state_dict']
    for name in old_state_dict:
        new_name = name[7:]
        new_state_dict[new_name] = old_state_dict[name]
    model.load_state_dict(new_state_dict)
    # ---------------------------------------------------------------- #
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, n_classes)
    model.classifier = model.classifier.cuda()
    return model

def pretrained_mobilenet_v1_1(snippet_duration: int,
                         sample_size: int,
                         n_classes,
                         pretrained_model_path):
    n_finetune_classes = 600
    model = get_model(num_classes=600, sample_size = sample_size, width_mult=2.0)
    model = model.cuda()
    print('Loading pretrained 3D mobilenet_v1 {}'.format(pretrained_model_path))
    pretrain = torch.load(pretrained_model_path)
    from collections import OrderedDict
    new_state_dict = OrderedDict()
    old_state_dict = pretrain['state_dict']
    for name in old_state_dict:
        new_name = name[7:]
        new_state_dict[new_name] = old_state_dict[name]
    model.load_state_dict(new_state_dict)
    # ---------------------------------------------------------------- #
    # --- MODIFICATION START ---
    # 1. Get the number of input features from the original fc layer.
    in_features = model.classifier[1].in_features
    
    # 2. Define the new projection head output size.
    projection_dim = 2304 # 128, 288
    
    # 3. Replace the original fc layer with a new nn.Sequential module.
    #    This new module contains:
    #    - A projection layer (Linear -> ReLU -> Linear) to create 128-dim embeddings.
    #    - The final classification layer that maps the embeddings to the number of classes.
    model.classifier = nn.Sequential(
        nn.Dropout(0.2),
        nn.Linear(in_features, projection_dim), # Projection layer
        nn.Linear(projection_dim, n_classes)   # Final classification layer
    )
    # --- MODIFICATION END ---
    model.classifier = model.classifier.cuda()
    return model
