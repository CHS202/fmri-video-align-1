import torch
import torch.nn as nn
import torchvision
from models.resnet_A import pretrained_resnet18, pretrained_resnet18_1
from models.shufflenet import pretrained_shufflenet_v1, pretrained_shufflenet_v1_1
from models.shufflenetv2 import pretrained_shufflenet_v2, pretrained_shufflenet_v2_1
from models.squeezenet import pretrained_squeezenet, pretrained_squeezenet_1
from models.mobilenet import pretrained_mobilenet_v1, pretrained_mobilenet_v1_1
from models.mobilenetv2 import pretrained_mobilenet_v2, pretrained_mobilenet_v2_1
class VisualStream(nn.Module):
    def __init__(self,
                 snippet_duration,
                 sample_size,
                 n_classes,
                 seq_len,
                 pretrained_model_path):
        super(VisualStream, self).__init__()
        self.snippet_duration = snippet_duration
        self.sample_size = sample_size
        self.n_classes = n_classes
        self.seq_len = seq_len
        self.ft_begin_index = 5
        self.pretrained_model_path = pretrained_model_path

        self._init_norm_val()
        self._init_hyperparameters()
        self._init_encoder()
        self._init_attention_subnets()
        self.gamma = torch.nn.Parameter(torch.ones(5)*(1/5), requires_grad=True)
    def _init_norm_val(self):
        self.NORM_VALUE = 255.0
        self.MEAN = 100.0 / self.NORM_VALUE

    def _init_encoder(self):
        resnet, _ = pretrained_resnet18(snippet_duration=self.snippet_duration,
                                         sample_size=self.sample_size,
                                         n_classes=self.n_classes,
                                         ft_begin_index=self.ft_begin_index,
                                         pretrained_model_path=self.pretrained_model_path)


        children = list(resnet.children())
        self.resnet = nn.Sequential(*children[:-2])  # delete the last fc and the avgpool layer
    def _init_hyperparameters(self):
        self.hp = {
            'nc': 2048,
            'k': 512,
            'm': 16,
            'hw': 4
        }

    def _init_attention_subnets(self):
        self.fc = nn.Linear(self.hp['k'], self.n_classes)
        # add a projection layer to the last fc layer 
        # self.fc = nn.Sequential(
        #     nn.Linear(self.hp['k'], 2304), # Projection layer 128, 288 or 2304
        #     nn.Linear(2304, self.n_classes)   # Final classification layer
        # )

    def _init_params(self):
        for subnet in [self.conv0, self.sa_net, self.ta_net, self.cwa_net, self.fc]:
            if subnet is None:
                continue
            for m in subnet.modules():
                self._init_module(m)
        self.ta_net['fc'].bias.data.fill_(1.0)

    def _init_module(self, m):
        if isinstance(m, nn.BatchNorm1d):
            m.weight.data.fill_(1)
            m.bias.data.zero_()
        elif isinstance(m, nn.Conv1d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out')

    def input_process(self,input):
        input = input.transpose(0, 1).contiguous()  # input.shape=[seq_len, batch, 3, 16, 112, 112]
        input.div_(self.NORM_VALUE).sub_(self.MEAN)

        seq_len, batch, nc, snippet_duration, sample_size, _ = input.size()
        input = input.view(seq_len * batch, nc, snippet_duration, sample_size, sample_size)
        return  input

    def forward(self, input: torch.Tensor,test_svm=False):
        input = input.transpose(0, 1).contiguous()  # input.shape=[seq_len, batch, 3, 16, 112, 112]
        input.div_(self.NORM_VALUE).sub_(self.MEAN)

        seq_len, batch, nc, snippet_duration, sample_size, _ = input.size()
        input = input.view(seq_len * batch, nc, snippet_duration, sample_size, sample_size)
        output = self.resnet(input)
        output = torch.squeeze(output, dim=2)
        output = torch.flatten(output, start_dim=2)
        F = output


        ###test###
        F = torch.mean(F, dim=2)
        F = F.view(seq_len, batch, self.hp['k']).contiguous()
        F = F.permute(1, 2, 0).contiguous()
        fSCT = torch.mean(F, dim=2)
        alpha = 0
        beta = 0
        gamma = 0


        output = self.fc(fSCT)
        return output, alpha, beta, gamma,fSCT

class CNN_3D(nn.Module):
    def __init__(self,
                 snippet_duration,
                 sample_size,
                 n_classes,
                 seq_len,
                 pretrained_model_path,network_choose):
        super(CNN_3D, self).__init__()
        self.snippet_duration = snippet_duration
        self.sample_size = sample_size
        self.n_classes = n_classes
        self.seq_len = seq_len
        self.ft_begin_index = 5
        self.pretrained_model_path = pretrained_model_path
        self.network_choose = network_choose
        self._init_norm_val()
        self._init_hyperparameters()
        self._init_encoder()
        self._init_attention_subnets()
        # self._init_params()
    def _init_norm_val(self):
        self.NORM_VALUE = 255.0
        self.MEAN = 100.0 / self.NORM_VALUE

    def _init_encoder(self):
        if self.network_choose == 'shufflenet_v1':
            model = pretrained_shufflenet_v1(snippet_duration=self.snippet_duration,
                                        sample_size=self.sample_size,
                                        n_classes=self.n_classes,
                                        pretrained_model_path=self.pretrained_model_path)
            self.gamma = torch.nn.Parameter(torch.ones(4) * (1 / 4), requires_grad=True)
        elif self.network_choose == 'shufflenet_v2':
            model = pretrained_shufflenet_v2(snippet_duration=self.snippet_duration,
                                    sample_size=self.sample_size,
                                    n_classes=self.n_classes,
                                    pretrained_model_path=self.pretrained_model_path)
            self.gamma = torch.nn.Parameter(torch.ones(3) * (1 / 3), requires_grad=True)
        elif self.network_choose == 'squeezenet':
            model = pretrained_squeezenet(snippet_duration=self.snippet_duration,
                                        sample_size=self.sample_size,
                                        n_classes=self.n_classes,
                                        pretrained_model_path=self.pretrained_model_path)
            self.gamma = torch.nn.Parameter(torch.ones(5) * (1 / 5), requires_grad=True)
        elif self.network_choose == 'mobilenet_v1':
            model = pretrained_mobilenet_v1(snippet_duration=self.snippet_duration,
                                        sample_size=self.sample_size,
                                        n_classes=self.n_classes,
                                        pretrained_model_path=self.pretrained_model_path)
            self.gamma = torch.nn.Parameter(torch.ones(6) * (1 / 6), requires_grad=True)
        elif self.network_choose == 'mobilenet_v2':
            model = pretrained_mobilenet_v2(snippet_duration=self.snippet_duration,
                                            sample_size=self.sample_size,
                                            n_classes=self.n_classes,
                                            pretrained_model_path=self.pretrained_model_path)
            self.gamma = torch.nn.Parameter(torch.ones(3) * (1 / 3), requires_grad=True)
        self.CNN = model

    def _init_hyperparameters(self):
        self.hp = {
            'nc': 2048,
            'k': 512,
            'm': 16,
            'hw': 4
        }

    def _init_attention_subnets(self):
        self.fc = nn.Linear(self.hp['k'], self.n_classes)

    def _init_params(self):
        for subnet in [self.conv0, self.sa_net, self.ta_net, self.cwa_net, self.fc]:
            if subnet is None:
                continue
            for m in subnet.modules():
                self._init_module(m)
        self.ta_net['fc'].bias.data.fill_(1.0)

    def _init_module(self, m):
        if isinstance(m, nn.BatchNorm1d):
            m.weight.data.fill_(1)
            m.bias.data.zero_()
        elif isinstance(m, nn.Conv1d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out')

    def input_process(self,input):
        input = input.transpose(0, 1).contiguous()  # input.shape=[seq_len, batch, 3, 16, 112, 112]
        input.div_(self.NORM_VALUE).sub_(self.MEAN)

        seq_len, batch, nc, snippet_duration, sample_size, _ = input.size()
        input = input.view(seq_len * batch, nc, snippet_duration, sample_size, sample_size)
        return  input

    def forward(self, input: torch.Tensor,test_svm=False):
        input = input.transpose(0, 1).contiguous()  # input.shape=[seq_len, batch, 3, 16, 112, 112]
        input.div_(self.NORM_VALUE).sub_(self.MEAN)

        seq_len, batch, nc, snippet_duration, sample_size, _ = input.size()
        input = input.view(seq_len * batch, nc, snippet_duration, sample_size, sample_size)
        output = self.CNN(input,test_svm)
        F = output
        if test_svm == False:
            F = F.view(seq_len, batch, self.n_classes).contiguous()
            F = F.permute(1, 2, 0).contiguous()
            output = torch.mean(F, dim=2)
            alpha = 0
            beta = 0
            gamma = 0
            return output, alpha, beta, gamma,output
        else:
            if self.network_choose == 'shufflenet_v1':
                F = F.view(seq_len, batch, 1440).contiguous()
                F = F.permute(1, 2, 0).contiguous()
                output = torch.mean(F, dim=2)
                alpha = 0
                beta = 0
                gamma = 0
                return output, alpha, beta, gamma, output
            elif self.network_choose == 'mobilenet_v1':
                F = F.view(seq_len, batch, 2048).contiguous()
                F = F.permute(1, 2, 0).contiguous()
                output = torch.mean(F, dim=2)
                alpha = 0
                beta = 0
                gamma = 0
                return output, alpha, beta, gamma, output


class VisualStream_VAA(nn.Module):
    def __init__(self,
                 snippet_duration,
                 sample_size,
                 n_classes,
                 seq_len,
                 pretrained_model_path):
        super(VisualStream_VAA, self).__init__()
        self.snippet_duration = snippet_duration
        self.sample_size = sample_size
        self.n_classes = n_classes
        self.seq_len = seq_len
        self.ft_begin_index = 5
        self.pretrained_model_path = pretrained_model_path

        self._init_norm_val()
        self._init_hyperparameters()
        self._init_encoder()
        self._init_attention_subnets()
        # self._init_params()
        self.gamma = torch.nn.Parameter(torch.ones(3)*(1/3), requires_grad=True)
    def _init_norm_val(self):
        self.NORM_VALUE = 255.0
        self.MEAN = 100.0 / self.NORM_VALUE

    def _init_encoder(self):
        resnet, _ = pretrained_resnet101(snippet_duration=self.snippet_duration,
                                         sample_size=self.sample_size,
                                         n_classes=self.n_classes,
                                         ft_begin_index=self.ft_begin_index,
                                         pretrained_model_path=self.pretrained_model_path)


        children = list(resnet.children())
        self.resnet = nn.Sequential(*children[:-2])  # delete the last fc and the avgpool layer
        for param in self.resnet.parameters():
            param.requires_grad = False
    def _init_hyperparameters(self):
        self.hp = {
            'nc': 2048,
            'k': 512,
            'm': 16,
            'hw': 4
        }

    def _init_attention_subnets(self):
        self.conv0 = nn.Sequential(
            *[nn.Conv1d(self.hp['nc'], self.hp['k'], 1, bias=True),
              nn.BatchNorm1d(self.hp['k']),
              nn.ReLU()])

        self.sa_net = nn.ModuleDict({
            'conv': nn.Sequential(
                nn.Conv1d(self.hp['k'], 1, 1, bias=False),
                nn.BatchNorm1d(1),
                nn.Tanh(),
            ),
            'fc': nn.Linear(self.hp['m'], self.hp['m'], bias=False),
            'softmax': nn.Softmax(dim=1)
        })

        self.ta_net = nn.ModuleDict({
            'conv': nn.Sequential(
                nn.Conv1d(self.hp['k'], 1, 1, bias=False),
                nn.BatchNorm1d(1),
                nn.Tanh(),
            ),
            'fc': nn.Linear(self.seq_len, self.seq_len, bias=True),
            'relu': nn.ReLU()
        })

        self.cwa_net = nn.ModuleDict({
            'conv': nn.Sequential(
                nn.Conv1d(self.hp['m'], 1, 1, bias=False),
                nn.BatchNorm1d(1),
                nn.Tanh(),
            ),
            'fc': nn.Linear(self.hp['k'], self.hp['k'], bias=False),
            'softmax': nn.Softmax(dim=1)
        })

        self.fc = nn.Linear(self.hp['k'], self.n_classes)

    def _init_params(self):
        for subnet in [self.conv0, self.sa_net, self.ta_net, self.cwa_net, self.fc]:
            if subnet is None:
                continue
            for m in subnet.modules():
                self._init_module(m)
        self.ta_net['fc'].bias.data.fill_(1.0)

    def _init_module(self, m):
        if isinstance(m, nn.BatchNorm1d):
            m.weight.data.fill_(1)
            m.bias.data.zero_()
        elif isinstance(m, nn.Conv1d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out')

    def forward(self, input: torch.Tensor,vaa=False):
        input = input.transpose(0, 1).contiguous()  # input.shape=[seq_len, batch, 3, 16, 112, 112]
        input.div_(self.NORM_VALUE).sub_(self.MEAN)

        seq_len, batch, nc, snippet_duration, sample_size, _ = input.size()
        input = input.view(seq_len * batch, nc, snippet_duration, sample_size, sample_size)
        with torch.no_grad():
            output = self.resnet(input)
            output = torch.squeeze(output, dim=2)
            output = torch.flatten(output, start_dim=2)
        F = self.conv0(output)  # [B x 512 x 16]

        Hs = self.sa_net['conv'](F)
        Hs = torch.squeeze(Hs, dim=1)
        Hs = self.sa_net['fc'](Hs)
        As = self.sa_net['softmax'](Hs)
        As = torch.mul(As, self.hp['m'])
        alpha = As.view(seq_len, batch, self.hp['m'])

        fS = torch.mul(F, torch.unsqueeze(As, dim=1).repeat(1, self.hp['k'], 1))
        mark1 = fS
        G = fS.transpose(1, 2).contiguous()
        Hc = self.cwa_net['conv'](G)
        Hc = torch.squeeze(Hc, dim=1)
        Hc = self.cwa_net['fc'](Hc)
        Ac = self.cwa_net['softmax'](Hc)
        Ac = torch.mul(Ac, self.hp['k'])
        beta = Ac.view(seq_len, batch, self.hp['k'])

        fSC = torch.mul(fS, torch.unsqueeze(Ac, dim=2).repeat(1, 1, self.hp['m']))
        mark2 = fSC
        fSC = torch.mean(fSC, dim=2)
        fSC = fSC.view(seq_len, batch, self.hp['k']).contiguous()
        fSC = fSC.permute(1, 2, 0).contiguous()

        Ht = self.ta_net['conv'](fSC)
        Ht = torch.squeeze(Ht, dim=1)
        Ht = self.ta_net['fc'](Ht)
        At = self.ta_net['relu'](Ht)
        gamma = At.view(batch, seq_len)

        fSCT = torch.mul(fSC, torch.unsqueeze(At, dim=1).repeat(1, self.hp['k'], 1))
        fSCT = torch.mean(fSCT, dim=2)
        mark3 = fSCT
        ###test###
        output = self.fc(fSCT)
        if vaa == False:
            return output, alpha, beta, gamma,fSCT
        else:
            return mark1,mark2,mark3