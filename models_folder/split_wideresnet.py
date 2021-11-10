import math
import torch
import torch.nn as nn
from layers import conv_type
from models.builder import get_builder

try:
    from torch.hub import load_state_dict_from_url
except ImportError:
    from torch.utils.model_zoo import load_url as load_state_dict_from_url

model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
    'resnext50_32x4d': 'https://download.pytorch.org/models/resnext50_32x4d-7cdf4587.pth',
    'resnext101_32x8d': 'https://download.pytorch.org/models/resnext101_32x8d-8ba56ff5.pth',
    'wide_resnet50_2': 'https://download.pytorch.org/models/wide_resnet50_2-95faca4d.pth',
    'wide_resnet101_2': 'https://download.pytorch.org/models/wide_resnet101_2-32ee1156.pth',
}


class BasicBlock(nn.Module):
    def __init__(self, in_planes, out_planes, stride, dropRate=0.0):
        super(BasicBlock, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_planes)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_planes, out_planes, kernel_size=3, stride=1,
                               padding=1, bias=False)
        self.droprate = dropRate
        self.equalInOut = (in_planes == out_planes)
        self.convShortcut = (not self.equalInOut) and nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride,
                               padding=0, bias=False) or None
    def forward(self, x):
        if not self.equalInOut:
            x = self.relu1(self.bn1(x))
        else:
            out = self.relu1(self.bn1(x))
        out = self.relu2(self.bn2(self.conv1(out if self.equalInOut else x)))
        if self.droprate > 0:
            out = F.dropout(out, p=self.droprate, training=self.training)
        out = self.conv2(out)
        return torch.add(x if self.equalInOut else self.convShortcut(x), out)

class NetworkBlock(nn.Module):
    def __init__(self, nb_layers, in_planes, out_planes, block, stride, dropRate=0.0):
        super(NetworkBlock, self).__init__()
        self.layer = self._make_layer(block, in_planes, out_planes, nb_layers, stride, dropRate)
    def _make_layer(self, block, in_planes, out_planes, nb_layers, stride, dropRate):
        layers = []
        for i in range(int(nb_layers)):
            layers.append(block(i == 0 and in_planes or out_planes, out_planes, i == 0 and stride or 1, dropRate))
        return nn.Sequential(*layers)
    def forward(self, x):
        return self.layer(x)

class WideResNet(nn.Module):
    def __init__(self, depth, num_classes, widen_factor=1, dropRate=0.0):
        super(WideResNet, self).__init__()
        nChannels = [16, 16*widen_factor, 32*widen_factor, 64*widen_factor]
        assert((depth - 4) % 6 == 0)
        n = (depth - 4) / 6
        block = BasicBlock
        # 1st conv before any network block
        self.conv1 = nn.Conv2d(3, nChannels[0], kernel_size=3, stride=1,
                               padding=1, bias=False)
        # 1st block
        self.block1 = NetworkBlock(n, nChannels[0], nChannels[1], block, 1, dropRate)
        # 2nd block
        self.block2 = NetworkBlock(n, nChannels[1], nChannels[2], block, 2, dropRate)
        # 3rd block
        self.block3 = NetworkBlock(n, nChannels[2], nChannels[3], block, 2, dropRate)
        # global average pooling and classifier
        self.bn1 = nn.BatchNorm2d(nChannels[3])
        self.relu = nn.ReLU(inplace=True)
#         self.fc = nn.Linear(nChannels[3], num_classes)
        self.nChannels = nChannels[3]
        self.final_feat_dim = nChannels[3]

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.bias.data.zero_()
    def forward(self, x):
        out = self.conv1(x)
        out = self.block1(out)
        out = self.block2(out)
        out = self.block3(out)
        out = self.relu(self.bn1(out))
        out = F.avg_pool2d(out, 8)
        out = out.view(-1, self.nChannels)
        return out#self.fc(out)
    
class BasicBlock(nn.Module):

    def __init__(self, builder, in_planes, out_planes, stride=1, slim_factor=1):
        super(BasicBlock, self).__init__()
        self.bn1 = builder.batchnorm(math.ceil(in_planes * slim_factor))
        self.relu1 = nn.ReLU(inplace=True)
        self.conv1 = builder.conv3x3(math.ceil(in_planes * slim_factor), math.ceil(out_planes * slim_factor), stride, padding=1, bias=False) ## Avoid residual links
        ***self.relu = builder.activation()
        self.bn2 = builder.batchnorm(math.ceil(out_planes * slim_factor), last_bn=True)  ## Avoid residual links
        self.relu2 = nn.ReLU(inplace=True)
        self.conv2 = builder.conv3x3(math.ceil(out_planes * slim_factor),
                                     math.ceil(out_planes * slim_factor), stride=1, padding=1, bias=False)  ## Avoid residual links
        
        self.droprate = dropRate
        self.equalInOut = (in_planes == out_planes)
#         self.convShortcut = (not self.equalInOut) and nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, padding=0, bias=False) or None
        self.convShortcut = (not self.equalInOut) and nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, padding=0, bias=False) or None

    def forward(self, x):
        if not self.equalInOut:
            x = self.relu1(self.bn1(x))
        else:
            out = self.relu1(self.bn1(x))
        out = self.relu2(self.bn2(self.conv1(out if self.equalInOut else x)))
        if self.droprate > 0:
            out = F.dropout(out, p=self.droprate, training=self.training)
        out = self.conv2(out)
        return torch.add(x if self.equalInOut else self.convShortcut(x), out)
    

# ResNet {{{
class ResNet(nn.Module):
    def __init__(self,cfg, builder, block, layers, base_width=64):

        super(ResNet, self).__init__()
        self.inplanes = 64
        slim_factor = cfg.slim_factor
        if slim_factor < 1:
            cfg.logger.info('WARNING: You are using a slim network')

        self.base_width = base_width
        if self.base_width // 64 > 1:
            print(f"==> Using {self.base_width // 64}x wide model")


        self.conv1 = builder.conv7x7(3, math.ceil(64*slim_factor), stride=2, first_layer=True)

        self.bn1 = builder.batchnorm(math.ceil(64*slim_factor))
        self.relu = builder.activation()
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(builder, block, 64, layers[0], slim_factor=slim_factor)

        self.layer2 = self._make_layer(builder, block, 128, layers[1], stride=2, slim_factor=slim_factor)

        self.layer3 = self._make_layer(builder, block, 256, layers[2], stride=2, slim_factor=slim_factor)

        self.layer4 = self._make_layer(builder, block, 512, layers[3], stride=2, slim_factor=slim_factor)

        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = builder.linear(math.ceil(512 * block.expansion * slim_factor), cfg.num_cls, last_layer=True)


    def _make_layer(self, builder, block, planes, blocks, stride=1, slim_factor=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            dconv = builder.conv1x1(math.ceil(self.inplanes * slim_factor),
                                    math.ceil(planes * block.expansion * slim_factor), stride=stride) ## Going into a residual link
            dbn = builder.batchnorm(math.ceil(planes * block.expansion * slim_factor))
            if dbn is not None:
                downsample = nn.Sequential(dconv, dbn)
            else:
                downsample = dconv

        layers = []
        layers.append(block(builder, self.inplanes, planes, stride, downsample, base_width=self.base_width, slim_factor=slim_factor))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(builder, self.inplanes, planes, base_width=self.base_width,
                                slim_factor=slim_factor))

        return nn.Sequential(*layers)

    def forward(self, x):
        # features = []
        x = self.conv1(x)
        # features.append(x)

        if self.bn1 is not None:
            x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        # features.append(x)
        # print('resnet 1 ',torch.norm(x[:,:x.shape[1]//split]))
        # self.layer1[0].split = split
        # self.layer1[1].split = split
        x = self.layer1(x)
        # features.append(x)

        x = self.layer2(x)
        # features.append(x)
        x = self.layer3(x)
        # features.append(x)
        x = self.layer4(x)
        # features.append(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        x = x.view(x.size(0), -1)
        # features.append(x)
        return x

    
from collections import OrderedDict, namedtuple

class _IncompatibleKeys(namedtuple('IncompatibleKeys', ['missing_keys', 'unexpected_keys'])):
    def __repr__(self):
        if not self.missing_keys and not self.unexpected_keys:
            return '<All keys matched successfully>'
        return super(_IncompatibleKeys, self).__repr__()

    __str__ = __repr__

def load_state_dict(model, state_dict,
                    strict: bool = True):
    r"""Copies parameters and buffers from :attr:`state_dict` into
    this module and its descendants. If :attr:`strict` is ``True``, then
    the keys of :attr:`state_dict` must exactly match the keys returned
    by this module's :meth:`~torch.nn.Module.state_dict` function.
    Arguments:
        state_dict (dict): a dict containing parameters and
            persistent buffers.
        strict (bool, optional): whether to strictly enforce that the keys
            in :attr:`state_dict` match the keys returned by this module's
            :meth:`~torch.nn.Module.state_dict` function. Default: ``True``
    Returns:
        ``NamedTuple`` with ``missing_keys`` and ``unexpected_keys`` fields:
            * **missing_keys** is a list of str containing the missing keys
            * **unexpected_keys** is a list of str containing the unexpected keys
    """
    missing_keys = []
    unexpected_keys = []
    error_msgs = []

    # copy state_dict so _load_from_state_dict can modify it
    metadata = getattr(state_dict, '_metadata', None)
    state_dict = state_dict.copy()
    if metadata is not None:
        state_dict._metadata = metadata

    def load(module, prefix=''):
        local_metadata = {} if metadata is None else metadata.get(prefix[:-1], {})
        module._load_from_state_dict(
            state_dict, prefix, local_metadata, True, missing_keys, unexpected_keys, error_msgs)
        for name, child in module._modules.items():
            if child is not None and not (child.__class__.__name__ == 'SplitLinear' or child.__class__.__name__ == 'Linear'):
                load(child, prefix + name + '.')

    load(model)
    load = None  # break load->load reference cycle

    if strict:
        if len(unexpected_keys) > 0:
            error_msgs.insert(
                0, 'Unexpected key(s) in state_dict: {}. '.format(
                    ', '.join('"{}"'.format(k) for k in unexpected_keys)))
        if len(missing_keys) > 0:
            error_msgs.insert(
                0, 'Missing key(s) in state_dict: {}. '.format(
                    ', '.join('"{}"'.format(k) for k in missing_keys)))

    if len(error_msgs) > 0:
        raise RuntimeError('Error(s) in loading state_dict for {}:\n\t{}'.format(
            model.__class__.__name__, "\n\t".join(error_msgs)))
    return _IncompatibleKeys(missing_keys, unexpected_keys)

# ResNet }}}
def Split_ResNet18(cfg, progress=True):
    model = ResNet(cfg,get_builder(cfg), BasicBlock, [2, 2, 2, 2])
    if cfg.pretrained == 'imagenet':
        arch = 'resnet18'
        state_dict = load_state_dict_from_url(model_urls[arch],
                                              progress=progress)
        load_state_dict(model,state_dict,strict=False)
    return model

def Split_ResNet34(cfg, progress=True):
    model = ResNet(cfg,get_builder(cfg), BasicBlock, [3, 4, 6, 3])
    if cfg.pretrained == 'imagenet':
        arch = 'resnet34'
        state_dict = load_state_dict_from_url(model_urls[arch],
                                              progress=progress)
        load_state_dict(model,state_dict,strict=False)
    return model

def Split_ResNet50(cfg,progress=True):
    model = ResNet(cfg,get_builder(cfg), Bottleneck, [3, 4, 6, 3])
    if cfg.pretrained == 'imagenet':
        arch = 'resnet50'
        state_dict = load_state_dict_from_url(model_urls[arch],
                                              progress=progress)
        load_state_dict(model,state_dict,strict=False)
    return model


def Split_ResNet101(cfg,progress=True):
    model = ResNet(cfg,get_builder(cfg), Bottleneck, [3, 4, 23, 3])
    if cfg.pretrained == 'imagenet':
        arch = 'resnet101'
        state_dict = load_state_dict_from_url(model_urls[arch],
                                              progress=progress)
        load_state_dict(model,state_dict,strict=False)
    return model


# def WideResNet50_2(cfg,pretrained=False):
#     return ResNet(cfg,
#         get_builder(cfg), Bottleneck, [3, 4, 6, 3], base_width=64 * 2
#     )
#
#
# def WideResNet101_2(cfg,pretrained=False):
#     return ResNet(cfg,
#         get_builder(cfg), Bottleneck, [3, 4, 23, 3], base_width=64 * 2
#     )

