from __future__ import absolute_import

from torch import nn
from torch.nn import functional as F
import torch
from torch.nn import init
from torch.autograd import Variable
from torchvision import models


class SELayer(nn.Module):
    # E' la prima parte del blocco SEA dove simula l'autoencoder
    def __init__(self, channel, reduction=16):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


class Classifier(nn.Module):
    # È tutta la FSR, con dropout e classificazione con tanti neuroni quanti ID
    # ma viene usato anche dopo la SEA, con dropout=0
    def __init__(self, num_feature=1024, dropout=0.25, num_classes=0):
        super(Classifier, self).__init__()
        self.dropout = dropout
        if dropout > 0:
            self.drop = nn.Dropout(dropout)
        if num_classes > 0:
            self.classifier = nn.Linear(num_feature, num_classes)
            init.normal_(self.classifier.weight, std=0.001)
            init.constant_(self.classifier.bias, 0)

    def forward(self, x, output_feature=None):

        if self.dropout > 0:
            x = self.drop(x)
        x = self.classifier(x)

        return x


def softmax(input, dim=1):
    transposed_input = input.transpose(dim, len(input.size()) - 1)
    softmaxed_output = F.softmax(transposed_input.contiguous().view(-1, transposed_input.size(-1)), dim=-1)
    return softmaxed_output.view(*transposed_input.size()).transpose(dim, len(input.size()) - 1)


class CapsuleLayer(nn.Module):
    # Rappresenta un layer di capsule
    def __init__(self, num_capsules, num_route_nodes, in_channels, out_channels, kernel_size=None, stride=None,
                 num_iterations=None):
        super(CapsuleLayer, self).__init__()
        # Capsule del layer corrente
        self.num_route_nodes = num_route_nodes

        self.num_iterations = num_iterations
        # Capsule del prossimo layer
        self.num_capsules = num_capsules

        if num_route_nodes != -1:
            # Se devo fare il RbA, istanzia i parametri delle matrici
            self.route_weights = nn.Parameter(torch.randn(num_capsules, num_route_nodes, in_channels, out_channels))
        else:
            # Se non devo fare il RbA (come tra i due P-Caps) fai delle semplici conv per reshapare
            self.capsules = nn.ModuleList(
                [nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=0) for _ in
                 range(num_capsules)])

    def squash(self, tensor, dim=-1):
        squared_norm = (tensor ** 2).sum(dim=dim, keepdim=True)
        scale = squared_norm / (1 + squared_norm)
        return scale * tensor / torch.sqrt(squared_norm)

    def forward(self, x):
        if self.num_route_nodes != -1:
            # Esegui il RbA
            priors = x[None, :, :, None, :] @ self.route_weights[:, None, :, :, :]

            logits = Variable(torch.zeros(*priors.size())).cuda()
            # Faccio le iterazioni del RbA
            for i in range(self.num_iterations):
                probs = softmax(logits, dim=2)
                outputs = self.squash((probs * priors).sum(dim=2, keepdim=True))

                if i != self.num_iterations - 1:
                    delta_logits = (priors * outputs).sum(dim=-1, keepdim=True)
                    logits = logits + delta_logits
        else:
            # Trasforma il 7x7x1024 in tante capsule
            outputs = [capsule(x).view(x.size(0), -1, 1) for capsule in self.capsules]
            # Poi concatena in modo diverso le capsule facendo il reshape e passa al secondo P-Caps
            outputs = torch.cat(outputs, dim=-1)
            outputs = self.squash(outputs)

        return outputs


class DenseNet(nn.Module):
    def __init__(self, depth=121, num_feature=1024, num_classes=632, num_iteration = 3):

        super(DenseNet, self).__init__()
        self.depth = depth
        self.base = models.densenet121(pretrained=True)
        self.seblock = SELayer(channel=num_feature, reduction=16)
        self.classifer1 = Classifier(num_feature=num_feature, dropout=0.25, num_classes=num_classes)
        self.classifer2 = Classifier(num_feature=num_feature, dropout=0, num_classes=num_classes)

        self.primary_capsules = CapsuleLayer(num_capsules=8, num_route_nodes=-1, in_channels=num_feature, out_channels=32,
                                             kernel_size=2, stride=2, num_iterations = num_iteration)
        self.digit_capsules = CapsuleLayer(num_capsules=num_classes, num_route_nodes=32 * 3 * 3, in_channels=8,
                                           out_channels=24, num_iterations = num_iteration)


    def forward(self, x, output_feature=None):
        # fa la densenet
        x = self.base.features(x)
        # GlobalAveragePooling
        y = F.avg_pool2d(x, x.size()[2:])
        # Flattening
        y = y.view(y.size(0), -1)
        # TODO: la forward la fa qua
        if output_feature == 'pool5':
            # Se devo buttarle fuori, allora normalizza il vettore e dallo in output
            # E' il featur vector che si usa nell'inferenza
            y = F.normalize(y)
            return y

        # FSR
        y = self.classifer1(y)

        # SEA
        y2 = self.seblock(x)
        y2 = F.avg_pool2d(y2, y2.size()[2:])
        y2 = y2.view(y2.size(0), -1)
        y2 = self.classifer2(y2)

        # Capsule
        z = self.primary_capsules(x)
        z = self.digit_capsules(z).squeeze().transpose(0, 1)

        # Nromalizza le classi credo
        classes = (z ** 2).sum(dim=-1) ** 0.5
        classes = F.softmax(classes, dim=-1)

        return classes, y, y2