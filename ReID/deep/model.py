from torch import nn
from torchvision import models
from torch.nn import functional as F


class ReIDModel(nn.Module):
    def __init__(self, args):
        super(ReIDModel, self).__init__()

        if args.model == "resnet50":
            model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
            self.base = nn.Sequential(*list(model.children())[:-2])
            outputn = 2048
        elif args.model == "resnet18":
            model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
            self.base = nn.Sequential(*list(model.children())[:-2])
            outputn = 512
        elif args.model == "alexnet":
            model = models.alexnet(weights=models.AlexNet_Weights.DEFAULT)
            self.base = nn.Sequential(*list(model.children())[:-2])
            outputn = 256
        else:
            raise Exception("Please specify the model")

        self.bn = nn.BatchNorm1d(outputn)
        self.classifier = nn.Linear(outputn, args.num_classes)

    def forward(self, x, feature_extraction=False):
        x = self.base(x)  # restituisce un 2048x7x7
        x = F.avg_pool2d(x, x.size()[2:])  # faccio una avgPool sul 7x7
        x = x.view(x.size(0), -1)  # e poi creo un unico feat_vector di 2048
        fv = self.bn(x)  # e normalizzo
        # feature vector

        if feature_extraction:
            return fv

        c = self.classifier(fv)
        # predicted class

        return fv, c