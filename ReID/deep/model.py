from torch import nn
from torchvision import models
from torch.nn import functional as F

def weights_init_kaiming(m):
    """
    Code from https://github.com/michuanhaohao/reid-strong-baseline/blob/master/modeling/baseline.py
    """
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.kaiming_normal_(m.weight, a=0, mode='fan_out')
        nn.init.constant_(m.bias, 0.0)
    elif classname.find('Conv') != -1:
        nn.init.kaiming_normal_(m.weight, a=0, mode='fan_in')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.0)
    elif classname.find('BatchNorm') != -1:
        if m.affine:
            nn.init.constant_(m.weight, 1.0)
            nn.init.constant_(m.bias, 0.0)


def weights_init_classifier(m):
    """
    Code from https://github.com/michuanhaohao/reid-strong-baseline/blob/master/modeling/baseline.py
    """
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.normal_(m.weight, std=0.001)
        if m.bias:
            nn.init.constant_(m.bias, 0.0)

class ReIDModel(nn.Module):
    def __init__(self, model:str="resnet50", num_classes:int=1000, use_bbneck:bool=True):
        super(ReIDModel, self).__init__()

        if model == "resnet50":
            model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
            self.base = nn.Sequential(*list(model.children())[:-2]) # avg-pool layer is still included (not removed)
            output_n = 2048
        elif model == "resnet18":
            model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
            self.base = nn.Sequential(*list(model.children())[:-2]) # avg-pool layer is still included (not removed)
            output_n = 512
        else:
            raise Exception("Please specify the model")
   
        #using bbneck idea, expressed in [1]
        self.use_bbneck = use_bbneck 

        if not self.use_bbneck:
            self.classifier = nn.Linear(output_n, num_classes)
        else:
            self.bottleneck = nn.BatchNorm1d(output_n)
            self.bottleneck.bias.requires_grad_(False)  # no shift
            self.classifier = nn.Linear(output_n, num_classes, bias=False)
            self.bottleneck.apply(weights_init_kaiming)
            self.classifier.apply(weights_init_classifier)

    def forward(self, x):
        """
        Returns:
            (features_vector, class_scores):    during training
            (features_vector):                  during inference
        """
        x = self.base(x)                        # (b, output_n, 7, 7)
        x = F.avg_pool2d(x, x.size()[2:])       # (b, output_n, 1, 1)
        global_feat = x.view(x.size(0), -1)     # flatten to (b, output_n)

        if self.use_bbneck:
            feat = self.bottleneck(global_feat)
        else:
            feat = global_feat

        if self.training:
            cls_score = self.classifier(feat)
            return global_feat, cls_score        # global features for triplet loss
        else:
            return feat                         # if we are in inference we just want the (final) feature vector.