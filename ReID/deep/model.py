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
    def __init__(self, model:str="resnet50", num_classes:int=1000, use_batch_norm:bool=True, force_descr_dim:int=-1):
        super(ReIDModel, self).__init__()

        if model == "resnet50":
            model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
            self.base = nn.Sequential(*list(model.children())[:-1]) # avg-pool layer is still included (not removed)
            output_n = 2048
        elif model == "resnet18":
            model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
            self.base = nn.Sequential(*list(model.children())[:-1]) # avg-pool layer is still included (not removed)
            output_n = 512
        else:
            raise Exception("Please specify the model")
   
        #using bbneck idea, expressed in [1]
        self.use_batch_norm = use_batch_norm 
        if force_descr_dim > 0 and force_descr_dim != output_n:
            final_dim = force_descr_dim
            print("Forcing feature vector to size", final_dim)
            self.bottleneck = nn.Linear(output_n, final_dim, bias=False)
            self.bottleneck.apply(weights_init_classifier)
        else:
            final_dim = output_n
            self.bottleneck = None

        if self.use_batch_norm:
            self.batch_norm = nn.BatchNorm1d(final_dim)
            self.batch_norm.bias.requires_grad_(False)  # no shift
            self.batch_norm.apply(weights_init_kaiming)

        self.classifier = nn.Linear(final_dim, num_classes, bias=False)
        self.classifier.apply(weights_init_classifier)

    def forward(self, x):
        """
        Returns:
            (features_vector, class_scores):    during training
            (features_vector):                  during inference
        """
        x = self.base(x)                        # (b, final_dim, 1, 1)
        global_feat = x.view(x.size(0), -1)     # flatten to (b, final_dim)

        if self.bottleneck is not None:
            global_feat = self.bottleneck(global_feat)

        if self.use_batch_norm:
            feat = self.batch_norm(global_feat)
        else:
            feat = global_feat

        if self.training:
            cls_score = self.classifier(feat)
            return global_feat, cls_score        # global features for triplet loss
        else:
            return feat                         # if we are in inference we just want the (final) feature vector.