import pretrainedmodels
import torch.nn as nn
import torch
# from metrics import *

def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred    = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))

    return res

def get_pretrainedmodels(model_name='resnet18', num_outputs=None, pretrained=True, freeze_conv = True):
    pretrained = 'imagenet' if pretrained else None
    model = pretrainedmodels.__dict__[model_name](num_classes=1000,
                                                  pretrained=pretrained)
    # freeze layer 
    # if not freeze all conv , cuda memory is not enough
    # only unfreeze layer4
    if freeze_conv:
        for name, param in model.named_parameters():
            if not name.startswith('layer4'):
                param.requires_grad = False

    # only out put feature layer
    # feature = nn.Sequential(*list(model.children())[:6] )

    out_features = model.last_linear.in_features

    # model.last_linear = nn.Sequential(
    #             # nn.BatchNorm1d(out_features),
    #             nn.Dropout(0.3),
    #             nn.Linear(out_features, num_outputs),
    #             )
    # print(model)
    model.last_linear = nn.Linear(out_features, num_outputs)

    return model

def update_params(model):
    params_to_update = []
    for _, param in model.named_parameters():
        if param.requires_grad == True:
            params_to_update.append(param)
    return params_to_update

class LabelSmoothSoftmaxCE(nn.Module):
    def __init__(self,
                 lb_pos=0.9,
                 lb_neg=0.005,
                 reduction='mean',
                 lb_ignore=255,
                 ):
        super(LabelSmoothSoftmaxCE, self).__init__()
        self.lb_pos = lb_pos
        self.lb_neg = lb_neg
        self.reduction = reduction
        self.lb_ignore = lb_ignore
        self.log_softmax = nn.LogSoftmax(1)

    def forward(self, logits, label):
        logs = self.log_softmax(logits)
        ignore = label.data.cpu() == self.lb_ignore
        n_valid = (ignore == 0).sum()
        label[ignore] = 0
        lb_one_hot = logits.data.clone().zero_().scatter_(1, label.unsqueeze(1), 1)
        label = self.lb_pos * lb_one_hot + self.lb_neg * (1-lb_one_hot)
        # print(label)
        ignore = ignore.nonzero()
        _, M = ignore.size()
        a, *b = ignore.chunk(M, dim=1)
        label[[a, torch.arange(label.size(1)), *b]] = 0

        if self.reduction == 'mean':
            loss = -torch.sum(torch.sum(logs*label, dim=1)) / n_valid
        elif self.reduction == 'none':
            loss = -torch.sum(logs*label, dim=1)
        return loss