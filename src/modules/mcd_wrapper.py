import copy
import torch.nn as nn
from torchvision.models.feature_extraction import get_graph_node_names
from torchvision.models.feature_extraction import create_feature_extractor
from torch.nn import functional as F


class McdWrapper(nn.Module):

    def __init__(self, model):
        super().__init__()
        self.task = model.task
        self.f_extr = self.__extract_feats(model)
        self.classifier1 = model.classifier
        self.classifier2 = copy.deepcopy(model.classifier)

    @staticmethod
    def __extract_feats(model):
        train_nodes, eval_nodes = get_graph_node_names(model)
        index = [i for i, name in enumerate(train_nodes) if 'classifier' in name][0] - 1
        return_nodes = [train_nodes[index]]
        return create_feature_extractor(model, return_nodes=return_nodes)

    def __fw_cl1(self, feats, input_shape):
        out1 = self.classifier1(feats)
        out1 = F.interpolate(out1, size=input_shape, mode='bilinear', align_corners=False)
        return out1

    def __fw_cl2(self, feats, input_shape):
        out2 = self.classifier2(feats)
        out2 = F.interpolate(out2, size=input_shape, mode='bilinear', align_corners=False)
        return out2

    def forward(self, x, classifier1=False, classifier2=False):
        input_shape = x.shape[-2:]
        feats = list(self.f_extr(x).values())[0]
        if classifier1 and classifier2:
            return self.__fw_cl1(feats, input_shape), self.__fw_cl2(feats, input_shape)
        if classifier1:
            return self.__fw_cl1(feats, input_shape)
        if classifier2:
            return self.__fw_cl2(feats, input_shape)
        return feats
