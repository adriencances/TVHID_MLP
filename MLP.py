import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class ClassificationMLP(nn.Module):
    def __init__(self, in_features=2048, nb_classes=5, nb_layers=3):
        super(ClassificationMLP, self).__init__()

        self.in_features = in_features
        self.nb_classes = nb_classes
        self.nb_layers = nb_layers

        feature_sizes = [self.in_features // 2**i for i in range(self.nb_layers)] + [self.nb_classes]
        self.layers = nn.ModuleList([nn.Linear(feature_sizes[i], feature_sizes[i + 1]) for i in range(self.nb_layers)])

    def forward(self, features1, features2):
        x = torch.cat((features1, features2), dim=1)
        for layer in self.layers[:-1]:
            x = F.relu(layer(x))
        x = self.layers[-1](x)
        # x = F.softmax(x, dim=1)
        return x


if __name__ == "__main__":
    model = ClassificationMLP()
    model.cuda()

    torch.manual_seed(0)
    input = torch.rand(8, 1024).cuda()

    output = model(input)
    print(input.shape)
    print(output.shape)
    print(output)
