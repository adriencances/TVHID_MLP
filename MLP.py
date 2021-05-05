import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class ClassificationMLP(nn.Module):
    def __init__(self, in_features=1024, nb_classes=5):
        super(ClassificationMLP, self).__init__()

        self.in_features = in_features
        self.nb_classes = nb_classes

        self.fc1 = nn.Linear(self.in_features, 512)
        self.fc2 = nn.Linear(512, self.nb_classes)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.softmax(x, dim=1)
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
