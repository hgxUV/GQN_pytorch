import torch
import torch.nn as nn
import torch.nn.functional as F


class Tower(nn.Module):
    def __init__(self):
        super(Tower, self).__init__()

        self.layer1 = nn.Conv2d(3, 256, kernel_size=2, stride=2)
        self.layer2 = nn.Conv2d(256, 128, kernel_size=3, padding=1)
        self.layer3 = nn.Conv2d(384, 256, kernel_size=2, stride=2)
        self.layer4 = nn.Conv2d(256+7, 128, kernel_size=3, padding=1)
        self.layer5 = nn.Conv2d(391, 256, kernel_size=3, padding=1)
        self.layer6 = nn.Conv2d(256, 256, kernel_size=1)

    def forward(self, x, view):
        v1 = view.unsqueeze(-1)
        v2 = v1.expand((5, 7, 16))

        v3 = v2.unsqueeze(-1)
        v4 = v3.expand((5, 7, 16, 16))

        x = F.relu(self.layer1(x))
        y = F.relu(self.layer2(x))
        x = torch.cat([x, y], dim=1)

        x = F.relu(self.layer3(x))

        x = torch.cat([x, v4], dim=1)
        y = F.relu(self.layer4(x))
        x = torch.cat([x, y], dim=1)

        x = F.relu(self.layer5(x))
        x = F.relu(self.layer6(x))

        return x.sum(0)