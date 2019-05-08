import torch
import torch.nn as nn
import torch.nn.functional as func


class Tower():
    def __init__(self):
        self.layer1 = nn.Sequential(
            nn.Conv2d(3, 256, kernel_size=2, stride=2),
            nn.ReLU()
        )

        self.layer2 = nn.Sequential(
            nn.Conv2d(256, 128, kernel_size=3, padding=1),
            nn.ReLU()
        )

        self.layer3 = nn.Sequential(
            nn.Conv2d(384, 256, kernel_size=2, stride=2),
            nn.ReLU()
        )

        self.layer4 = nn.Sequential(
            nn.Conv2d(256+7, 128, kernel_size=3, padding=1),
            nn.ReLU()
        )

        self.layer5 = nn.Sequential(
            nn.Conv2d(391, 256, kernel_size=3, padding=1),
            nn.ReLU()
        )

        self.layer6 = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=1),
            nn.ReLU()
        )

    def forward(self, x, view):
        v1 = view.unsqueeze(-1)
        v2 = v1.expand((5, 7, 16))

        v3 = v2.unsqueeze(-1)
        v4 = v3.expand((5, 7, 16, 16))

        x = self.layer1(x)
        y = self.layer2(x)
        x = torch.cat([x, y], dim=1)

        x = self.layer3(x)

        x = torch.cat([x, v4], dim=1)
        y = self.layer4(x)
        x = torch.cat([x, y], dim=1)

        x = self.layer5(x)
        x = self.layer6(x)

        return x.sum(0)