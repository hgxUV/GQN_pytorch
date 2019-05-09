import torch
import torch.nn as nn

class Latent(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size):
        super(Latent, self).__init__()

        self.no_features = in_channels
        self.gaussian_params_conv = nn.Conv2d(in_channels, out_channels, kernel_size)

    def forward(self, input):

        gaussian_params = self.gaussian_params_conv(input)

        means = gaussian_params[:, :, :, 0:self.no_features]
        stds = gaussian_params[:, :, :, self.no_features:]
        stds = nn.Softplus(stds)

        gaussian_params = torch.cat((stds, means), -1)

        distributions = torch.distributions.Normal(loc=means, scale=means)
        latent = distributions.sample()

        return latent, gaussian_params


class ImageReconstruction(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, sigma):
        super(ImageReconstruction, self).__init__()

        self.sigma = sigma

        self.rgb_means_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size),
            nn.ReLU()
        )

    def forward(self, input):
        rgb_means_conv = self.RGB_means_conv(input)
        stds = torch.ones(rgb_means_conv.shape) * self.sigma
        dist =torch.distributions.Normal(loc=rgb_means_conv, scale=stds)
        x_pred = dist.sample()
        return x_pred

