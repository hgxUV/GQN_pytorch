import torch
import torch.nn as nn
import torch.nn.functional as F

def loss(output_image, target_image, priors, posteriors):
    model_loss = nn.MSELoss()(output_image, target_image)
    dist_loss = distribution_loss(priors, posteriors)
    #regularization_loss = tf.scalar_mul(1000, tf.losses.get_regularization_loss())
    total_loss = model_loss + dist_loss
    return total_loss, model_loss, dist_loss


def distribution_loss(prior, posterior, no_features):
        distritution = lambda x: torch.distributions.Normal(loc=x[:, :, :, 0:no_features],
                                                         scale=x[:, :, :, no_features:])
        prior = distritution(prior)
        posterior = distritution(posterior)
        dist_loss = F.kl_div(posterior, prior)
        dist_loss = dist_loss.mean()
        return dist_loss

def get_pixel_std(sigma_i, sigma_f, sigma_n, global_step):
    sigma = torch.max(sigma_f + (sigma_i - sigma_f)*(1 - global_step / sigma_n), sigma_f)
    return sigma / torch.tensor([255.0])


class Latent(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size):
        super(Latent, self).__init__()

        self.no_features = in_channels
        self.gaussian_params_conv = nn.Conv2d(in_channels, out_channels*2, kernel_size, padding=2)
        self.softplus = nn.Softplus()

    def forward(self, input):

        gaussian_params = self.gaussian_params_conv(input)

        means = gaussian_params[:, 0:self.no_features, ...]
        stds = gaussian_params[:, self.no_features:, ...]
        stds = self.softplus(stds)

        gaussian_params = torch.cat((stds, means), 1)

        distributions = torch.distributions.Normal(loc=means, scale=means)
        latent = distributions.sample()

        return latent, gaussian_params


class ImageReconstruction(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size):
        super(ImageReconstruction, self).__init__()

        self.rgb_means_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, padding=2),
            nn.ReLU()
        )

    def forward(self, input, sigma):
        rgb_means_conv = self.rgb_means_conv(input)
        stds = torch.ones(rgb_means_conv.shape) * sigma
        dist =torch.distributions.Normal(loc=rgb_means_conv, scale=stds)
        x_pred = dist.sample()
        return x_pred

