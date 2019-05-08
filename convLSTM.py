import torch.nn as nn

class Net(nn.Module):

    def __init__(self, in_channels,  out_channels, kernel_size):
        super(Net, self).__init__()

        self.forget_conv = nn.Conv2d(in_channels, out_channels, kernel_size)
        self.input_conv = nn.Conv2d(in_channels, out_channels, kernel_size)
        self.update_conv = nn.Conv2d(in_channels, out_channels, kernel_size)
        self.output_conv = nn.Conv2d(in_channels, out_channels, kernel_size)

    def forward(self, input, state):

        forget = nn.Sigmoid(self.forget_conv(input))
        updated_state = nn.Sigmoid(self.input_conv(input)) * nn.Tanh(self.update_conv(input))
        state = state * forget + updated_state
        output = nn.Tanh(state) * nn.Sigmoid(self.output_conv(input))
        return output, state