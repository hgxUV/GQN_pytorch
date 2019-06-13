import torch.nn as nn

class ConvLSTM(nn.Module):

    def __init__(self, in_channels,  out_channels, kernel_size):
        super(ConvLSTM, self).__init__()

        self.forget_conv = nn.Conv2d(in_channels, out_channels, kernel_size, padding=2)
        self.input_conv = nn.Conv2d(in_channels, out_channels, kernel_size, padding=2)
        self.update_conv = nn .Conv2d(in_channels, out_channels, kernel_size, padding=2)
        self.output_conv = nn.Conv2d(in_channels, out_channels, kernel_size, padding=2)
        self.sigmoid = nn.Sigmoid()
        self.tanh = nn.Tanh()

    def forward(self, input, state):

        forget = self.sigmoid(self.forget_conv(input))
        updated_state = self.sigmoid(self.input_conv(input)) * self.tanh(self.update_conv(input))
        state = state * forget + updated_state
        output = self.tanh(state) * self.sigmoid(self.output_conv(input))
        return output, state