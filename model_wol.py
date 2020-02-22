import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable

import sys

class Net(nn.Module):

    def __init__(self, cuda_enabled = True):
        super(Net, self).__init__()

        self.cuda_enabled = cuda_enabled

        # cnn 1

        self.conv1_input_size = 1
        self.conv1_output_channel = 8
        self.conv1_kernel_size = (2, 2)
        self.conv1_stride = (1, 1)
        self.conv1 = nn.Conv2d(
            self.conv1_input_size,
            self.conv1_output_channel,
            self.conv1_kernel_size,
            self.conv1_stride
        )

        self.conv2_input_size = 8
        self.conv2_output_channel = 16
        self.conv2_kernel_size = (2, 2)
        self.conv2_stride = (1, 1)
        self.conv2 = nn.Conv2d(
            self.conv2_input_size,
            self.conv2_output_channel,
            self.conv2_kernel_size,
            self.conv2_stride
        )

        self.maxpool1_kernel_size = (2, 2)
        self.maxpool1_stride = (2, 2)
        self.maxpool1 = nn.MaxPool2d(
            self.maxpool1_kernel_size,
            self.maxpool1_stride
        )

        # cnn 2

        self.conv3_input_size = 16
        self.conv3_output_channel = 24
        self.conv3_kernel_size = (2, 2)
        self.conv3_stride = (1, 1)
        self.conv3 = nn.Conv2d(
            self.conv3_input_size,
            self.conv3_output_channel,
            self.conv3_kernel_size,
            self.conv3_stride
        )

        self.conv4_input_size = 24
        self.conv4_output_channel = 32
        self.conv4_kernel_size = (2, 2)
        self.conv4_stride = (1, 1)
        self.conv4 = nn.Conv2d(
            self.conv4_input_size,
            self.conv4_output_channel,
            self.conv4_kernel_size,
            self.conv4_stride
        )

        self.maxpool2_kernel_size = (2, 2)
        self.maxpool2_stride = (2, 2)
        self.maxpool2 = nn.MaxPool2d(
            self.maxpool2_kernel_size,
            self.maxpool2_stride
        )

        # cnn 3

        self.conv5_input_size = 32
        self.conv5_output_channel = 48
        self.conv5_kernel_size = (2, 2)
        self.conv5_stride = (1, 1)
        self.conv5 = nn.Conv2d(
            self.conv5_input_size,
            self.conv5_output_channel,
            self.conv5_kernel_size,
            self.conv5_stride
        )

        self.conv6_input_size = 48
        self.conv6_output_channel = 56
        self.conv6_kernel_size = (1, 1)
        self.conv6_stride = (1, 1)
        self.conv6 = nn.Conv2d(
            self.conv6_input_size,
            self.conv6_output_channel,
            self.conv6_kernel_size,
            self.conv6_stride
        )

        # lstm
        self.lstm_input_size = 3416
        self.lstm_hidden_size = 64
        self.lstm_num_layers = 1
        self.lstm_hidden = None
        self.lstm_cell = None
        self.lstm = nn.LSTM(
            self.lstm_input_size,
            self.lstm_hidden_size,
            self.lstm_num_layers,
            batch_first = True,
            bidirectional = False
        )

        self.linear1_input_size = 64
        self.linear1_output_size = 35
        self.linear1 = nn.Linear(
            self.linear1_input_size,
            self.linear1_output_size
        )

        # initialize
        nn.init.xavier_uniform_(self.conv1.weight, gain=np.sqrt(2))
        nn.init.constant_(self.conv1.bias, 0.1)
        nn.init.xavier_uniform_(self.conv2.weight, gain=np.sqrt(2))
        nn.init.constant_(self.conv2.bias, 0.1)
        nn.init.xavier_uniform_(self.conv3.weight, gain=np.sqrt(2))
        nn.init.constant_(self.conv3.bias, 0.1)
        nn.init.xavier_uniform_(self.conv4.weight, gain=np.sqrt(2))
        nn.init.constant_(self.conv4.bias, 0.1)
        nn.init.xavier_uniform_(self.conv5.weight, gain=np.sqrt(2))
        nn.init.constant_(self.conv5.bias, 0.1)
        nn.init.xavier_uniform_(self.conv6.weight, gain=np.sqrt(2))
        nn.init.constant_(self.conv6.bias, 0.1)

        nn.init.xavier_uniform_(self.linear1.weight, gain=np.sqrt(2))
        nn.init.constant_(self.linear1.bias, 0.1)

        self.batch_normalize = nn.BatchNorm2d(self.conv6_output_channel)

    def forward(self, input):

        batch_size = int(input.size()[0])

        output = self.conv1(input)
        output = self.conv2(output)
        output = self.maxpool1(output)
        output = F.relu(output)

        output = self.conv3(output)
        output = self.conv4(output)
        output = self.maxpool2(output)
        output = F.relu(output)

        output = self.conv5(output)
        output = self.conv6(output)
        output = F.relu(output)

        output = self.batch_normalize(output)

        output = output.permute(0, 3, 2, 1)
        output.contiguous()
        output = output.reshape(batch_size, -1, self.lstm_input_size)

        output, self.lstm_hidden = self.lstm(output)
        output.contiguous()
        output = output.reshape(batch_size, -1)

        output = self.linear1(output)

        return output

    def reset_hidden(self, batch_size):
        zeros = torch.zeros(self.lstm_num_layers * 2, batch_size, self.lstm_hidden_size)
        zeros = zeros.cuda() if self.cuda_enabled else zeros
        self.lstm_hidden = Variable(zeros)

    def reset_cell(self, batch_size):
        zeros = torch.zeros(self.lstm_num_layers * 2, batch_size, self.lstm_hidden_size)
        zeros = zeros.cuda() if self.cuda_enabled else zeros
        self.lstm_cell = Variable(zeros)
