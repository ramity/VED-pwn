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

        self.classes = 13 + 1
        self.cuda_enabled = cuda_enabled

        # batch normalize
        self.batch_normalize = nn.BatchNorm1d(600)

        # lstm
        self.lstm_input_size = 17
        self.lstm_hidden_size = 10
        self.lstm_num_layers = 10
        self.lstm_hidden = None
        self.lstm_cell = None
        self.lstm = nn.LSTM(
            self.lstm_input_size,
            self.lstm_hidden_size,
            self.lstm_num_layers,
            batch_first = True,
            dropout = 0.2,
            bidirectional = True
        )

        # linear
        # 2 = number of directions the lstm layer outputs
        self.linear_input_size = 2 * self.lstm_hidden_size
        self.linear_output_size = self.classes
        self.linear = nn.Linear(
            self.linear_input_size,
            self.linear_output_size
        )

        # softmax
        self.softmax = nn.Softmax(dim=1)

        # log_softmax
        self.log_softmax = nn.LogSoftmax(dim=1)

        # initialize
        # nn.init.xavier_uniform_(self.lstm.weight_ih_l0, gain=np.sqrt(2))
        # nn.init.xavier_uniform_(self.lstm.weight_ih_l1, gain=np.sqrt(2))
        # nn.init.xavier_uniform_(self.lstm.weight_ih_l2, gain=np.sqrt(2))
        # nn.init.xavier_uniform_(self.lstm.weight_hh_l0, gain=np.sqrt(2))
        # nn.init.xavier_uniform_(self.lstm.weight_hh_l1, gain=np.sqrt(2))
        # nn.init.xavier_uniform_(self.lstm.weight_hh_l2, gain=np.sqrt(2))
        # nn.init.constant_(self.lstm.bias_ih_l0, 0.1)
        # nn.init.constant_(self.lstm.bias_ih_l1, 0.1)
        # nn.init.constant_(self.lstm.bias_ih_l2, 0.1)
        # nn.init.constant_(self.lstm.bias_hh_l0, 0.1)
        # nn.init.constant_(self.lstm.bias_hh_l1, 0.1)
        # nn.init.constant_(self.lstm.bias_hh_l2, 0.1)
        nn.init.xavier_uniform_(self.linear.weight, gain=np.sqrt(2))
        nn.init.constant_(self.linear.bias, 0.1)

    def forward(self, input):

        batch_size = int(input.size()[0])

        output = self.batch_normalize(input)

        output, self.lstm_hidden = self.lstm(output)
        output.contiguous()
        output = output.reshape(-1, self.linear_input_size)

        output = self.linear(output)
        softmax_output = self.softmax(output)
        log_softmax_output = self.log_softmax(output)

        return softmax_output, log_softmax_output

    def reset_hidden(self, batch_size):
        zeros = torch.zeros(self.lstm_num_layers * 2, batch_size, self.lstm_hidden_size)
        zeros = zeros.cuda() if self.cuda_enabled else zeros
        self.lstm_hidden = Variable(zeros)

    def reset_cell(self, batch_size):
        zeros = torch.zeros(self.lstm_num_layers * 2, batch_size, self.lstm_hidden_size)
        zeros = zeros.cuda() if self.cuda_enabled else zeros
        self.lstm_cell = Variable(zeros)
