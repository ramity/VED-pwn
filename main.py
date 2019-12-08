import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from torchvision import datasets, transforms
from torch.autograd import Variable

import numpy
import time
import os
import sys

from decode import decode
from model import *

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def encode(input, class_list_encoding):

    output = []

    for char in input:

        index = class_list_encoding.index(char)
        output.append(index)

    return numpy.asarray(output, dtype=int)

# init parameters
batch_size = 512
class_list_encoding = [" ", "+", "-", ".", "0", "1", "2", "3", "4", "5", "6", "7", "8", "9"]
classes = len(class_list_encoding)
cuda = True if torch.cuda.is_available() else False
epochs = 100
evaluate = False
kwargs = {'num_workers': 0, 'pin_memory': False} if cuda else {}
log_interval = 1
lr = 0.01
seed = 1
validate_batch_size = 512

# set seed
torch.manual_seed = seed
if cuda:
    torch.cuda.manual_seed = seed
np.random.seed(seed)

# define variables for loading datasets
input_directory = "./batches/"
input_files = os.listdir(input_directory)
training_batch_indexes = range(0, 68)
validation_batch_indexes = range(68, 89)

# load and init the model
model = Net(cuda)
model.load_state_dict(torch.load("./compiled/1575764687-epoch-0-train-02.919383-test-02.907543.pt"))

if cuda:
    model.cuda()

optimizer = torch.optim.Adam(model.parameters(), lr=lr)
criterion = nn.CTCLoss(
    blank=0,
    reduction='mean',
    zero_infinity=False
)

# define train method
def train(epoch, batch_index, batch_size, loss_average):

    model.train()

    batch_time = AverageMeter()
    losses = AverageMeter()
    end = time.time()

    train_sequences_path = input_directory + "sequences-batch-" + str(batch_index) + ".npy"
    train_labels_path = input_directory + "labels-batch-" + str(batch_index) + ".npy"
    random_order = numpy.random.shuffle(np.arange(batch_size))
    train_data = numpy.load(train_sequences_path)[random_order]
    train_labels = numpy.load(train_labels_path)[random_order]
    train_data = numpy.squeeze(train_data, axis=0)
    train_labels = numpy.squeeze(train_labels, axis=0)
    train_data = torch.Tensor(train_data)
    train_labels = torch.IntTensor(train_labels)

    input = train_data
    target = train_labels
    batch_size = input.shape[0]

    model.reset_hidden(batch_size)
    model.reset_cell(batch_size)
    model.zero_grad()

    if cuda:
        input, target = input.cuda(), target.cuda()

    input = input.view(input.shape[0], input.shape[1], input.shape[2])
    input, target = Variable(input), Variable(target)
    softmax_output, log_softmax_output = model(input)

    softmax_output = softmax_output.view(batch_size, -1, classes)
    log_softmax_output = log_softmax_output.view(batch_size, -1, classes)

    predicted_digits = decode(softmax_output, class_list_encoding)
    distance_delta = AverageMeter()

    print(predicted_digits.shape)
    sys.exit(1)

    # for id in range(0, batch_size):
    #
    #
    #
    #     for digit_id in range(0, 11):


    # nn.CTCLoss expects a LogSoftmaxed output
    log_softmax_output = log_softmax_output.permute(1, 0, 2)
    input_lengths = torch.full(size=(batch_size,), fill_value=600, dtype=torch.int32)
    target_lengths = torch.full(size=(batch_size,), fill_value=11, dtype=torch.int32)

    latitude_target = target.narrow(1, 0, 1)
    longitude_target = target.narrow(1, 1, 1)
    latitude_target = latitude_target.view(batch_size, -1)
    longitude_target = longitude_target.view(batch_size, -1)

    # latitude loss
    latitude_loss = criterion(log_softmax_output, latitude_target, input_lengths, target_lengths)
    longitude_loss = criterion(log_softmax_output, longitude_target, input_lengths, target_lengths)
    loss = latitude_loss + longitude_loss
    losses.update(loss.item())

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    batch_time.update(time.time() - end)
    end = time.time()

    print('Train Epoch: {:03d} [{:05d}/{:05d} ({:03.0f}%)]\t'
          'Loss {loss.val:.4f} (avg: {loss_average:.4f})\t'
          'Time {batch_time.val:.3f} (avg: {batch_time.avg:.3f}, sum: {batch_time.sum:.3f})\t'.format(
        epoch, batch_index * batch_size, 34304,
        100 * ((batch_index * batch_size) / 34304), loss=losses, loss_average=loss_average, batch_time=batch_time))

    return losses.avg

# define validate method
def validate(epoch, batch_index, batch_size, loss_average):

    model.eval()

    batch_time = AverageMeter()
    losses = AverageMeter()
    end = time.time()

    with torch.no_grad():

        train_sequences_path = input_directory + "sequences-batch-" + str(batch_index) + ".npy"
        train_labels_path = input_directory + "labels-batch-" + str(batch_index) + ".npy"
        random_order = numpy.random.shuffle(np.arange(batch_size))
        train_data = numpy.load(train_sequences_path)[random_order]
        train_labels = numpy.load(train_labels_path)[random_order]
        train_data = numpy.squeeze(train_data, axis=0)
        train_labels = numpy.squeeze(train_labels, axis=0)
        train_data = torch.Tensor(train_data)
        train_labels = torch.IntTensor(train_labels)

        input = train_data
        target = train_labels
        batch_size = input.shape[0]

        model.reset_hidden(batch_size)
        model.reset_cell(batch_size)
        model.zero_grad()

        if cuda:
            input, target = input.cuda(), target.cuda()

        input = input.view(input.shape[0], input.shape[1], input.shape[2])
        input, target = Variable(input), Variable(target)
        softmax_output, log_softmax_output = model(input)

        softmax_output = softmax_output.view(batch_size, -1, classes)
        log_softmax_output = log_softmax_output.view(batch_size, -1, classes)

        # nn.CTCLoss expects a LogSoftmaxed output
        log_softmax_output = log_softmax_output.permute(1, 0, 2)
        input_lengths = torch.full(size=(batch_size,), fill_value=600, dtype=torch.int32)
        target_lengths = torch.full(size=(batch_size,), fill_value=11, dtype=torch.int32)

        latitude_target = target.narrow(1, 0, 1)
        longitude_target = target.narrow(1, 1, 1)
        latitude_target = latitude_target.view(batch_size, -1)
        longitude_target = longitude_target.view(batch_size, -1)

        # latitude loss
        latitude_loss = criterion(log_softmax_output, latitude_target, input_lengths, target_lengths)
        longitude_loss = criterion(log_softmax_output, longitude_target, input_lengths, target_lengths)
        loss = latitude_loss + longitude_loss
        losses.update(loss.item())

        batch_time.update(time.time() - end)
        end = time.time()
        print('Test Epoch: {:03d} [{:05d}/{:05d} ({:03.0f}%)]\t'
              'Loss {loss.val:.4f} (avg: {loss_average:.4f})\t'
              'Time {batch_time.val:.3f} (avg: {batch_time.avg:.3f}, sum: {batch_time.sum:.3f})\t'.format(
            epoch, (batch_index - min(validation_batch_indexes)) * validate_batch_size, 9728,
            100 * (((batch_index - min(validation_batch_indexes)) * validate_batch_size) / 9728), loss=losses, loss_average=loss_average, batch_time=batch_time))

        return losses.avg

for epoch in range(0, epochs):

    train_loss = AverageMeter()
    validation_loss = AverageMeter()

    for batch_index in training_batch_indexes:
        train_loss.update(train(epoch, batch_index, batch_size, train_loss.avg))

    print('-' * 107)

    for batch_index in validation_batch_indexes:
        validation_loss.update(validate(epoch, batch_index, batch_size, validation_loss.avg))

    print('-' * 107)

    filename = str(int(time.time()))
    filename = filename + "-epoch-" + str(epoch)
    filename = filename + "-train-{:09f}".format(train_loss.avg)
    filename = filename + "-test-{:09f}".format(validation_loss.avg)
    filepath = "./compiled/" + filename + ".pt"
    print("Saving ", filepath)
    torch.save(model.state_dict(), filepath)
    print("Saved ", filepath)

    print('-' * 107)
