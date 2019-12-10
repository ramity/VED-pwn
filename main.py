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

# init parameters
batch_size = 512
vehicle_ids = [10, 276, 301, 323, 340, 349, 351, 355, 366, 371, 374, 388, 410, 411, 449, 450, 452, 455, 457, 458, 459, 462, 465, 468, 484, 488, 528, 531, 550, 560, 561, 565, 569, 575, 584]
cuda = True if torch.cuda.is_available() else False
epochs = 100
kwargs = {'num_workers': 0, 'pin_memory': False} if cuda else {}
log_interval = 1
lr = 0.001
seed = 1
validate_batch_size = 512

# set seed
torch.manual_seed = seed
if cuda:
    torch.cuda.manual_seed = seed
np.random.seed(seed)

# define variables for loading datasets
input_directory = "./processed/"
input_files = os.listdir(input_directory)
training_batch_indexes = [32, 22, 34, 5, 16, 20, 18, 4, 21, 23,10, 3, 17, 7, 15, 11, 1, 13, 6, 24, 31, 2, 33, 9, 8, 12, 25]
validation_batch_indexes = [25, 14, 30, 26, 27, 28, 29]

# load and init the model
model = Net(cuda)
model.load_state_dict(torch.load("./processed-compiled/1575869782-epoch-0-train-00.566840-test-00.417132.pt"))

if cuda:
    model.cuda()

optimizer = torch.optim.Adam(model.parameters(), lr=lr)
criterion = nn.CrossEntropyLoss(
    reduction="sum"
)

# define train method
def train(epoch, batch_index, batch_size):

    model.train()

    batch_time = AverageMeter()
    losses = AverageMeter()
    end = time.time()

    train_sequences_path = input_directory + "sequences-batch-" + str(batch_index) + ".npy"
    train_labels_path = input_directory + "labels-batch-" + str(batch_index) + ".npy"
    order = np.arange(batch_size)
    numpy.random.shuffle(order)
    train_data = numpy.load(train_sequences_path)[order]
    train_labels = numpy.load(train_labels_path)[order]
    train_data = torch.Tensor(train_data)
    train_labels = torch.LongTensor(train_labels)

    input = train_data
    target = train_labels
    target = torch.max(target, 1)[1]
    batch_size = input.shape[0]

    model.reset_hidden(batch_size)
    model.reset_cell(batch_size)
    model.zero_grad()

    if cuda:
        input, target = input.cuda(), target.cuda()

    input = input.view(input.shape[0], 1, input.shape[1], input.shape[2])
    input, target = Variable(input), Variable(target)

    output = model(input)

    correct = 0
    for n in range(0, batch_size):
        a = int(output[n].max(0)[1])
        b = int(target[n])
        if a == b:
            correct += 1
    train_accuracy = correct / batch_size

    print(train_accuracy)

    loss = criterion(output, target)
    losses.update(loss.item() / batch_size)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    batch_time.update(time.time() - end)
    end = time.time()

    print('Train Epoch: {:03d} [{:05d}/{:05d} ({:03.0f}%)]\t'
          'Loss {loss.val:.4f} (avg: {loss.avg:.4f})\t'
          'Time {batch_time.val:.3f} (avg: {batch_time.avg:.3f}, sum: {batch_time.sum:.3f})\t'.format(
        epoch, batch_index * batch_size, 34304,
        100 * ((batch_index * batch_size) / 34304), loss=losses, batch_time=batch_time))

    return train_accuracy

# define validate method
def validate(epoch, batch_index, batch_size):

    model.eval()

    batch_time = AverageMeter()
    losses = AverageMeter()
    end = time.time()

    with torch.no_grad():

        train_sequences_path = input_directory + "sequences-batch-" + str(batch_index) + ".npy"
        train_labels_path = input_directory + "labels-batch-" + str(batch_index) + ".npy"
        order = np.arange(batch_size)
        numpy.random.shuffle(order)
        train_data = numpy.load(train_sequences_path)[order]
        train_labels = numpy.load(train_labels_path)[order]
        train_data = torch.Tensor(train_data)
        train_labels = torch.LongTensor(train_labels)

        input = train_data
        target = train_labels
        target = torch.max(target, 1)[1]
        batch_size = input.shape[0]

        model.reset_hidden(batch_size)
        model.reset_cell(batch_size)
        model.zero_grad()

        if cuda:
            input, target = input.cuda(), target.cuda()

        input = input.view(input.shape[0], 1, input.shape[1], input.shape[2])
        input, target = Variable(input), Variable(target)

        output = model(input)

        correct = 0
        for n in range(0, batch_size):
            a = int(output[n].max(0)[1])
            b = int(target[n])
            if a == b:
                correct += 1
        test_accuracy = correct / batch_size

        print(test_accuracy)

        loss = criterion(output, target)
        losses.update(loss.item() / batch_size)

        batch_time.update(time.time() - end)
        end = time.time()
        print('Test Epoch: {:03d} [{:05d}/{:05d} ({:03.0f}%)]\t'
              'Loss {loss.val:.4f} (avg: {loss.avg:.4f})\t'
              'Time {batch_time.val:.3f} (avg: {batch_time.avg:.3f}, sum: {batch_time.sum:.3f})\t'.format(
            epoch, (batch_index - min(validation_batch_indexes)) * validate_batch_size, 9728,
            100 * (((batch_index - min(validation_batch_indexes)) * validate_batch_size) / 9728), loss=losses, batch_time=batch_time))

        return test_accuracy

for epoch in range(0, epochs):

    epoch_accuracy = AverageMeter()

    for batch_index in training_batch_indexes:
        epoch_accuracy.update(train(epoch, batch_index, batch_size))

    print('-' * 107)

    train_accuracy = epoch_accuracy.avg
    print(train_accuracy)
    epoch_accuracy = AverageMeter()

    print('-' * 107)

    for batch_index in validation_batch_indexes:
        epoch_accuracy.update(validate(epoch, batch_index, batch_size))

    print('-' * 107)

    test_accuracy = epoch_accuracy.avg
    print(test_accuracy)

    print('-' * 107)

    filename = str(int(time.time()))
    filename = filename + "-epoch-" + str(epoch)
    filename = filename + "-train-{:09f}".format(train_accuracy)
    filename = filename + "-test-{:09f}".format(test_accuracy)
    filepath = "./processed-compiled/" + filename + ".pt"
    print("Saving ", filepath)
    torch.save(model.state_dict(), filepath)
    print("Saved ", filepath)

    print('-' * 107)
