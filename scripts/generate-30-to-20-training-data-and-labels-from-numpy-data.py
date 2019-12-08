import os
import sys
import numpy
import math

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

input_sequence_size = 600
output_sequence_size = 20

data_directory = "../npy/"
data_files = os.listdir(data_directory)

counter = AverageMeter()

for file in data_files:

    file_path = data_directory + file
    file_name = file[:-4]
    file_extension = file[-3:]

    numpy_data = numpy.load(file_path)
    numpy_data_rows = numpy_data.shape[0]

    # y=30x+20 => y-20=30x => (y-20)/30=x
    possible_sequences = math.floor((numpy_data_rows - 20) / 30)

    # skip files that can't be used
    if possible_sequences == 0:
        continue

    counter.update(numpy_data_rows)
    print(counter.avg)

    # input_numpy_sequences = []
    # output_numpy_sequences = []
    #
    # print(numpy_data_rows)
    # print(possible_sequences)
    # sys.exit(1)
    #
    # for x in range(0, possible_sequences):
    #     first_index = (input_sequence_size * x)
    #     last_index = (input_sequence_size * x)
