import os
import sys
import numpy
import math
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--input_batch_size", type=int, default=512)
parser.add_argument("--input_sequence_size", type=int, default=600)
parser.add_argument("--input_sequence_stride", type=int, default=150)
args = parser.parse_args()

class_list_encoding = [" ", "+", "-", ".", "0", "1", "2", "3", "4", "5", "6", "7", "8", "9"]

def encode(input, class_list_encoding):

    output = []

    for char in input:

        index = class_list_encoding.index(char)
        output.append(index)

    return numpy.asarray(output, dtype=int)

input_batch_size = args.input_batch_size
input_sequence_size = args.input_sequence_size
input_sequence_stride = args.input_sequence_stride
input_sequence_features = 17

current_output_batch_index = 0
offset = 0
sequences = numpy.zeros((input_batch_size, input_sequence_size, input_sequence_features))
labels = numpy.zeros((input_batch_size, 2, 11))

data_directory = "../npy/"
data_files = os.listdir(data_directory)
columns = ["DayNum", "VehId", "Trip", "Timestamp(ms)", "Latitude[deg]", "Longitude[deg]", "Vehicle Speed[km/h]", "MAF[g/sec]", "Engine RPM[RPM]", "Absolute Load[%]", "OAT[DegC]", "Fuel Rate[L/hr]", "Air Conditioning Power[kW]", "Air Conditioning Power[Watts]", "Heater Power[Watts]", "HV Battery Current[A]", "HV Battery SOC[%]", "HV Battery Voltage[V]", "Short Term Fuel Trim Bank 1[%]", "Short Term Fuel Trim Bank 2[%]", "Long Term Fuel Trim Bank 1[%]", "Long Term Fuel Trim Bank 2[%]"]
output_directory = "../batches/"

for file in data_files:

    file_path = data_directory + file
    file_name = file[:-4]
    file_extension = file[-3:]

    numpy_data = numpy.load(file_path)
    numpy_data_rows = numpy_data.shape[0]

    possible_sequences = math.floor((numpy_data_rows - input_sequence_size) / input_sequence_stride)

    # skip files that can't be used
    if possible_sequences <= 0:
        continue

    for x in range(0, possible_sequences):

        # save training sequence data if size threshold is met
        if offset >= input_batch_size:

            sequences_output_file_path = output_directory + "sequences-batch-" + str(current_output_batch_index) + ".npy"
            labels_output_file_path = output_directory + "labels-batch-" + str(current_output_batch_index) + ".npy"

            numpy.save(sequences_output_file_path, sequences)
            numpy.save(labels_output_file_path, labels)

            sequences = numpy.zeros((input_batch_size, input_sequence_size, input_sequence_features))
            labels = numpy.zeros((input_batch_size, 2, 11))

            offset = 0
            current_output_batch_index += 1

        # slice array
        start_index = input_sequence_stride * x
        final_index = input_sequence_size + start_index
        slice = numpy_data[start_index:final_index]

        # get required data for label
        start_latitude = slice[0][4]
        start_longitude = slice[0][5]
        final_latitude = slice[-1][4]
        final_longitude = slice[-1][5]

        # NOTE: labels use start_* as the origin
        label_latitude = final_latitude - start_latitude
        label_longitude = final_longitude - start_longitude
        label_latitude = "{:.8f}".format(label_latitude)
        label_longitude = "{:.8f}".format(label_longitude)
        label_latitude = ("+" + label_latitude) if label_latitude[0] != "-" else label_latitude
        label_longitude = ("+" + label_longitude) if label_longitude[0] != "-" else label_longitude
        label_latitude = encode(label_latitude, class_list_encoding)
        label_longitude = encode(label_longitude, class_list_encoding)

        label = numpy.asarray([label_latitude, label_longitude])
        sequence = numpy.delete(slice, [0, 1, 2, 4, 5], 1)

        sequences[offset] = sequence
        labels[offset] = label
        offset += 1
