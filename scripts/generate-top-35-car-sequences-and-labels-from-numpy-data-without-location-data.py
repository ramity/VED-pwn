import os
import sys
import numpy
import math
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--input_batch_size", type=int, default=512)
parser.add_argument("--input_sequence_size", type=int, default=256)
parser.add_argument("--input_sequence_stride", type=int, default=256)
args = parser.parse_args()

input_batch_size = args.input_batch_size
input_sequence_size = args.input_sequence_size
input_sequence_stride = args.input_sequence_stride
input_sequence_features = 17

vehicle_ids = [10, 276, 301, 323, 340, 349, 351, 355, 366, 371, 374, 388, 410, 411, 449, 450, 452, 455, 457, 458, 459, 462, 465, 468, 484, 488, 528, 531, 550, 560, 561, 565, 569, 575, 584]
current_output_batch_index = 0
offset = 0
sequences = numpy.zeros((input_batch_size, input_sequence_size, input_sequence_features))
labels = numpy.zeros((input_batch_size, len(vehicle_ids)), dtype=int)

data_directory = "../npy/"
files = os.listdir(data_directory)
columns = ["DayNum", "VehId", "Trip", "Timestamp(ms)", "Latitude[deg]", "Longitude[deg]", "Vehicle Speed[km/h]", "MAF[g/sec]", "Engine RPM[RPM]", "Absolute Load[%]", "OAT[DegC]", "Fuel Rate[L/hr]", "Air Conditioning Power[kW]", "Air Conditioning Power[Watts]", "Heater Power[Watts]", "HV Battery Current[A]", "HV Battery SOC[%]", "HV Battery Voltage[V]", "Short Term Fuel Trim Bank 1[%]", "Short Term Fuel Trim Bank 2[%]", "Long Term Fuel Trim Bank 1[%]", "Long Term Fuel Trim Bank 2[%]"]
output_directory = "../processed-wol/"

for file in files:

    v_removed = file[1:]
    bits = v_removed.split("-t")
    vehicle_id = bits[0]
    trip_id = bits[1].split(".npy")[0]

    if int(vehicle_id) in vehicle_ids:

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
                labels = numpy.zeros((input_batch_size, len(vehicle_ids)), dtype=int)

                offset = 0
                current_output_batch_index += 1

            # slice array
            start_index = input_sequence_stride * x
            final_index = input_sequence_size + start_index
            slice = numpy_data[start_index:final_index]

            label = numpy.zeros(len(vehicle_ids), dtype=int)
            label[vehicle_ids.index(int(vehicle_id))] = 1
            sequence = numpy.delete(slice, [0, 1, 2, 4, 5], 1)

            sequences[offset] = sequence
            labels[offset] = label
            offset += 1
