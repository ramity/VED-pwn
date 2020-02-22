import os
import sys
import numpy

input_directory = "../processed-wol/"
input_files = os.listdir(input_directory)

labels = numpy.array()

for file in input_files:

    print(file)
