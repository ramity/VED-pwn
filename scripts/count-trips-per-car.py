import os
import sys

files = os.listdir("../npy/")
data = {}

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

trip_count = AverageMeter()

for filename in files:

    v_removed = filename[1:]
    bits = v_removed.split("-t")
    vehicle_id = bits[0]
    trip_id = bits[1].split(".npy")[0]

    if vehicle_id in data:
        data[vehicle_id] += 1
    else:
        data[vehicle_id] = 1

for id in data:

    trip_count.update(int(data[id]))

filtered_data = data.copy()

for id in data:

    if data[id] < 200:

        del filtered_data[id]

outputString = "["

for id in filtered_data:

    outputString += str(id) + ", "

outputString = outputString[:-2] + "]"
print(outputString)
print(len(filtered_data))

print(trip_count.avg)
