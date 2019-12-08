import os
import sys
import numpy
import pandas

inputDataPath = "../VED/Data/"
outputDataPath = "../npy/"
columns = ["DayNum", "VehId", "Trip", "Timestamp(ms)", "Latitude[deg]", "Longitude[deg]", "Vehicle Speed[km/h]", "MAF[g/sec]", "Engine RPM[RPM]", "Absolute Load[%]", "OAT[DegC]", "Fuel Rate[L/hr]", "Air Conditioning Power[kW]", "Air Conditioning Power[Watts]", "Heater Power[Watts]", "HV Battery Current[A]", "HV Battery SOC[%]", "HV Battery Voltage[V]", "Short Term Fuel Trim Bank 1[%]", "Short Term Fuel Trim Bank 2[%]", "Long Term Fuel Trim Bank 1[%]", "Long Term Fuel Trim Bank 2[%]"]

# get all files in Data directory
files = os.listdir(inputDataPath)

# iterate over all files
for file in files:
    if file[-3:] == "csv":
        fileName = file[:-4]

        data = pandas.read_csv(inputDataPath + file)
        data = data.sort_values(by=["DayNum", "VehId", "Trip", "Timestamp(ms)"])
        data = data.fillna(0)

        vehIds = data["VehId"].unique()
        for vehId in vehIds:
            vehSpecificData = data[data["VehId"] == vehId]
            tripIds = vehSpecificData["Trip"].unique()
            for tripId in tripIds:
                tripSpecificData = vehSpecificData[vehSpecificData["Trip"] == tripId]
                numpyData = tripSpecificData.to_numpy()
                outputFilePath = outputDataPath + "v" + str(vehId) + "-t" + str(tripId) + ".npy"
                numpy.save(outputFilePath, numpyData)

# -----------------------------------------
# Test if the conversion worked as expected
# -----------------------------------------
# data = numpy.load(outputDataPath + "v2-t685.npy")
# numpy.set_printoptions(threshold=sys.maxsize)
# print(data[0][4], data[0][5])
# print(data[data.shape[0] - 1][4], data[data.shape[0] - 1][5])
