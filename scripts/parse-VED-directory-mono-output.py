import os
import sys
import numpy
import pandas

inputDataPath = "../VED/Data/"
outputFilePath = "../npy/"
columns = ["DayNum", "VehId", "Trip", "Timestamp(ms)", "Latitude[deg]", "Longitude[deg]", "Vehicle Speed[km/h]", "MAF[g/sec]", "Engine RPM[RPM]", "Absolute Load[%]", "OAT[DegC]", "Fuel Rate[L/hr]", "Air Conditioning Power[kW]", "Air Conditioning Power[Watts]", "Heater Power[Watts]", "HV Battery Current[A]", "HV Battery SOC[%]", "HV Battery Voltage[V]", "Short Term Fuel Trim Bank 1[%]", "Short Term Fuel Trim Bank 2[%]", "Long Term Fuel Trim Bank 1[%]", "Long Term Fuel Trim Bank 2[%]"]

# get all files in Data directory
files = os.listdir(inputDataPath)
dfList = []

# iterate over all files
for file in files:
    if file[-3:] == "csv":
        data = pandas.read_csv(inputDataPath + file)
        data = data.fillna(0)
        dfList.append(data)

huge = pandas.concat(dfList)
huge = huge.sort_values(by=["Trip", "VehId", "DayNum", "Timestamp(ms)"])
csvData = huge.to_csv(index=False)
with open(outputFilePath, "a") as fd:
    fd.write(csvData)
