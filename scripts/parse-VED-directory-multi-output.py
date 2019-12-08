import os
import sys
import numpy
import pandas

inputDataPath = "../VED/Data/"
outputDataPath = "../csvs/"
columns = ["DayNum", "VehId", "Trip", "Timestamp(ms)", "Latitude[deg]", "Longitude[deg]", "Vehicle Speed[km/h]", "MAF[g/sec]", "Engine RPM[RPM]", "Absolute Load[%]", "OAT[DegC]", "Fuel Rate[L/hr]", "Air Conditioning Power[kW]", "Air Conditioning Power[Watts]", "Heater Power[Watts]", "HV Battery Current[A]", "HV Battery SOC[%]", "HV Battery Voltage[V]", "Short Term Fuel Trim Bank 1[%]", "Short Term Fuel Trim Bank 2[%]", "Long Term Fuel Trim Bank 1[%]", "Long Term Fuel Trim Bank 2[%]"]

# get all files in Data directory
files = os.listdir(inputDataPath)

# iterate over all files
for file in files:
    if file[-3:] == "csv":
        data = pandas.read_csv(inputDataPath + file)
        data = data.sort_values(by=["Trip", "VehId", "DayNum", "Timestamp(ms)"])
        data = data.fillna(0)
        tripIds = data["Trip"].unique()

        for tripId in tripIds:
            tripData = data[data["Trip"] == tripId]
            vehIds = tripData["VehId"].unique()
            for vehId in vehIds:
                vehSpecificTripData = tripData[tripData["VehId"] == vehId]
                csvData = tripData.to_csv(index=False)
                outputFilePath = outputDataPath + str(tripId) + "-" + str(vehId) + ".csv"
                print("Writing: " + outputFilePath)
                with open(outputFilePath, "w") as fd:
                    fd.write(csvData)

print(files)
