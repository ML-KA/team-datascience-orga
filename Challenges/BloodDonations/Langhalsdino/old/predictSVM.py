import csv
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn import svm
import numpy as np

features = []
target = []
testFeatures = []
testTarget = []

def get_data(filename):
    with open(filename, 'r') as csvfile:
        csvFileReader = csv.reader(csvfile)
        next(csvFileReader)	# skipping column names
        for row in csvFileReader:
            arr = row[0].split(';')
            # monthSinceLastDonation, numberOfDonations, totalVolume, monthSinceFirstDonation, target
            features.append([float(arr[1]),float(arr[2]),float(arr[4])])
            target.append(float(arr[5]))
    return

def get_validation_data(filename):
    with open(filename, 'r') as csvfile:
        csvFileReader = csv.reader(csvfile)
        next(csvFileReader)	# skipping column names
        for row in csvFileReader:
            arr = row[0].split(';')
            # monthSinceLastDonation, numberOfDonations, totalVolume, monthSinceFirstDonation, target
            testFeatures.append([float(arr[0]),float(arr[1]),float(arr[3])])
            testTarget.append(float(arr[4]))
    return

def label_data(filename, mySVM):
    rows = []
    del features [:]
    del target [:]
    with open("TestData.csv", 'r') as csvfile:
        csvFileReader = csv.reader(csvfile)
        next(csvFileReader)
        for row in csvFileReader:
            arr = row[0].split(';')
            rows.append(arr[0].split('adsf'))
            # monthSinceLastDonation, numberOfDonations, totalVolume, monthSinceFirstDonation, target
            features.append([float(arr[0]),float(arr[1]),float(arr[2]),float(arr[3]),float(arr[4])])

    for feature in features:
        temp = np.array([feature[1], feature[2], feature[4]]).reshape((1, -1))
        target.append(float(mySVM.predict(temp)))

    for i in range(0,len(rows)):
        rows[i].append(target[i])

    with open(filename, "wb") as f:
        writer = csv.writer(f, quoting=csv.QUOTE_ALL)
        writer.writerows(rows)

# Train my data
def trainModel():
    get_data("TrainingSet.csv")

    mySVM = svm.SVC()
    mySVM.fit(features, target)
    print(mySVM.score(features, target))
    get_validation_data("labeld_test.csv")
    print(mySVM.score(testFeatures, testTarget))
    label_data("result.csv", mySVM)
    #plot_linear(mySVM)

trainModel()
