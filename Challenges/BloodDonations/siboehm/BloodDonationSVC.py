# CSV importing
import pandas as pd
from sklearn import svm

print("DrivenData Challenge | BloodDonations")

# Location of files
training_file = "BloodDonationTraining.csv"
test_file = "BloodDonationTest.csv"
result_file = "resultSVC.csv"

# Get Training Data
trainingPD = pd.read_csv(training_file, header=0)
training_headers = list(trainingPD.columns.values)
trainingData = trainingPD.as_matrix()
# Extract Values and Target
valuesTraining = trainingData[:, [1, 2, 3, 4]]
target = trainingData[:, 5]

# Get Test Data
testPD = pd.read_csv(test_file, header=0)
testData = testPD.as_matrix()
# Extract Values and ids for Saving
valuesTest = testData[:, [1, 2, 3, 4]]
idTest = testData[:, 0]

# Set probability to get estimation instead of most probable class
clf = svm.SVC(probability=True)

# Fit SVC to training data
clf.fit(valuesTraining, target)

# Predict target for training data
testResults = clf.predict_proba(valuesTest)[:, 1]

# Write data to file
resultDF = pd.DataFrame(testResults, idTest, ["Made Donation in March 2007"])
resultDF.to_csv(result_file)

print("Wrote results to:", result_file)
