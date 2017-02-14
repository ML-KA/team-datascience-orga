import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random
from sklearn import model_selection, metrics
from sklearn import linear_model, svm, ensemble
from imblearn.over_sampling import RandomOverSampler

# Location of files
training_file = "BloodDonationTraining.csv"
test_file = "BloodDonationTest.csv"
result_file = "resultSVC.csv"

models = {
    "Logistic Regression": (linear_model.LogisticRegression, {}),
    "Logistic Regression With Param": (linear_model.LogisticRegression, {
        "fit_intercept": False,
        "max_iter": 99999999,
        "intercept_scaling": 0.0000001
    }),
    "SVM": (svm.SVC, {
        "probability": True
    }),
    "Random Forest": (ensemble.RandomForestClassifier, {}),
    "Gradient Boosting": (ensemble.GradientBoostingClassifier, {})
}


# Load the training data without changing anything
def loadDataTrain():
    train_df = pd.read_csv(training_file, header=0)
    # train_headers = list(train_df.columns.values)
    train_data = train_df.as_matrix()

    # Extract Values and Target
    train_values = train_data[:, [1, 2, 3, 4]]
    train_target = train_data[:, 5]

    return {
        "train_values": train_values,
        "train_target": train_target,
    }


# Load the training data and remove the volume column
def loadDataTrainNoVolume():
    train_df = pd.read_csv(training_file, header=0)
    train_df = train_df.drop("Total Volume Donated (c.c.)", 1)
    train_data = train_df.as_matrix()

    # Extract Values and Target
    train_values = train_data[:, [1, 2, 3]]
    train_target = train_data[:, 4]

    return {
        "train_values": train_values,
        "train_target": train_target,
    }


# Oversample using simple duplication for 50/50
def loadDataTrainOversampled():
    train_data = loadDataTrainNoVolume()
    train_values = train_data["train_values"]
    train_target = train_data["train_target"]

    ros = RandomOverSampler()
    values_res, target_res = ros.fit_sample(train_values, train_target)

    return {
        "train_values": values_res,
        "train_target": target_res,
    }


# Load the test data
def loadDataTest():
    # Get Test Data
    test_df = pd.read_csv(test_file, header=0)
    test_data = test_df.as_matrix()
    # Extract Values and ids for Saving
    test_values = test_data[:, [1, 2, 3, 4]]
    test_id = test_data[:, 0]

    return {"test_values": test_values, "test_id": test_id}


# Load the test data but remove the volume column
def loadDataTestNoVolume():
    test_df = pd.read_csv(test_file, header=0)
    test_df = test_df.drop("Total Volume Donated (c.c.)", 1)
    test_data = test_df.as_matrix()

    # Extract Values and Target
    test_values = test_data[:, [1, 2, 3]]
    test_id = test_data[:, 0]

    return {"test_values": test_values, "test_id": test_id}


# Load the training data, oversample and distribute it somewhat evenly
def loadDataTrainOversampledDistributed():
    train_df = pd.read_csv(training_file)
    train_df = train_df.drop("Total Volume Donated (c.c.)", 1)
    train_df_sorted = train_df.sort_index()
    train_data = train_df_sorted.as_matrix()

    # Extract Values and Target
    train_values = train_data[:, [1, 2, 3]]
    train_target = train_data[:, 4]

    ros = RandomOverSampler()
    values_res, target_res = ros.fit_sample(train_values, train_target)

    return {
        "train_values": values_res,
        "train_target": target_res,
    }


# Cross validate the given model on the given dataset using StratifiedKFold with 10 folds
# Loss function is logloss, return scores sorted ascending
def crossValidateModels(models, rState, train_values, train_target):
    model_scores = {}
    metric = metrics.log_loss
    splitter = model_selection.StratifiedKFold(n_splits=10, shuffle=True, random_state=rState)

    # for train_index, test_index in splitter.split(train_values, train_target):
    #     print("Train", train_target[train_index])

    # crossvalidate
    for (modelName, (ModelClass, params)) in models.items():
        model = ModelClass(**params)
        scores = model_selection.cross_val_score(
            cv=splitter,
            estimator=model,
            X=train_values,
            y=train_target,
            scoring=metrics.make_scorer(
                metric, greater_is_better=False, needs_proba=True))
        model_scores[modelName] = scores.mean()

    # sort the result
    sorted_names = sorted(model_scores, key=model_scores.get)
    sorted_scores = []
    for name in sorted_names:
        sorted_scores.append((name, model_scores[name]))
    return sorted_scores


# fit the model and create predictions for the test values, write it all to correctly formatted csv
def createUploadFile(model_class, train_data, test_data, params={}):
    train_values = train_data["train_values"]
    train_target = train_data["train_target"]
    test_values = test_data["test_values"]
    test_id = test_data["test_id"]
    model = model_class(**params)
    model.fit(train_values, train_target)
    predictions = model.predict_proba(test_values)[:, 1]

    # Write data to file
    result_df = pd.DataFrame(predictions, test_id,
                             ["Made Donation in March 2007"])
    result_df.to_csv(result_file)

    print("Wrote result to:", result_file)


def printScores(scores):
    for (model, score) in scores:
        print(model + ":", score)

def getHex(r, g, b):
    r = hex(r)[2:]
    g = hex(g)[2:]
    b = hex(b)[2:]
    if len(r)<2:
        r = "0" + r
    if len(g)<2:
        g = "0" + g
    if len(b)<2:
        b = "0" + b
    col = r + g + b
    return "#" + col.upper()

def plotDataByGroup(model_scores):
    plt.subplot(212)

    index = np.arange(5)
    bar_width = 0.2

    opacity = 0.4
    col = 0
    n = 0
    names = []

    for key in model_scores:
        group = model_scores[key]
        value = []
        names = []
        for model in group:
            names.append(model[0])
            value.append(abs(model[1]))

        # add bar graph
        pltbar = plt.bar(index + (bar_width * n), value, bar_width,
                     alpha=opacity,
                     color=getHex((col*10) % 255, (col*5) % 255, (col*2) % 255),
                     label=key)
        col = random.randrange(0,256)
        n = n + 1

    plt.xlabel('Group')
    plt.ylabel('Scores')
    plt.title('Scores by group and technique')
    plt.xticks(index + 0.4, names, rotation=5)
    plt.legend()

def plotDataByModel(model_scores):
    plt.subplot(211)

    index = np.arange(4)
    bar_width = 0.19

    opacity = 0.4
    col = 0
    n = 0

    value = []
    name = []
    lastKey = ""
    for key in model_scores:
         lastKey = key

    for model in model_scores[lastKey]:
        value.append([])

    for key in model_scores:
        group = model_scores[key]
        n = 0;
        for model in group:
            value[n].append(abs(model[1]))
            n = n + 1
        name.append(key)

    n = 0
    for val in value:
        pltbar = plt.bar(index + (bar_width * n), val, bar_width,
                     alpha=opacity,
                     color=getHex((col*10) % 255, (col*5) % 255, (col*2) % 255),
                     label=model_scores[lastKey][n][0])
        col = random.randrange(0,256)
        n = n + 1

    plt.xlabel('Model')
    plt.ylabel('Scores')
    plt.title('Scores by group and technique')
    plt.xticks(index + 0.4, name)
    plt.legend()

def main():
    model_scores = {}
    # Define random state for StratifiedKFold shuffle
    randomState = 0

    # Scores without modifying data, applying the models without modifications
    train_data = loadDataTrain()
    model_scores["no Modifications"] = crossValidateModels(models, randomState, **train_data)
    print("Scores without modifying:")
    printScores(model_scores["no Modifications"])

    # Scores with remove Volume from data, direct correlation with num of donations
    train_data = loadDataTrainNoVolume()
    model_scores["Volume removed"] = crossValidateModels(models, randomState, **train_data)
    print("\nScores with Volume in Data Removed")
    printScores(model_scores["Volume removed"])

    # Scores with remove Volume and oversampling
    train_data = loadDataTrainOversampled()
    model_scores["Add oversampling"] = crossValidateModels(models, randomState, **train_data)
    print("\nScores with Oversampling")
    printScores(model_scores["Add oversampling"])

    # Score with rm Vol, oversampling, sorted by id
    # for sort of evenly distributed target
    train_data = loadDataTrainOversampledDistributed()
    model_scores["Evenly distribute"] = crossValidateModels(models, randomState, **train_data)
    print("\nScores with distribution")
    printScores(model_scores["Evenly distribute"])

    # Create the file to upload
    train_data= loadDataTrainNoVolume()
    test_data = loadDataTestNoVolume()
    createUploadFile(linear_model.LogisticRegression, train_data, test_data)

    plt.figure(figsize=(15,9))
    plotDataByGroup(model_scores)
    plotDataByModel(model_scores)
    plt.show()

main()
