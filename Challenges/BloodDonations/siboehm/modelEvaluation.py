import pandas as pd
from sklearn import model_selection, metrics
from sklearn import linear_model, svm, ensemble

# Location of files
training_file = "BloodDonationTraining.csv"
test_file = "BloodDonationTest.csv"
result_file = "resultSVC.csv"

models = {
    "Logistic Regression": (linear_model.LogisticRegression, {}),
    "SVM": (svm.SVC, {
        "probability": True
    }),
    "Random Forest": (ensemble.RandomForestClassifier, {})
}


def loadDataTrain():
    global train_values, train_target, train_headers

    train_df = pd.read_csv(training_file, header=0)
    train_headers = list(train_df.columns.values)
    train_data = train_df.as_matrix()

    # Extract Values and Target
    train_values = train_data[:, [1, 2, 3, 4]]
    train_target = train_data[:, 5]


def loadDataTest():
    global test_values, test_id
    # Get Test Data
    test_df = pd.read_csv(test_file, header=0)
    test_data = test_df.as_matrix()
    # Extract Values and ids for Saving
    test_values = test_data[:, [1, 2, 3, 4]]
    test_id = test_data[:, 0]


def applyModels():
    for (modelName, (ModelClass, params)) in models.items():
        print(modelName)
        model = ModelClass(**params)
        model.fit(train_values, train_target)
        response = model.predict_proba(test_values)
        print(response)


def crossValidateModels():
    metric = metrics.log_loss
    for (modelName, (ModelClass, params)) in models.items():
        model = ModelClass(**params)
        print("Cross-validating: ", modelName)
        scores = model_selection.cross_val_score(
            scoring=metrics.make_scorer(
                metric, greater_is_better=False),
            cv=10,
            estimator=model,
            X=train_values,
            y=train_target)
        print("Mean score: ", scores.mean())


def main():
    loadDataTrain()
    crossValidateModels()


if __name__ == '__main__':
    main()
