from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
import pandas as pd


class Classifier:
    def __init__(self):
        self.X, self.y = fetch_openml('mnist_784', version=1, return_X_y=True)  # load data
        self.y = self.y.astype(int)  # convert string to int
        self.X = self.X / 255.  # normalize data
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X, self.y, test_size=10000, random_state=42)  # split data
        self.results = None
        self.model = None

    def train(self):
        self.model.fit(self.X_train, self.y_train)  # train the model
        self.results = self.model.cv_results_
        return self.model.best_estimator_, self.model.best_params_  # set the best hyperparameters

    def predict(self, x):
        return self.model.predict(x)  # return the predicted label

    def test(self):  # HOF to test the model
        return self.predict(self.X_test)  # return the predicted labels of the test set

    def get_results(self):
        return pd.DataFrame(self.results)
