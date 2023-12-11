from sklearn.datasets import fetch_openml
from atom import ATOMClassifier


class Classifier:
    def __init__(self):
        X, y = fetch_openml('mnist_784', version=1, return_X_y=True)  # load data
        y = y.astype(int)  # convert string to int
        X = X / 255.  # normalize data
        self.model = ATOMClassifier(X, y, test_size=10000, n_jobs=-1, logger="auto",
                                    device="gpu", engine="cuml", verbose=2, random_state=1)
        self.results = None

    def train(self):
        self.model.feature_selection(strategy="pca", solver="full")
