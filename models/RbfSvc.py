from optuna.distributions import FloatDistribution
from sklearn.datasets import fetch_openml
from atom import ATOMClassifier


class RbfSvc:

    def __init__(self):
        X, y = fetch_openml('mnist_784', version=1, return_X_y=True)  # load data
        y = y.astype(int)  # convert string to int
        X = X / 255.  # normalize data
        self.model = ATOMClassifier(X, y, test_size=10000, n_jobs=-1, logger="auto",
                                    device="gpu", engine="cuml", verbose=2, random_state=1)
        self.results = None

    def train(self):
        self.model.feature_selection(strategy="pca", solver="full")
        self.model.run(
            models="SVM",
            metric="accuracy",
            n_trials=10,
            parallel=True,
            est_params={
                "kernel": "rbf",
                "gamma": "scale",
                "shrinking": True,
                "probability": False,
            },
            ht_params={
                "distributions": {
                    "C": FloatDistribution(high=10.0, log=True, low=0.01, step=None),
                },
            }
        )
        self.results = self.model.evaluate()
