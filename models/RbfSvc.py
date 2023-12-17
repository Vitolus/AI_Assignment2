from atom import ATOMClassifier
from optuna.distributions import FloatDistribution
from models.Classifier import Classifier


class RbfSvc(Classifier):
    def __init__(self):
        super().__init__()
        self.model = ATOMClassifier(self.X, self.y, test_size=10000, n_jobs=-1, n_rows=40000,
                                    device="gpu", engine="cuml", verbose=2, random_state=1)

    def train(self):
        super().train()
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
                    "C": FloatDistribution(high=1.0, log=True, low=0.001, step=None),
                },
            }
        )
        self.results = self.model.evaluate()
