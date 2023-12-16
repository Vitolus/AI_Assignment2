from optuna.distributions import FloatDistribution
from models.Classifier import Classifier


class PolySvc(Classifier):
    def __init__(self):
        super().__init__()

    def train(self):
        super().train()
        self.model.run(
            models="SVM",
            metric="accuracy",
            n_trials=10,
            parallel=True,
            est_params={
                "kernel": "poly",
                "degree": 2,
                "gamma": "scale",
                "shrinking": True,
                "probability": False,
            },
            ht_params={
                "distributions": {
                    "C": FloatDistribution(high=1.0, log=True, low=0.001, step=None),
                    "coef0": FloatDistribution(high=1.0, log=False, low=-1.0, step=None),
                },
            }
        )
        self.results = self.model.evaluate()
