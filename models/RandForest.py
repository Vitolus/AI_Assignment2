from optuna.distributions import FloatDistribution, CategoricalDistribution, IntDistribution
from models.Classifier import Classifier


class RandForest(Classifier):
    def __init__(self):
        super().__init__()

    def train(self):
        super().train()
        self.model.run(
            models="RF",
            metric="accuracy",
            n_trials=10,
            parallel=True,
            est_params={
                "max_depth": None,
                "min_samples_split": 2,
                "min_samples_leaf": 1,
                "max_features": "sqrt",
                "bootstrap": True,
                "max_samples": None,
            },
            ht_params={
                "distributions": {
                    "n_estimators": IntDistribution(high=1000, log=False, low=10, step=10),
                    "criterion": CategoricalDistribution(choices=("gini", "entropy")),
                    "ccp_alpha": FloatDistribution(high=0.035, log=False, low=0.0, step=0.005),
                },
            }
        )
        self.results = self.model.evaluate()
