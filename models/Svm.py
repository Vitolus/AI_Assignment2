from models.Classifier import Classifier
from sklearn.model_selection import GridSearchCV
from sklearn import svm


class Svm(Classifier):
    def __init__(self, hyperparams):
        super().__init__()  # call parent constructor
        # self.X_train, self.X_test, self.y_train, self.y_test self.results are inherited from Classifier
        self.model = GridSearchCV(svm.LinearSVC(random_state=42), hyperparams, cv=10, n_jobs=-1,
                                  return_train_score=False, verbose=3)

    def train(self):
        self.model.fit(self.X_train, self.y_train)  # train the model
        self.results = self.model.cv_results_
        return self.model.best_estimator_, self.model.best_params_  # set the best hyperparameters

    def predict(self, x):
        return self.model.predict(x)  # return the predicted label

    def test(self):  # HOF to test the model
        return self.predict(self.X_test)  # return the predicted labels of the test set
