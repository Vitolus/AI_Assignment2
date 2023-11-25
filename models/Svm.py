from models.Classifier import Classifier
from sklearn.model_selection import GridSearchCV
from sklearn import svm


class Svm(Classifier):
    def __init__(self, hyperparams):
        super().__init__()  # call parent constructor
        # self.X_train, self.X_test, self.y_train, self.y_test self.results are inherited from Classifier
        self.model = GridSearchCV(svm.SVC(random_state=42), hyperparams, cv=10, n_jobs=-1,
                                  return_train_score=False, verbose=3)