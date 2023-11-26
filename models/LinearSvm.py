from models.Classifier import Classifier
from sklearn.model_selection import GridSearchCV
from sklearn import svm
from atom import ATOMClassifier


class LinearSvm(Classifier):
    def __init__(self, hyperparams):
        super().__init__()  # call parent constructor
        # self.X_train, self.X_test, self.y_train, self.y_test self.results are inherited from Classifier
        # self.model = GridSearchCV(svm.LinearSVC(dual='auto', random_state=42), hyperparams, cv=10, n_jobs=-1,
        #                           return_train_score=True, verbose=3)
        self.model = ATOMClassifier(self.X, self.y, test_size=10000, n_rows=1, n_jobs=-1, device='gpu',
                                    engine='cuml', verbose=2, random_state=666)
