import Classifier
from sklearn.model_selection import GridSearchCV
from sklearn import svm


class Svm(Classifier):
    def __init__(self, hyperparams):  # gamma is the kernel coefficient
        super.__init__(self)  # call parent constructor
        self.hyperparams = hyperparams  # set hyperparameters
        self.model = GridSearchCV(svm.linearSVC(), self.hyperparams, cv=10)  # 10-fold cross validation

    def train(self):  # train the model
        self.model.fit(self.X_train, self.y_train)  # train the model
        return self.model.best_params_  # return the best hyperparameters

    def predict(self, x):  # predict the label of x
        return self.model.predict(x)  # return the predicted label

    def _test(self):  # test the model
        return self.predict(self.X_test)  # return the predicted labels of the test set
