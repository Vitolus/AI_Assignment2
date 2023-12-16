from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split, KFold
import cupy as cp
from scipy.special import gammaln
from sklearn.metrics import accuracy_score
import time


class NaiveBayes:
    def __init__(self):
        self.params = {}
        self.best_params = {}
        self.classes = None

        # Load and preprocess the data
        X, y = fetch_openml('mnist_784', version=1, return_X_y=True)
        X = X / 255.  # normalize data
        y = y.astype(int)  # convert string to int

        # Split the data into training and testing sets
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, y, test_size=10000,
                                                                                random_state=1)
        self.X_train, self.y_train, self.X_test, self.y_test = map(cp.asarray, [self.X_train, self.y_train,
                                                                                self.X_test, self.y_test])

    def fit(self, X_train, y_train):
        self.classes = cp.unique(y_train)
        for c in self.classes:
            X_c = X_train[y_train == c]
            E_X = cp.mean(X_c, axis=0)
            Var_X = cp.var(X_c, axis=0)
            K = (E_X * (1 - E_X)) / Var_X - 1
            self.params[c.item()] = {
                'alpha': K * E_X,
                'beta': K * (1 - E_X)
            }
            # self.params[c] = {
            #     'alpha': self.alpha + cp.sum(X_c, axis=0),
            #     'beta': self.beta + cp.sum(1 - X_c, axis=0)
            # }

    def validate(self, X):
        posteriors = []
        for c in self.classes:
            log_prior = cp.log(cp.mean(self.y_train == c))
            c = c.item()
            log_likelihood = cp.sum(gammaln(X + self.params[c]['alpha']) + gammaln(1 - X + self.params[c]['beta']) -
                                    gammaln(X + 1) - gammaln(1 - X + 1) -
                                    gammaln(self.params[c]['alpha'] + self.params[c]['beta']) +
                                    gammaln(self.params[c]['alpha']) + gammaln(self.params[c]['beta']), axis=1)
            posteriors.append(log_prior + log_likelihood)
        return self.classes[cp.argmax(cp.array(posteriors), axis=0)]

    def predict(self, X):
        self.params = self.best_params
        return self.validate(X)

    def cross_validate(self):
        kf = KFold(n_splits=10, shuffle=True, random_state=1)
        accuracies = []
        best_accuracy = 0
        iteration = 0
        for train_index, val_index in kf.split(self.X_train):
            start_time = time.perf_counter()
            X_train, X_val = self.X_train[train_index], self.X_train[val_index]
            y_train, y_val = self.y_train[train_index], self.y_train[val_index]
            self.fit(X_train, y_train)
            y_pred = self.validate(X_val)
            accuracy = accuracy_score(y_val.get(), y_pred.get())
            accuracies.append(accuracy)
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                self.best_params = self.params
            print(f'Fold {iteration} time: {time.perf_counter() - start_time} seconds')
            print(f'Fold {iteration} Accuracy: {accuracy}')
            iteration += 1
        print(f'Average Cross-Validation Accuracy: {cp.mean(cp.array(accuracies))}')

    def train(self):
        self.cross_validate()
        start_time = time.perf_counter()
        y_pred = self.predict(self.X_test)
        print(f'Testing time: {time.perf_counter() - start_time} seconds')
        print(f'Test Accuracy: {accuracy_score(self.y_test.get(), y_pred.get())}')
