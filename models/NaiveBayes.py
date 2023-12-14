import cupy as cp
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split, KFold
from scipy.special import gamma
from sklearn.metrics import accuracy_score
import time


class NaiveBayes:

    def __init__(self):
        # Load data
        X, y = fetch_openml('mnist_784', version=1, return_X_y=True)
        y = y.astype(int)  # convert string to int
        X = X / 255.  # normalize data
        # Use a subset of the data (10,000 samples) for both training and testing
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, y, test_size=10000,
                                                                                random_state=1)
        # Convert data to cupy arrays for GPU operations
        self.X_train = cp.array(self.X_train)
        self.y_train = cp.array(self.y_train)
        self.X_test = cp.array(self.X_test)
        self.y_test = cp.array(self.y_test)
        self.params = []
        self.classes = cp.unique(self.y_train)

    def fit(self):
        # Perform 10-fold cross-validation
        kf = KFold(n_splits=10, shuffle=True, random_state=1)
        accuracy_list = []
        best_accuracy = 0
        best_params = None
        iteration = 0
        for train_index, val_index in kf.split(self.X_train):
            start_time = time.perf_counter()
            X_fold_train, X_fold_val = self.X_train.take(train_index, axis=0), self.X_train.take(val_index, axis=0)
            y_fold_train, y_fold_val = self.y_train.take(train_index), self.y_train.take(val_index)
            self.params = []
            for i, c in enumerate(self.classes):
                X_where_c = X_fold_train[cp.where(y_fold_train == c)]
                E_X = cp.mean(X_where_c, axis=0)
                Var_X = cp.var(X_where_c, axis=0)
                K = (E_X * (1 - E_X)) / Var_X - 1
                alpha = K*E_X
                beta = K*(1-E_X)
                self.params.append((alpha, beta))
            N = X_fold_val.shape[0]
            C = len(self.classes)
            P = cp.zeros((N, C))
            for i in range(C):
                alpha, beta = self.params[i]
                P[:, i] = cp.sum(cp.log(gamma(alpha + beta)) - cp.log(gamma(alpha)) - cp.log(gamma(beta)) + (alpha - 1)
                                 * cp.log(X_fold_val) + (beta - 1) * cp.log(1 - X_fold_val), axis=1)
            y_pred_val = self.classes[cp.argmax(P, axis=1)]
            # Compute accuracy for this fold
            accuracy_fold = accuracy_score(y_fold_val.get(), y_pred_val.get())
            accuracy_list.append(accuracy_fold)
            print(f'Training Fold {iteration} time: {time.perf_counter() - start_time} seconds')
            print(f'Training Fold {iteration} Accuracy: {accuracy_fold}')
            # If this fold's accuracy is the best so far, store its parameters
            if accuracy_fold > best_accuracy:
                best_accuracy = accuracy_fold
                best_params = self.params
            iteration += 1
        # After all folds, set the model's parameters to the best ones found
        self.params = best_params
        # Compute average accuracy across all folds
        average_accuracy = cp.mean(cp.array(accuracy_list))
        print(f'Average Training Cross-Validation Accuracy: {average_accuracy.get()}')

    def predict(self):
        start_time = time.perf_counter()
        N = self.X_test.shape[0]
        C = len(self.classes)
        P = cp.zeros((N, C))
        for i in range(C):
            alpha, beta = self.params[i]
            P[:, i] = cp.sum(cp.log(gamma(alpha + beta)) - cp.log(gamma(alpha)) - cp.log(gamma(beta)) + (alpha - 1)
                             * cp.log(self.X_test) + (beta - 1) * cp.log(1 - self.X_test), axis=1)
        y_pred_test = self.classes[cp.argmax(P, axis=1)]
        accuracy = accuracy_score(self.y_test.get(), y_pred_test.get())
        print(f'Test Set time: {time.perf_counter() - start_time} seconds')
        print(f'Test Set Accuracy: {accuracy}')
