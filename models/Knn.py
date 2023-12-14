import cupy as cp
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.metrics import pairwise_distances, accuracy_score
import time


class Knn:

    def __init__(self, n_neighbors=5):
        # Load data
        X, y = fetch_openml('mnist_784', version=1, return_X_y=True)
        y = y.astype(int)  # convert string to int
        X = X / 255.  # normalize data
        self.n_neighbors = n_neighbors

        # Use a subset of the data (10,000 samples) for both training and testing
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, y, test_size=10000, random_state=1)

    def fit(self):
        # Convert data to cupy arrays for GPU operations
        self.X_train = cp.array(self.X_train)
        self.y_train = cp.array(self.y_train)
        self.X_test = cp.array(self.X_test)
        self.y_test = cp.array(self.y_test)

        # Perform 10-fold cross-validation
        kf = KFold(n_splits=10, shuffle=True, random_state=1)

        accuracy_list = []
        iteration = 0
        for train_index, val_index in kf.split(self.X_train):
            start_time = time.perf_counter()
            X_fold_train, X_fold_val = self.X_train.take(train_index, axis=0), self.X_train.take(val_index, axis=0)
            y_fold_train, y_fold_val = self.y_train.take(train_index), self.y_train.take(val_index)

            # Compute pairwise distances
            distances = pairwise_distances(X_fold_val.get(), X_fold_train.get())

            # Find k-nearest neighbors
            knn_indices = cp.argpartition(distances, kth=self.n_neighbors, axis=1)[:, :self.n_neighbors]
            knn_labels = cp.take(y_fold_train, knn_indices)

            # Make predictions by choosing the most common label among neighbors
            y_pred_val = []
            for labels in knn_labels:
                y_pred_val.append(cp.argmax(cp.bincount(labels)))
            y_pred_val = cp.array(y_pred_val)

            # Compute accuracy for this fold
            accuracy_fold = accuracy_score(y_fold_val.get(), y_pred_val.get())
            accuracy_list.append(accuracy_fold)
            print(f'Training Fold {iteration} time: {time.perf_counter() - start_time} seconds')
            print(f'Training Fold {iteration} Accuracy: {accuracy_fold}')
            iteration += 1

        # Compute average accuracy across all folds
        average_accuracy = cp.mean(cp.array(accuracy_list))
        print(f'Average Training Cross-Validation Accuracy: {average_accuracy.get()}')

    def predict(self):
        start_time = time.perf_counter()
        # Compute pairwise distances
        distances = pairwise_distances(self.X_test.get(), self.X_train.get())

        # Find k-nearest neighbors
        knn_indices = cp.argpartition(distances, kth=self.n_neighbors, axis=1)[:, :self.n_neighbors]
        knn_labels = cp.take(self.y_train, knn_indices)

        # Make predictions by choosing the most common label among neighbors
        y_pred_test = []
        for labels in knn_labels:
            y_pred_test.append(cp.argmax(cp.bincount(labels)))
        y_pred_test = cp.array(y_pred_test)

        # Compute accuracy on the test set
        test_accuracy = accuracy_score(self.y_test.get(), y_pred_test.get())
        print(f'Test Set time: {time.perf_counter() - start_time} seconds')
        print(f'Test Set Accuracy: {test_accuracy}')
