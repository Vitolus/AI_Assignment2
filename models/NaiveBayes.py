import cupy as cp
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split, KFold
from scipy.special import gamma
from sklearn.metrics import accuracy_score
import time
import matplotlib.pyplot as plt


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
        self.X_train, self.y_train, self.X_test, self.y_test = map(cp.asarray, [self.X_train, self.y_train,
                                                                                self.X_test, self.y_test])
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
                X_where_c = X_fold_train[cp.where(y_fold_train == c)]  # Get all training samples with label c
                E_X = cp.mean(X_where_c, axis=0)  # Compute the mean of the training samples with label c
                Var_X = cp.var(X_where_c, axis=0)  # Compute the variance of the training samples with label c
                K = (E_X * (1 - E_X)) / Var_X - 1  # Compute the K value for the Beta distribution
                alpha = K * E_X  # Compute the alpha parameter for the Beta distribution
                beta = K * (1 - E_X)  # Compute the beta parameter for the Beta distribution
                self.params.append((alpha, beta))  # Store the parameters for the Beta distribution
            y_pred_val = self.predict(X_fold_val)  # Make predictions
            accuracy_fold = accuracy_score(y_fold_val.get(), y_pred_val.get())  # Compute accuracy
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

    def predict(self, X):
        N = X.shape[0]
        C = len(self.classes)
        P = cp.zeros((N, C))
        for i in range(C):
            alpha, beta = self.params[i]
            P[:, i] = cp.sum((gamma(alpha + beta) / (gamma(alpha) * gamma(beta))) * (X ** (alpha - 1)) *
                             ((1 - X) ** (beta - 1)), axis=1)
        return self.classes[cp.argmax(P, axis=1)]

    def _visualize(self):
        # Compute the mean of the Beta distribution for each pixel
        means = cp.array([a / (a + b) for a, b in self.params])
        # Reshape the means into 28x28 images
        images = means.reshape(-1, 28, 28)
        # Plot the images
        fig, axs = plt.subplots(2, 5, figsize=(10, 4))
        for i, ax in enumerate(axs.flat):
            ax.imshow(images[i].get(), cmap='gray')
            ax.axis('off')
        plt.show()
