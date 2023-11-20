from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split


class Classifier:
    def __init__(self):
        x, y = fetch_openml('mnist_784', version=1, return_X_y=True)  # load data
        y = y.astype(int)  # convert string to int
        x = x / 255.  # normalize data
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            x, y, test_size=0.2, random_state=42)  # split data
