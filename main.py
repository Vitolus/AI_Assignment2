import models.LinearSvm as LSvm
import models.Svm as Svm
import models.RandForest as Rf
import models.Svc as Svc
from sklearn.metrics import classification_report
from sklearn.datasets import load_breast_cancer
from atom import ATOMClassifier
from sklearn.datasets import fetch_openml


def linear_svm():
    classifier = LSvm.LinearSvm({"C": [0.01, 0.1, 1]})  # create an instance of the Svm class
    linear_estimator, linear_params = classifier.train()  # train the model
    print("Best linear params: ", linear_params)  # print the best hyperparameters
    df = classifier.get_results()  # get the results of the grid search
    df.to_excel('linear_svm.xlsx')  # save the results to an Excel file


def poly_svm():
    # create an instance of the Svm class
    classifier = Svm.Svm({'C': [0.01, 0.1, 1], 'kernel': ['poly'], 'degree': [2]})
    poly_estimator, poly_params = classifier.train()  # train the model
    print("Best polynomail params: ", poly_params)  # print the best hyperparameters
    print(classification_report(classifier.y_test, classifier.test()))  # print the classification report
    df = classifier.get_results()  # get the results of the grid search
    df.to_excel('poly_svm.xlsx')  # save the results to an Excel file


def rbf_svm():
    classifier = Svm.Svm({'C': [0.01, 0.1, 1], 'kernel': ['rbf']})  # create an instance of the Svm class
    rbf_estimator, rbf_params = classifier.train()  # train the model
    print("Best rbf params: ", rbf_params)  # print the best hyperparameters
    print(classification_report(classifier.y_test, classifier.test()))  # print the classification report
    df = classifier.get_results()  # get the results of the grid search
    df.to_excel('rbf_svm.xlsx')  # save the results to an Excel file


def random_forest():
    # create an instance of the RandForest class
    classifier = Rf.RandForest({'n_estimators': [100, 1000], 'criterion': ['log_loss'], 'bootstrap': [False]})
    rf_estimator, rf_params = classifier.train()  # train the model
    print("Best random forest params: ", rf_params)  # print the best hyperparameters
    print(classification_report(classifier.y_test, classifier.test()))  # print the classification report
    df = classifier.get_results()  # get the results of the grid search
    df.to_excel('results/random_forest.xlsx')  # save the results to an Excel file

def svc():
    classifier = Svc.Svc()  # create an instance of the Svc class
    svc_params = classifier.train()  # train the model
    print("Best svc params: ", svc_params)  # print the best hyperparameters

## TEST METHOD FOR ATOM-ML AND CUML LIBRARIES
def atom_test():
    # X, y = load_breast_cancer(return_X_y=True, as_frame=True)
    X, y = fetch_openml('mnist_784', version=1, return_X_y=True)  # load data
    y = y.astype(int)  # convert string to int
    X = X / 255.  # normalize data
    # Initialize atom
    atom = ATOMClassifier(X, y, n_jobs=-1, logger="auto" , device="gpu", engine="cuml", verbose=2)
    atom.feature_selection(strategy="pca")
    # Train models
    atom.run(
        models=["LR", "RF", "XGB"],
        metric="accuracy"
    )
    # Analyze the results
    atom.evaluate()


if __name__ == '__main__':
    svc()
    # atom_test()
    # linear_svm()  # call the linear_svm function
    # poly_svm()  # call the poly_svm function
    # rbf_svm()  # call the rbf_svm function
    # random_forest()  # call the random_forest function

