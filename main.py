import models.LinearSvm as LSvm
import models.Svm as Svm
import models.RandForest as rf
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt


def linear_svm():
    classifier = LSvm.LinearSvm({'C': [0.01, 0.1, 1]})  # create an instance of the Svm class
    linear_estimator, linear_params = classifier.train()  # train the model
    print("Best linear params: ", linear_params)  # print the best hyperparameters
    print(classification_report(classifier.y_test, classifier.test()))
    df = classifier.get_results()
    df.to_excel('linear_svm.xlsx')  # save the results to an excel file
    df.plot()


def poly_svm():
    classifier = Svm.Svm({'C': [0.01, 0.1, 1], 'kernel': ['poly'], 'degree': [2]})
    poly_estimator, poly_params = classifier.train()
    print("Best polynomail params: ", poly_params)
    print(classification_report(classifier.y_test, classifier.test()))
    df = classifier.get_results()
    df.to_excel('poly_svm.xlsx')
    df.plot()


def rbf_svm():
    classifier = Svm.Svm({'C': [0.01, 0.1, 1], 'kernel': ['rbf']})
    rbf_estimator, rbf_params = classifier.train()
    print("Best rbf params: ", rbf_params)
    print(classification_report(classifier.y_test, classifier.test()))
    df = classifier.get_results()
    df.to_excel('rbf_svm.xlsx')
    df.plot()


def random_forest():
    classifier = rf.RandForest({'n_estimators': [10, 100, 1000], 'criterion': ['log_loss', 'entropy'],
                                'max_features': ['sqrt', 'log2', None], 'bootstrap': [True, False]})
    rf_estimator, rf_params = classifier.train()
    print("Best random forest params: ", rf_params)
    print(classification_report(classifier.y_test, classifier.test()))
    df = classifier.get_results()
    df.to_excel('random_forest.xlsx')
    df.plot()


if __name__ == '__main__':
    # linear_svm()
    # poly_svm()
    # rbf_svm()
    random_forest()
    plt.show()
