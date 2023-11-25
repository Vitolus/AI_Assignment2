import models.LinearSvm as LSvm
import models.Svm as Svm
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt


def linear_svm():
    svm_clf = LSvm.LinearSvm({'C': [0.01, 0.1, 1]})  # create an instance of the Svm class
    linear_estimator, linear_params = svm_clf.train()  # train the model
    print("Best linear params: ", linear_params)  # print the best hyperparameters
    print(classification_report(svm_clf.y_test, svm_clf.test()))
    df = svm_clf.get_results()
    df.to_excel('linear_svm.xlsx')  # save the results to an excel file
    df.plot()


def poly_svm():
    svm_clf = Svm.Svm({'C': [0.01, 0.1, 1], 'kernel': 'poly', 'degree': 2})
    poly_estimator, poly_params = svm_clf.train()
    print("Best polynomail params: ", poly_params)
    print(classification_report(svm_clf.y_test, svm_clf.test()))
    df = svm_clf.get_results()
    df.to_excel('poly_svm.xlsx')
    df.plot()


def rbf_svm():
    svm_clf = Svm.Svm({'C': [0.01, 0.1, 1], 'kernel': 'rbf'})
    rbf_estimator, rbf_params = svm_clf.train()
    print("Best rbf params: ", rbf_params)
    print(classification_report(svm_clf.y_test, svm_clf.test()))
    df = svm_clf.get_results()
    df.to_excel('rbf_svm.xlsx')
    df.plot()


if __name__ == '__main__':
    linear_svm()
    poly_svm()
    rbf_svm()
    plt.show()
