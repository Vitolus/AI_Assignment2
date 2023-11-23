import models.Svm as Svm


if __name__ == '__main__':
    hyperparams = {'C': [0.1, 1]}  # C is the regularization parameter
    svm_clf = Svm.Svm(hyperparams)  # create an instance of the Svm class
    estimator, params = svm_clf.train()  # train the model
    print(svm_clf.test())  # test the model
    print(svm_clf.get_results())  # print the results
