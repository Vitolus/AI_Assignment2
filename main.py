import models.Svm as svm


if __name__ == '__main__':
    hyperparams = {'C': [1, 10, 100, 1000], 'gamma': [0.1, 0.01, 0.001, 0.0001]}
    model = svm.Svm(hyperparams)
    model.train()
    model.test()
