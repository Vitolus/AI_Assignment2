import models.LinearSvm as LSvm
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt

if __name__ == '__main__':
    hyperparams = {'C': [0.01, 0.1, 1, 10]}  # C is the regularization parameter
    svm_clf = LSvm.LinearSvm(hyperparams)  # create an instance of the Svm class
    estimator, params = svm_clf.train()  # train the model
    print("Best params: ", params)  # print the best hyperparameters
    print(classification_report(svm_clf.y_test, svm_clf.test()))
    df = svm_clf.get_results()
    df.to_csv('results/linear_svm.csv', index=False)  # save the results to a csv file
    df.plot()
    plt.show()
