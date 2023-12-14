from models import (LinearSvc as LSvc,
                    PolySvc as PSvc,
                    RbfSvc as RSvc,
                    RandForest as RForest,
                    Knn)


def linear_svc():
    classifier = LSvc.LinearSvc()
    classifier.train()


def poly_svc():
    classifier = PSvc.PolySvc()
    classifier.train()


def rbf_svc():
    classifier = RSvc.RbfSvc()
    classifier.train()


def random_forest():
    classifier = RForest.RandForest()
    classifier.train()


def k_nn():
    classifier = Knn.Knn(n_neighbors=5)
    classifier.fit()
    classifier.predict()


if __name__ == '__main__':
    # linear_svc()
    # poly_svc()
    # rbf_svc()
    # random_forest()
    k_nn()
