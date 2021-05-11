from sklearn.svm import SVC
from sklearn.metrics import euclidean_distances


def triangular_kernel(X, Y):
    return 1 - abs(euclidean_distances(X, Y))


triangular = SVC(kernel=triangular_kernel, C=3)
rbf = SVC()

classifiers = {
    'SVM_triangular_kernel': triangular,
    'SVM_RBF_kernel': rbf
}

