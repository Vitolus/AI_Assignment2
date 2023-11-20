# 1. Requirements

Write a handwritten digit classifier for the MNIST database. These are composed of 70000 28x28 pixel gray-scale images
of handwritten digits divided into 60000 training set and 10000 test set.

Train the following classifiers on the dataset:

* SVM using linear, polynomial of degree 2, and RBF kernels;
* Random forests
* Naive Bayes classifier where each pixel is distributed according to a Beta distribution of parameters &alpha;,
  &beta;:\
  d(x; a, b)=
  <div class="frac"><span>&Gamma;(&alpha;+&beta;)</span>
    <span class="symbol">/</span>
    <span class="bottom">&Gamma;(&alpha;)&Gamma;(&beta;)</span></div>
  x<sup>&alpha;-1</sup>(1-x)<sup>&beta;-1</sup>
* k-NN
  Use 10 way cross validation to optimize the parameters for each classifier.

# 2. Introduction

# 3. Deisgn the solution

## 3.1. SVM

There are several hyperparameters that we can tune:

* `C`: This is the regularization parameter, also known as the cost parameter. This tells the SVM optimization how much
  you want to avoid misclassifying each training example. A smaller value of C creates a wider margin, which may allow
  more misclassifications. A larger C creates a narrower margin and thus may reduce the number of misclassifications.

* `kernel`: This specifies the kernel type to be used in the algorithm. It could be ‘linear’, ‘poly’, ‘rbf’, ‘sigmoid’,
  or a custom function. The choice of the kernel function can have a significant impact on the performance of the model.

* `degree`: This is the degree of the polynomial kernel function (‘poly’) and is ignored by all other kernels. It
  essentially controls the complexity of the model.

* `gamma`: This defines how far the influence of a single training example reaches, with low values meaning ‘far’ and
  high values meaning ‘close’. It can be seen as the inverse of the radius of influence of samples selected by the model
  as support vectors.

* `coef0`: This is the independent term in the kernel function. It is only significant in ‘poly’ and ‘sigmoid’.

* `shrinking`: This is a boolean that determines whether to use the shrinking heuristic. The shrinking heuristic is a
  technique that removes some of the constraints in the problem and makes the computations faster.

* `probability`: This is a boolean that determines whether to enable probability estimates. This must be enabled prior
  to calling fit, and will slow down that method.

* `tol`: This is the tolerance for stopping criterion.

* `class_weight`: This allows you to specify weights for the classes. This can be useful if your classes are imbalanced.

### 3.1.1. Linear kernel

#### 3.1.1.1. Formalization

Both `LinearSVC` and `SVC(kernel='linear')` can be used for binary classification problems (i.e., when you have only 2
classes). The choice between the two often depends on the size of your dataset and the specific requirements of your
problem.

* `LinearSVC` uses a one-vs-rest strategy for multi-class problems, but for binary classification, this distinction does
  not matter. It can be faster to train on large datasets because it scales linearly with the number of data points.

* `SVC(kernel='linear')`, on the other hand, uses a one-vs-one strategy for multi-class problems. For binary
  classification, this is equivalent to one-vs-rest. `SVC(kernel='linear')` can be slower on large datasets because the
  training time scales quadratically with the number of data points.

In terms of the decision boundary, both `LinearSVC` and `SVC(kernel='linear')` will produce the same results because
they both use a linear kernel.

So, if you have a large dataset, `LinearSVC` might be a better choice due to its efficiency.

