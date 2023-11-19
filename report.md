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