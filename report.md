Write a handwritten digit classifier for the MNIST database. These are composed of 70000 28x28 pixel gray-scale images
of handwritten digits divided into 60000 training set and 10000 test set.

Train the following classifiers on the dataset:

* SVM  using linear, polynomial of degree 2, and RBF kernels;
* Random forests
* Naive Bayes classifier where each pixel is distributed according to a Beta distribution of parameters α, β:
$$ d(x; a, b)=\frac{\Gamma(\alfa+\beta)}{\Gamma(\alfa)\Gamma(\beta)}x^\alfa-1^(1-x)\beta-1 $$
* k-NN

Use 10 way cross validation to optimize the parameters for each classifier.