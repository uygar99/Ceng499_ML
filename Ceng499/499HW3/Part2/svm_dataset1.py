import pickle
from matplotlib import pyplot as plt
from sklearn import svm
import numpy as np

dataset, labels = pickle.load(open("../data/part2_dataset1.data", "rb"))
C = [1, 5, 10, 20]
linear_SVMs = []
polynomial_SVMs = []
rbf_SVMs = []
for c in C:
    svm_linear = svm.SVC(kernel='linear', C=c)
    svm_linear = svm_linear.fit(dataset, labels)
    linear_SVMs.append(svm_linear)
    svm_polynomial = svm.SVC(kernel='poly', degree=3, C=c)
    svm_polynomial = svm_polynomial.fit(dataset, labels)
    polynomial_SVMs.append(svm_polynomial)
    svm_rbf = svm.SVC(kernel='rbf', C=c)
    svm_rbf = svm_rbf.fit(dataset, labels)
    rbf_SVMs.append(svm_rbf)

# I partially used the code in scikit learn documentation
# https://scikit-learn.org/0.18/auto_examples/svm/plot_iris.html

h = .01
x_min, x_max = dataset[:, 0].min() - 1, dataset[:, 0].max() + 1
y_min, y_max = dataset[:, 1].min() - 1, dataset[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                     np.arange(y_min, y_max, h))

# title for the plots
titles = ['SVC with linear kernel',
          'SVC with RBF kernel',
          'SVC with polynomial (degree 3) kernel']

for index in range(4):
    for i, clf in enumerate((linear_SVMs[index], rbf_SVMs[index], polynomial_SVMs[index])):
        plt.subplot(2, 2, i + 1)
        plt.subplots_adjust(wspace=0.4, hspace=0.4)
        Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
        Z = Z.reshape(xx.shape)
        plt.contourf(xx, yy, Z, cmap=plt.cm.coolwarm, alpha=0.8)
        plt.scatter(dataset[:, 0], dataset[:, 1], c=labels, cmap=plt.cm.coolwarm)
        plt.xlabel('Sepal length')
        plt.ylabel('Sepal width')
        plt.xlim(xx.min(), xx.max())
        plt.ylim(yy.min(), yy.max())
        plt.xticks(())
        plt.yticks(())
        plt.title(titles[i] + " when C= " + str(C[index]))
    plt.show()
