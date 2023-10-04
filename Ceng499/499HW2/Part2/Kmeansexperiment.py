import math
from Part2.KMeans import KMeans
import pickle
import matplotlib.pyplot as plt
import numpy as np

dataset1 = pickle.load(open("../data/part2_dataset_1.data", "rb"))
dataset2 = pickle.load(open("../data/part2_dataset_2.data", "rb"))
ks = [2, 3, 4, 5, 6, 7, 8, 9, 10]
ls1 = [[], [], [], [], [], [], [], [], []]  #  first two will be empty
ls2 = [[], [], [], [], [], [], [], [], []]
means1 = []
means2 = []

for k in ks:
    print('K:%d' % k)
    i = 0
    while i < 10:
        min_loss1 = math.inf
        min_loss2 = math.inf
        j = 0
        while j < 10:
            model1 = KMeans(dataset1, k)
            model2 = KMeans(dataset2, k)
            centers1, clusters1, loss1 = model1.run()
            centers2, clusters2, loss2 = model2.run()
            if min_loss1 > loss1:
                min_loss1 = loss1
            if min_loss2 > loss2:
                min_loss2 = loss2
            j += 1
        ls1[k-2].append(min_loss1)
        ls2[k-2].append(min_loss2)
        i += 1
i = 0
while i < 9:
    mean = 0
    if len(ls1[i]) > 0:
        mean = np.mean(np.array(ls1[i]))
        print(('(dataset1) Confidence interval k=%d: %.3f ' + '+-' + ' %.3f') % (i+2, mean, 1.96 * np.std(ls1[i]) / (math.sqrt(len(ls1[i])))))
    means1.append(mean)
    i += 1

i = 0
while i < 9:
    mean2 = 0
    if len(ls2[i]) > 0:
        mean2 = np.mean(np.array(ls2[i]))
        print(('(dataset2) Confidence interval k=%d: %.3f ' + '+-' + ' %.3f') % (i+2, mean2, 1.96 * np.std(ls2[i]) / (math.sqrt(len(ls2[i])))))
    means2.append(mean2)
    i += 1


plt.plot(ks, means1)
plt.ylabel('Loss')
plt.xlabel('Cluster count')
plt.title('Cluster-Loss graph of Dataset1')
plt.show()
plt.plot(ks, means2)
plt.ylabel('Loss')
plt.xlabel('Cluster count')
plt.title('Cluster-Loss graph of Dataset2')
plt.show()
