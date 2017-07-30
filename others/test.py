import jieba
import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans
from scipy.spatial.distance import cdist

import mytfidf

# 小函数
def cal_len(vec):
    vec = np.mat(vec)
    num = (float)(vec * vec.T)
    return np.sqrt(num)

def norm(vec):
    vec = np.array(vec)
    return vec / cal_len(vec)
# 小函数


# X = np.mat([[1, 1], [1, 2], [5, 5], [6, 5], [0, 0], [1, -1], [-1, -1]])
# print(X)

# n_clusters = 2
# kmeans = KMeans(n_clusters=n_clusters, random_state=0).fit(X)

# print(kmeans.labels_.tolist())
# print(kmeans.predict([6, 6]))
# print(kmeans.cluster_centers_)


# for i in range(0, 10):
#     for j in range(0, 5):
#         if j > 3:
#             break
#         print(i, ', ', j)


# a = [[1, 0, 0], [0, 0, 0], [1, 1, 1], [0, 0, 0]]


mytfidf.myprint('hello')



'''
import numpy as np
from functools import reduce
from operator import add

from collections import Counter

def norm(vec):
    vec = np.mat(vec)
    return vec / cal_len(vec)

def cal_len(vec):
    vec = np.mat(vec)
    num = (float)(vec * vec.T)
    return np.sqrt(num)

def cos_sim(v1, v2):
    v1 = np.mat(v1)
    v2 = np.mat(v2)
    num = (float)(v1 * v2.T)
    print('num = ' + str(num))
    print('v1.len = ' + str(cal_len(v1)))
    print('v2.len = ' + str(cal_len(v2)))
    return num / (cal_len(v1) * cal_len(v2))

    


# list = [1, 1, 1]
# a = np.matrix(list)

if __name__ == '__main__':
    # v1 = np.array([1, 1, 1])
    # v2 = np.array([2, 2, 2])


    # f1 = open('./word_set.txt', 'r', encoding='utf-8')
    # content = f1.read()
    # f1.close()
    # aList = content.split('\n')
    # aList.remove(aList[-1])
    # for i in range(0, len(aList)):
    #     aList[i] = aList[i].split(' ')
    #     aList[i] = aList[i][0]

    # f2 = open('./word_set_1.txt', 'r', encoding='utf-8')
    # content = f2.read()
    # f2.close()
    # bList = content.split('\n')
    # bList.remove(bList[-1])
    # for i in range(0, len(bList)):
    #     bList[i] = bList[i].split(' ')
    #     bList[i] = bList[i][0]

    # mstr = ''
    # for a in aList:
    #     if a in bList:
    #         pass
    #     else:
    #         mstr += a + ' '
    # print(mstr)
    

    a = np.array([1, 1, 1])
    print(a)

    b = a.tolist()
    print(b)



    # # print(aCounter)
    # # i = 0
    # for a in aCounter:
    #     print(a + '\t' + str(aCounter[a]))
    #     # if int(a) != 1:
    #     #     print('yes')
    '''