import os
import gc
import codecs
import numpy as np
import math
from sklearn.cluster import KMeans
import platform

NEW_LINE = '\n'
if platform.system() == 'Windows':
    NEW_LINE = '\r\n'
    
vector_size = 0

def cal_len(vec):
    vec = np.mat(vec)
    num = (float)(vec * vec.T)
    return math.sqrt(num)

def norm(vec):
    vec = np.mat(vec)
    return vec / cal_len(vec)

def cos_sim(v1, v2):
    v1 = np.mat(v1)
    v2 = np.mat(v2)
    num = (float)(v1 * v2.T)
    return num

def loadRN(filePath, encoding='utf-8'):
    global vector_size
    f = codecs.open(filePath, 'r', encoding)
    content = f.read()
    f.close()
    RN_list = content.split(NEW_LINE)
    RN_list.remove(RN_list[-1])
    vector_size = int(RN_list[0])
    RN_list.remove(RN_list[0])
    RN_id = []
    RN_vector = []
    a = 1
    for vec in RN_list:
        parts = vec.split(' ')
        RN_id.append(int(parts[0]))
        temp_vector = [0 for ii in range(0, vector_size)]
        for j in range(1, len(parts)):
            temp = parts[j].split(':')
            temp_vector[ int(temp[0]) ] = float(temp[1])
        RN_vector.append(temp_vector)
        print('load RN_vector', a)
        a += 1
    return RN_id, RN_vector


if __name__ == '__main__':
    RN_id, RN_vector = loadRN('./InputFile/RN_vector.txt')
    print('load all')

    # 聚类，重新找负例
    # RN_vector = np.array(RN_vector)
    # print(RN_vector)

    print('cluster...')
    n_clusters = 8
    kmeans = KMeans(n_clusters=n_clusters, random_state=0).fit(RN_vector)
    labels = kmeans.labels_.tolist()
    print('...finished')
    # print('kmeans labels:', labels)

    f = codecs.open('./InputFile/cluster.txt', 'w', 'utf-8')
    for l in labels:
        f.write(str(l) + ' ')
    f.close()