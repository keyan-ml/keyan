import os
import gc
import codecs
import numpy as np
import math
from sklearn.cluster import KMeans

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

def loadp(filePath, encoding='utf-8'):
    f = codecs.open(filePath, 'r', encoding)
    content = f.read()
    f.close()
    p_list = content.split(NEW_LINE)
    p_list.remove(p_list[-1])
    vector_size = int(p_list[0])
    p_list.remove(p_list[0])
    p_id = []
    p_vector = []
    a = 1
    for vec in p_list:
        parts = vec.split(' ')
        p_id.append(int(parts[0]))
        temp_vector = [0 for ii in range(0, vector_size)]
        for j in range(1, len(parts)):
            temp = parts[j].split(':')
            temp_vector[ int(temp[0]) ] = float(temp[1])
        p_vector.append(temp_vector)
        print('load p_vector', a)
        a += 1
    return p_id, p_vector

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

def loadLabels(filePath, encoding='utf-8'):
    f = codecs.open(filePath, 'r', encoding)
    content = f.read()
    f.close()
    labels_list = content.split(' ')
    if labels_list[-1] == '':
        labels_list.remove(labels_list[-1])
    for i in range(0, len(labels_list)):
        labels_list[i] = int(labels_list[i])
    return labels_list


def getCenter(p_vector, u_vector):
    Alpha = 16
    Beta = 4
    pos_num = len(p_vector)
    unlabel_num = len(u_vector)

    p_vector = np.mat(p_vector)
    p_sum = p_vector.sum( axis=0 )
    u_vector = np.mat(u_vector)
    u_sum = u_vector.sum( axis=0 )

    p_center = Alpha * p_sum / pos_num - Beta * u_sum / unlabel_num
    u_center = Alpha * u_sum / unlabel_num - Beta * p_sum / pos_num

    p_center = norm(p_center)
    u_center = norm(u_center)
    return p_center, u_center


if __name__ == '__main__':
    RN_id, RN_vector = loadRN('./InputFile/RN_vector.txt')
    labels = loadLabels('./InputFile/cluster.txt')
    n_clusters = 8
    
    # N = [[] for ii in range(0, n_clusters)] # 为分类计算 p u 中心向量而设，不需记录 id

    # # RN_vector = RN_vector.tolist()
    # for i in range(0, len(labels)):
    #     # N_id[labels[i]].append(i)
    #     N[labels[i]].append(RN_vector[i])

    # p_center_pre = p_center
    p_id, p_vector = loadp("./InputFile/p_vector.txt")
    # 计算各组的中心向量
    print('cal center...')
    p_center = [[] for i in range(0, n_clusters)]
    u_center = [[] for i in range(0, n_clusters)]
    for i in range(0, n_clusters):
        N = []
        for j in range(0, len(labels)):
            if labels[j] == i:
                N.append(RN_vector[j])
        p_center[i], u_center[i] = getCenter(p_vector, N)
        print('cal center', i, '/', n_clusters)
    # 计算各组的中心向量
    
    print('get RN again...')
    pre_cnt = len(RN_id)
    # 计算余弦距离，重新确定可靠负例
    a = 0
    i = 0
    # print(len(RN_id), len(RN_vector))
    # exit()
    while i < len(RN_vector):
        print('process neg', a)
        a += 1
        d = RN_vector[i]
        # 找到距离当前文档最近的正例中心向量
        temp_dist = []
        for j in range(0, len(p_center)):
            temp_dist.append(cos_sim(d, p_center[j]))
        p_d_dist = max(temp_dist)
        # 找到距离当前文档最近的正例中心向量
        flag = False
        for u in u_center:
            u_d_dist = cos_sim(d, u)
            if p_d_dist < u_d_dist:
                flag = True
                break
        if flag:
            i += 1
            continue
        else:
            RN_id.remove(RN_id[i])
            RN_vector.remove(RN_vector[i])
    # 计算余弦距离，重新确定可靠负例
    print('...finished')

    print('Here is them: ', len(RN_id), '/', pre_cnt)
    # print(RN_id)
    # for i in range(0, len(RN_id)):
    #     print('id: ' + str(i))
        # print(RN_vector[i], '\n')
    # 聚类，重新找负例
    
    # for i in range(0, len(p_vector)):
    #     p_vector[i] = (np.array(p_vector[i]).tolist())[0]
    # for i in range(0, len(RN_vector)):
    #     RN_vector[i] = (np.array(RN_vector[i]).tolist())[0]
    # print(p_vector)
    # print(RN_vector)

    # fw = codecs.open('./p_vector.txt', 'w', 'utf-8')
    # for doc in p_vector:
    #     for w in doc:
    #         fw.write(w + ' ')
    #     fw.write(NEW_LINE)
    
    
    # p_center = p_center_pre
    # best_d = []
    # best_vector = []
    # # to be continue...
    '''
    '''
    fw = codecs.open('../svm/train.txt', 'w', 'utf-8')
    a = 1
    for doc in p_vector:
        out_str = '1'
        for i in range(0, len(doc)):
            if doc[i] != 0:
                out_str += ' ' + str(i + 1) + ':' + str(doc[i])
        if out_str != '1':
            fw.write(out_str)
            fw.write(NEW_LINE)
            print('write pos', a)
            a += 1
    a = 1
    for doc in RN_vector:
        out_str = '-1'
        for i in range(0, len(doc)):
            if doc[i] != 0:
                out_str += ' ' + str(i + 1) + ':' + str(doc[i])
        if out_str != '-1':
            fw.write(out_str)
            fw.write(NEW_LINE)
            print('write neg', a)
            a += 1
    fw.close()
    
    