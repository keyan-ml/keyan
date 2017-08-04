import os
import gc
import codecs
import jieba
import numpy as np
import math
from sklearn.cluster import KMeans

NEW_LINE = '\n'

words = 0
size = 0

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

def toset(mlist):
    temp = set()
    for elem in mlist:
        temp.add(elem)
    return temp

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

def loadNlpResult(filePath, encoding='utf-8'):
    fr = codecs.open(filePath, 'r', encoding)
    content = fr.read()
    fr.close()
    text_list = content.split(NEW_LINE)
    text_list.remove(text_list[-1])
    for i in range(0, len(text_list)):
        parts = text_list[i].split('|')
        # text_list[i] = parts[-1].strip()
        text_list[i] = parts[-1].strip().split(' ')
    return text_list

def loadFeature(filePath, encoding='utf-8'):
    f = codecs.open(filePath, 'r', encoding)
    content = f.read()
    f.close()
    text_list = content.split(NEW_LINE)
    text_list.remove( text_list[-1] )
    for i in range(0, len(text_list)):
        parts = text_list[i].split(' ')
        text_list[i] = parts[0]
    return text_list


if __name__ == '__main__':
    pos_num = 0
    unlabel_num = 0

    pos_list = loadNlpResult('./InputFile/yuliao_pos.nlpresult')
    unlabel_list = loadNlpResult('./InputFile/yuliao_unlabel.nlpresult')
    pos_num = len(pos_list)
    unlabel_num = len(unlabel_list)
    print('pos', pos_num, ',', 'unlabel', unlabel_num)

    # bow
    feature_list = loadFeature('./InputFile/feature_set.txt')
    for i in range(0, len(pos_list)):
        j = 0
        while j < len(pos_list[i]):
            if pos_list[i][j] in feature_list:
                pos_list[i][j] = feature_list.index( pos_list[i][j] )
                j += 1
            else:
                pos_list[i].remove( pos_list[i][j] )
    p_vector = []
    a = 0
    for doc in pos_list:
        words = list(toset(doc))
        words.sort()
        temp_vector = [0 for ii in range(0, len(feature_list))]
        for word in words:
            temp_vector[word] = doc.count(word)
        if cal_len(temp_vector) != 0:
            temp_vector = norm(temp_vector)
            p_vector.append(temp_vector.tolist()[0])
            print('p_vector', a)
            a += 1

    for i in range(0, len(unlabel_list)):
        j = 0
        while j < len(unlabel_list[i]):
            if unlabel_list[i][j] in feature_list:
                unlabel_list[i][j] = feature_list.index( unlabel_list[i][j] )
                j += 1
            else:
                unlabel_list[i].remove( unlabel_list[i][j] )
    u_vector = []
    a = 0
    for doc in unlabel_list:
        words = list(toset(doc))
        words.sort()
        temp_vector = [0 for ii in range(0, len(feature_list))]
        for word in words:
            temp_vector[word] = doc.count(word)
        if cal_len(temp_vector) != 0:
            temp_vector = norm(temp_vector)
            u_vector.append(temp_vector.tolist()[0])
            print('u_vector', a)
            a += 1
    # bow
    pos_num = len(p_vector)
    unlabel_num = len(u_vector)

    # f = codecs.open('./InputFile/test.txt', 'w', 'utf-8')
    # a = 0
    # for vec in p_vector:
    #     for temp in vec:
    #         if temp != 0:
    #             f.write(str(temp) + ' ')
    #             print('pos', a)
    #             a += 1
    #     f.write(NEW_LINE)
    # a = 0
    # for vec in u_vector:
    #     for temp in vec:
    #         if temp != 0:
    #             f.write(str(temp) + ' ')
    #             print('unlabel', a)
    #             a += 1
    #     f.write(NEW_LINE)
    # f.close()

    
    # f = codecs.open('./InputFile/p_vector.txt', 'w', 'utf-8')
    # f.write(str( len(p_vector[0]) ) + NEW_LINE)
    # a = 1
    # for i in range(0, len(p_vector)):
    #     out_str = str(p_id[i])
    #     for j in range(0, len(p_vector[i])):
    #         if p_vector[i][j] != 0:
    #             out_str += ' ' + str(j) + ':' + str(p_vector[i][j])
    #     if out_str != str(p_id[i]):
    #         f.write(out_str + NEW_LINE)
    #         print('write p_vector', a, '/', len(p_vector))
    #         a += 1
    # f.close()

    # 计算 中心向量
    print('cal center...')
    p_center, u_center = getCenter(p_vector, u_vector)
    print('...finished')
    # print('p_center:\n', p_center.tolist())
    # print('u_center:\n', u_center.tolist())
    # 计算 中心向量

    
    # 找 可靠负例
    print('find RN...')
    i = 0
    while i < len(u_vector):
        print('process u_vector', i)
        d = u_vector[i]
        if cos_sim(d, p_center) > cos_sim(d, u_center):
            u_vector.remove(d)
        else:
            i += 1
    RN_vector = u_vector

    print('[first]total ', len(RN_vector), '/', unlabel_num)
    # 找 可靠负例

    # f = codecs.open('./InputFile/RN_vector.txt', 'w', 'utf-8')
    # f.write(str( len(RN_vector[0]) ) + NEW_LINE)
    # a = 1
    # for i in range(0, len(RN_vector)):
    #     out_str = str(RN_id[i])
    #     for j in range(0, len(RN_vector[i])):
    #         if RN_vector[i][j] != 0:
    #             out_str += ' ' + str(j) + ':' + str(RN_vector[i][j])
    #     if out_str != str(RN_id[i]):
    #         f.write(out_str + NEW_LINE)
    #         print('write RN_vector', a, '/', len(RN_vector))
    #         a += 1
    # f.close()

    # cluster
    print('cluster...')
    n_clusters = 8
    kmeans = KMeans(n_clusters=n_clusters, random_state=0).fit(RN_vector)
    labels = kmeans.labels_.tolist()
    print('...finished')
    # print('kmeans labels:\n', labels)
    # cluster
    
    
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
    pre_cnt = len(RN_vector)
    # 计算余弦距离，重新确定可靠负例
    a = 0
    i = 0
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
        else:
            RN_vector.remove(RN_vector[i])
    # 计算余弦距离，重新确定可靠负例
    print('...finished')

    print('Here is them: ', len(RN_vector), '/', pre_cnt)
    
    # 保存为训练集
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
    # 保存为训练集
    