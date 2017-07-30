import os
import gc
import codecs
import jieba
import numpy as np
import math
from sklearn import feature_extraction
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer

NEW_LINE = '\r\n'

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
    global pos_num, unlabel_num
    Alpha = 16
    Beta = 4

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
        text_list[i] = parts[-1].strip()
        # text_list[i] = parts[-1].strip().split(' ')
    return text_list


if __name__ == '__main__':
    pos_num = 0
    unlabel_num = 0

    # word2vec = loadVector('./InputFile/vectors.txt')
    # word2id, id2word = loadWordLib('../data/Word_Library.wordlib')
    # word_num = len(word2id)
    # print('total word', word_num)

    pos_list = loadNlpResult('./InputFile/yuliao_pos.nlpresult')
    unlabel_list = loadNlpResult('./InputFile/yuliao_unlabel.nlpresult')
    pos_num = len(pos_list)
    unlabel_num = len(unlabel_list)
    print(pos_num, unlabel_num)
    
    test_list = loadNlpResult('./InputFile/yuliao_test.nlpresult')


    text_list = pos_list + unlabel_list + test_list

    # tfidf
    print('cal tfidf ...')
    corpus = text_list
    vectorizer = CountVectorizer()
    transformer = TfidfTransformer()
    tfidf = transformer.fit_transform( vectorizer.fit_transform(corpus) )
    word_set = vectorizer.get_feature_names()
    weight = tfidf.toarray()
    print('cal tfidf finished')
    # tfidf
    f = codecs.open('./InputFile/tfidf_word.txt', 'w', 'utf-8')
    for word in word_set:
        f.write(word + NEW_LINE)
    f.close()

    # 计算 中心向量
    p_id = list( range(1, pos_num + 1) )
    u_id = list( range(1, unlabel_num + 1) )
    p_vector = weight[0:pos_num]
    u_vector = weight[pos_num:]

    f = codecs.open('./InputFile/p_vector.txt', 'w', 'utf-8')
    f.write(str( len(p_vector[0]) ) + NEW_LINE)
    a = 1
    for i in range(0, len(p_vector)):
        out_str = str(p_id[i])
        for j in range(0, len(p_vector[i])):
            if p_vector[i][j] != 0:
                out_str += ' ' + str(j) + ':' + str(p_vector[i][j])
        if out_str != str(p_id[i]):
            f.write(out_str + NEW_LINE)
            print('write p_vector', a, '/', len(p_vector))
            a += 1
    f.close()

    '''
    i = 0
    while i < len(p_vector):
        if cal_len(p_vector[i]) > 0:
            i += 1
        else:
            p_id.remove(p_id[i])
            p_vector.remove(p_vector[i])
            continue

    i = 0
    while i < len(u_vector):
        if cal_len(u_vector[i]) > 0:
            i += 1
        else:
            u_id.remove(u_id[i])
            u_vector.remove(u_vector[i])
            continue
    '''
    # for i in range(0, len(p_vector)):
    #     p_vector[i] = (np.array(p_vector[i]).tolist())[0]
    # for i in range(0, len(u_vector)):
    #     u_vector[i] = (np.array(u_vector[i]).tolist())[0]
    # print(p_vector)
    p_center, u_center = getCenter(p_vector, u_vector)

    # print('p_center:\n', p_center.tolist())
    # print('u_center:\n', u_center.tolist())
    # 计算 中心向量

    
    # 找 可靠负例
    RN_id = []
    RN_vector = []
    
    for i in range(0, len(u_vector)):
        d = u_vector[i]
        if cos_sim(d, p_center) <= cos_sim(d, u_center):
            RN_id.append(u_id[i])
            RN_vector.append(d)

    # print(RN_id)
    # print(RN_vector)

    # print('first:', RN_id)
    print('[first]total ', len(RN_id), '/', unlabel_num)
    # 找 可靠负例

    f = codecs.open('./InputFile/RN_vector.txt', 'w', 'utf-8')
    f.write(str( len(RN_vector[0]) ) + NEW_LINE)
    a = 1
    for i in range(0, len(RN_vector)):
        out_str = str(RN_id[i])
        for j in range(0, len(RN_vector[i])):
            if RN_vector[i][j] != 0:
                out_str += ' ' + str(j) + ':' + str(RN_vector[i][j])
        if out_str != str(RN_id[i]):
            f.write(out_str + NEW_LINE)
            print('write RN_vector', a, '/', len(RN_vector))
            a += 1
    f.close()




'''
def loadWordLib(filePath, encoding='utf-8'):
    f = codecs.open(filePath, 'r', encoding)
    content = f.read()
    f.close()
    word_list = content.split(NEW_LINE)
    word_list.remove(word_list[-1])
    word2id = {}
    id2word = {}
    word_set = []
    for a in word_list:
        parts = a.split(' ')
        word2id[parts[0].strip()] = int(parts[1])
        id2word[int(parts[1])] = parts[0].strip()
        # word_set.append(parts[0].strip())
    return word2id, id2word
'''
'''
def toVector(text_list, word2vec):
    zero_vec = np.mat([0.0 for ii in range(0, size)])
    for i in range(0, len(text_list)):
        temp = word2vec.get(text_list[i][0], zero_vec)
        for j in range(1, len(text_list[i])):
            if text_list[i][j] in word2vec.keys():
                temp += word2vec[ text_list[i][j] ]
        text_list[i] = norm(temp)
    return text_list
'''

'''
def loadVector(filePath):
    global words, size
    fr = codecs.open(filePath, 'r', 'utf-8')
    content = fr.read()
    fr.close()
    text_list = content.split(NEW_LINE)
    temp = text_list[0].split(' ')
    words = int(temp[0])
    size = int(temp[1])
    text_list.remove(text_list[0])
    text_list.remove(text_list[-1])
    word2vec = {}
    i = 0
    for text in text_list:
        parts = text.split(' ')
        word = parts[0]
        while parts[-1] == '':
            parts.remove(parts[-1])
            parts.remove(parts[0])
        for j in range(0, len(parts)):
            parts[j] = float(parts[j])
        word2vec[word] = norm( np.mat(parts) )
        print('load vector', i)
        i += 1
    return word2vec
'''
