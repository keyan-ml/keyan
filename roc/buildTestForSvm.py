import os
import codecs
import numpy as np
import math

NEW_LINE = '\n'

def cal_len(vec):
    vec = np.mat(vec)
    num = (float)(vec * vec.T)
    return math.sqrt(num)

def norm(vec):
    vec = np.mat(vec)
    return vec / cal_len(vec)

def toset(mlist):
    temp = set()
    for elem in mlist:
        temp.add(elem)
    return temp

def loadTestNlpResult(filePath, encoding='utf-8'):
    fr = codecs.open(filePath, 'r', encoding)
    content = fr.read()
    fr.close()
    text_list = content.split(NEW_LINE)
    text_list.remove(text_list[-1])
    text_class = []
    for i in range(0, len(text_list)):
        parts = text_list[i].split('|')
        text_class.append(parts[0])
        # text_list[i] = parts[-1].strip()
        text_list[i] = parts[-1].strip().split(' ')
    return text_class, text_list

# def loadTfidfWord(filePath, encoding='utf-8'):
#     f = codecs.open(filePath, 'r', encoding)
#     content = f.read()
#     f.close()
#     words = content.split(NEW_LINE)
#     words.remove(words[-1])
#     return words

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
    test_class, test_list = loadTestNlpResult('./InputFile/yuliao_test.nlpresult')
    
    # bow
    feature_list = loadFeature('./InputFile/feature_set.txt')
    for i in range(0, len(test_list)):
        j = 0
        while j < len(test_list[i]):
            if test_list[i][j] in feature_list:
                test_list[i][j] = feature_list.index( test_list[i][j] )
                j += 1
            else:
                test_list[i].remove( test_list[i][j] )
    test_vector = []
    i = 0
    while i < len(test_list):
        doc = test_list[i]
        words = toset(doc)
        temp_vector = [0 for ii in range(0, len(feature_list))]
        for word in words:
            temp_vector[word] = doc.count(word)
        if cal_len(temp_vector) == 0:
            test_list.remove( test_list[i] )
            test_class.remove( test_class[i] )
        else:
            temp_vector = norm(temp_vector)
            test_vector.append(temp_vector.tolist()[0])
            print('test_vector', i)
            i += 1
    # bow
    # print(len(test_class), len(test_list))

    # 保存为测试集
    fw = codecs.open('../svm/test.txt', 'w', 'utf-8')
    a = 1
    for i in range(0, len(test_vector)):
        doc = test_vector[i]
        out_str = test_class[i]
        for j in range(0, len(doc)):
            if doc[j] != 0:
                out_str += ' ' + str(j + 1) + ':' + str(doc[j])
        if out_str != test_class[i]:
            fw.write(out_str)
            fw.write(NEW_LINE)
            print('write test', a)
            a += 1
    fw.close()
    # 保存为测试集
    