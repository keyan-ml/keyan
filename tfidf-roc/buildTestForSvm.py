import os
import codecs
import numpy as np
import math
from sklearn import feature_extraction
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer


NEW_LINE = '\r\n'

def toset(mlist):
    temp = set()
    for elem in mlist:
        temp.add(elem)
    return temp

def loadNlpResult(filePath, encoding='utf-8'):
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

def loadTfidfWord(filePath, encoding='utf-8'):
    f = codecs.open(filePath, 'r', encoding)
    content = f.read()
    f.close()
    words = content.split(NEW_LINE)
    words.remove(words[-1])
    return words


if __name__ == '__main__':
    test_class, test_list = loadNlpResult('./InputFile/yuliao_test.nlpresult')
    tfidf_word = loadTfidfWord('./InputFile/tfidf_word.txt')
    '''
    # tfidf
    print('cal tfidf ...')
    corpus = test_list
    vectorizer = CountVectorizer()
    transformer = TfidfTransformer()
    tfidf = transformer.fit_transform( vectorizer.fit_transform(corpus) )
    word_set = vectorizer.get_feature_names()
    weight = tfidf.toarray()
    print('cal tfidf finished')
    # tfidf
    test_vector = weight.tolist()
    '''
    for i in range(0, len(test_list)):
        j = 0
        while j < len(test_list[i]):
            if test_list[i][j] in tfidf_word:
                test_list[i][j] = tfidf_word.index(test_list[i][j])
                j += 1
            else:
                test_list[i].remove(test_list[i][j])
        # test_list[i].sort()
        print('process test', i)


    fw = codecs.open('../svm/test.txt', 'w', 'utf-8')
    a = 1
    for i in range(0, len(test_list)):
        out_str = test_class[i]
        temp_set = toset(test_list[i])
        word_list = list(temp_set)
        word_list.sort()
        for word in word_list:
            # if word in tfidf_word:
            out_str += ' ' + str(word) + ':' + str(test_list[i].count(word))
        if out_str != test_class[i]:
            fw.write(out_str + NEW_LINE)
            print('write test', a)
            a += 1
    fw.close()
