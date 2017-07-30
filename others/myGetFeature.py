import os
import codecs
import jieba
import numpy as np
import math

NEW_LINE = '\r\n'


# 小函数
def cal_len(vec):
    vec = np.mat(vec)
    num = (float)(vec * vec.T)
    return math.sqrt(num)

def norm(vec):
    vec = np.array(vec)
    return vec / cal_len(vec)

def toset(mlist):
    temp = set()
    for elem in mlist:
        temp.add(elem)
    return temp
# 小函数


def loadDocs(filePath, encoding='utf-8'):
    f = codecs.open(filePath, 'r', encoding=encoding)
    content = f.read()
    f.close()
    text_list = str(content).split(NEW_LINE)
    text_list.remove( text_list[-1] )
    # for i in range(0, len(text_list)):
    #     text_list[i] = text_list[i].strip()
    return text_list


if __name__ == '__main__':
    pos_num = 0
    unlabel_num = 0
    word_set = set()


    # io 预备
    text_list = []
    # 读取 pos
    temp = loadDocs('./Lpu_Input/yuliao_pos.csv')
    pos_num = len(temp)
    text_list.extend(temp)
    # 读取 pos


    # 读取 unlabel
    temp = loadDocs('./Lpu_Input/yuliao_unlabel.csv')
    unlabel_num = len(temp)
    text_list.extend(temp)
    # 读取 unlabel

    print('pos:', pos_num, 'unlabel:', unlabel_num)
    # io 预备


    # 分词
    jieba.add_word('不喜欢')
    jieba.add_word('耗油')
    jieba.add_word('不合格')

    for i in range(0, len(text_list)):
        temp = text_list[i].split('|')
        text_list[i] = jieba.lcut( temp[-1], cut_all=False )
        # text_list[i] = list(seg_list)
        print('segment', i)
    # 分词


    # 去停用词
    # 读取停用词
    f = codecs.open('./data/stop_word_UTF_8.txt', 'r', encoding='utf-8')
    content = f.read()
    f.close()
    stop_word_list = str(content).split(NEW_LINE)
    stop_word_list.remove( stop_word_list[-1] )
    # 读取停用词

    fw = codecs.open('./test/jieba_segment.txt', 'w', encoding='utf-8')

    for i in range(0, len(text_list)):
        j = 0
        while j < len(text_list[i]):
            if text_list[i][j] in stop_word_list:
                text_list[i].remove( text_list[i][j] )
            else:
                word_set.add( text_list[i][j] )
                j += 1
        fw.write( ' '.join(text_list[i]) + NEW_LINE )
        print('quTYC', i)

    fw.close()

    # for i in range(0, len(text_list)):
    #     print('#' + str(i) + '\t' + ' '.join(text_list[i]))
    # 去停用词


    # 词库排序, 整理
    print('cal word_set start')
    word_set = list(word_set)
    word_set.sort()

    f = codecs.open('./temp/word_set.txt', 'w', encoding='utf-8')
    word_num = len(word_set)
    id = 0
    for word in word_set:
        f.write( word + ' ' + str(id) + NEW_LINE )
        id += 1
    f.close()
    print('cal word_set finished')

    # print('word set:')
    # print(word_set)
    # 词库排序, 整理


    # 计算 tfidf，并选取特征词
    print('cal tfidf start')
    f_tfidf_temp = codecs.open('./temp/tfidf.txt', 'w', 'utf-8')
    myword_set = []
    # f_tfidf = codecs.open("./test/tfidf.txt", 'w', encoding = "utf-8")
    D_num = len(text_list)
    for doc_i in range(0, len(text_list)):
        this_doc = text_list[doc_i]
        # 计算一个文档（一句）的tfidf
        doc_weight = [0 for i in range(0, word_num)]
        for word in this_doc:
            doc_weight[ word_set.index(word) ] += 1

        cnt = len(this_doc)
        temp_set = toset(this_doc)
        if cnt != 0:
            for word in temp_set:
                doc_weight[ word_set.index(word) ] /= cnt

                cnt_doc = 0
                for doc_temp in text_list:
                    if word in doc_temp:
                        cnt_doc += 1
                idf = math.log( D_num / cnt_doc ) + 1
                doc_weight[ word_set.index(word) ] *= idf

            doc_weight = np.array(norm(doc_weight))
            doc_weight = doc_weight.tolist()
            
        # #记录每个文档（句子）的tf-idf词语权重（太大了）
        # out_str = ''
        # for j in range(0, len(doc_weight)):
        #     out_str += word_set[j] + ':' + str(doc_weight[j]) + ' '
        # f_tfidf.write(out_str.strip() + '\n')
        # #记录每个文档（句子）的tf-idf词语权重
        # 计算一个文档（一句）的tfidf

        temp_word = list(temp_set)
        temp_tfidf = []
        for word in temp_word:
            temp_tfidf.append(doc_weight[ word_set.index(word) ])
            f_tfidf_temp.write(word + ':' + str( doc_weight[ word_set.index(word) ] ) + ' ')
        f_tfidf_temp.write(NEW_LINE)
        
        # 去掉一半，tfidf比较小的
        times = len(temp_word) // 2 # 去词的比例
        for i in range(0, times):
            min_index = temp_tfidf.index( min(temp_tfidf) )
            temp_word.remove( temp_word[min_index] )
            temp_tfidf.remove( temp_tfidf[min_index] )
        # 去掉一半，tfidf比较小的

        # 加入词集（未去重）
        myword_set.extend(temp_word)
        # 加入词集（未去重）

        print('process doc', doc_i)
        # weight.append(doc_weight)
        # for elem in doc_weight:
        #     f.write(str(elem) + '\t')
        # f.write('\n')
        # print('cal weight', i)
    # f.close()
    # print('cal tfidf finished')
    # f_tfidf.close()
    f_tfidf_temp.close()

    feature_set = toset(myword_set)
    # myword_set = feature_set.tolist()

    fw = codecs.open('./data/feature_set.txt', 'w', encoding='utf-8')
    for word in feature_set:
        fw.write(word + NEW_LINE)
    fw.close()
    