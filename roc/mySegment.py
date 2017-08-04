import os
import sys
import codecs
import jieba

NEW_LINE = '\n'


def loadDocs(filePath, encoding='utf-8'):
    f = codecs.open(filePath, 'r', encoding=encoding)
    content = f.read()
    f.close()
    text_list = content.split(NEW_LINE)
    text_list.remove( text_list[-1] )
    # for i in range(0, len(text_list)):
    #     text_list[i] = text_list[i].strip()
    return text_list

def mySegment(text_list):
    for i in range(0, len(text_list)):
        parts = text_list[i].split('|')
        sResult = jieba.lcut(parts[-1], cut_all=False)
        text_list[i] = parts[:-1]
        text_list[i].append( sResult )
        print('mySegment', i)
    return text_list

def loadTYC():
    f = codecs.open('../data/stop_word_UTF_8.txt', 'r', encoding='utf-8')
    content = f.read()
    f.close()
    stop_word_list = content.split(NEW_LINE)
    stop_word_list.remove( stop_word_list[-1] )
    return stop_word_list

def quTYC(text_list, stop_word_list):
    for i in range(0, len(text_list)):
        j = 0
        while j < len(text_list[i][-1]):
            word = text_list[i][-1][j]
            flag = False
            for sw in stop_word_list:
                if word in sw or sw in word:
                    flag = True
                    text_list[i][-1].remove( word )
                    break
            if flag == False:
                j += 1
        print('quTYC', i)
    return text_list


# 匹配 特征集
def loadFeature():
    f = codecs.open('./InputFile/feature_set.txt', 'r', encoding='utf-8')
    content = f.read()
    f.close()
    my_feature_set = content.split(NEW_LINE)
    my_feature_set.remove( my_feature_set[-1] )
    for i in range(0, len(my_feature_set)):
        my_feature_set[i] = my_feature_set[i].split(' ')[0]
    return my_feature_set
def feature_process(text_list, my_feature_set):
    for i in range(0, len(text_list)):
        j = 0
        while j < len(text_list[i][-1]):
            word = text_list[i][-1][j]
            if word in my_feature_set:
                j += 1
            else:
                text_list[i][-1].remove(word)
        print('feature_process', i)
    return text_list
# 匹配 特征集



def segPos():
    pos_num = 0
    stop_word_list = loadTYC() # 停用词集
    # my_feature_set = loadFeature() # 特征集
    
    # 处理 pos
    text_list = loadDocs("./InputFile/yuliao_pos.csv")
    pos_num = len(text_list)

    text_list = mySegment(text_list) # 分词
    text_list = quTYC(text_list, stop_word_list) # 去停用词
    # text_list = feature_process(text_list, my_feature_set) # 有关特征词处理

    fw = codecs.open('./InputFile/yuliao_pos.nlpresult', 'w', encoding='utf-8')
    for doc in text_list:
        if doc[-1] != []:
            for i in range(0, len(doc) - 1):
                fw.write(doc[i] + '|')
            fw.write( ' '.join(doc[-1]) + NEW_LINE )
    fw.close()

    print('pos Done')
    # 处理 pos


def segUnlabel():
    unlabel_num = 0
    stop_word_list = loadTYC() # 停用词集

    # 处理 unlabel
    text_list = loadDocs("./InputFile/yuliao_unlabel.csv")
    unlabel_num = len(text_list)

    text_list = mySegment(text_list) # 分词
    text_list = quTYC(text_list, stop_word_list) # 去停用词
    # text_list = feature_process(text_list, my_feature_set) # 有关特征词处理

    fw = codecs.open('./InputFile/yuliao_unlabel.nlpresult', 'w', encoding='utf-8')
    for doc in text_list:
        if doc[-1] != []:
            for i in range(0, len(doc) - 1):
                fw.write(doc[i] + '|')
            fw.write( ' '.join(doc[-1]) + NEW_LINE )
    fw.close()

    print('unlabel Done')
    # 处理 unlabel
    
def segTest():
    test_num = 0
    stop_word_list = loadTYC() # 停用词集
    
    # 处理 test
    text_list = loadDocs("./InputFile/yuliao_test.csv")
    test_num = len(text_list)

    text_list = mySegment(text_list) # 分词
    text_list = quTYC(text_list, stop_word_list) # 去停用词
    # text_list = feature_process(text_list, my_feature_set) # 有关特征词处理

    fw = open('./InputFile/yuliao_test.nlpresult', 'w', encoding='utf-8')
    for doc in text_list:
        if doc[-1] != []:
            for i in range(0, len(doc) - 1):
                fw.write(doc[i] + '|')
            fw.write( ' '.join(doc[-1]) + "\n" )
    fw.close()

    print('test Done')
    # 处理 test
    
if __name__ == '__main__':
    segPos()
    segUnlabel()
    segTest()