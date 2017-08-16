import os
import sys
import codecs
import jieba
import jieba.posseg as pseg
import platform

NEW_LINE = '\n'
if platform.system() == 'Windows':
    NEW_LINE = '\r\n'
jieba.add_word('不喜欢')
jieba.add_word('没有')
jieba.add_word('奔驰', tag='nz')


def loadDocs(filePath, encoding='utf-8'):
    f = codecs.open(filePath, 'r', encoding=encoding)
    content = f.read()
    f.close()
    text_list = content.split(NEW_LINE)
    text_list.pop(-1)
    # for i in range(0, len(text_list)):
    #     text_list[i] = text_list[i].strip()
    return text_list

def mySegment(text_list):
    flag_list = []
    for i in range(0, len(text_list)):
        parts = text_list[i].split('|')
        # sResult = jieba.lcut(parts[-1])
        words = pseg.cut(parts[-1])
        sResult = []
        flags = []
        for w in words:
            sResult.append( w.word )
            flags.append( w.flag )
        text_list[i] = parts[:-1]
        text_list[i].append( sResult )
        flag_list.append( flags )
        print('mySegment', i)
    return text_list, flag_list

def loadTYC():
    f = codecs.open('../data/stop_word_UTF_8.txt', 'r', encoding='utf-8')
    content = f.read()
    f.close()
    stop_word_list = content.split(NEW_LINE)
    stop_word_list.pop(-1)
    return stop_word_list

def quTYC(text_list, flag_list, stop_word_list):
    for i in range(0, len(text_list)):
        j = 0
        while j < len(text_list[i][-1]):
            # word = text_list[i][-1][j]
            '''
            flag = False
            for sw in stop_word_list:
                # if word in sw or sw in word:
                if sw in word:
                    flag = True
                    text_list[i][-1].remove( word )
                    break
            if flag == False:
                j += 1
            '''
            if text_list[i][-1][j] in stop_word_list:
                text_list[i][-1].pop( j )
                flag_list[i].pop(j)
            else:
                j += 1
        print('quTYC', i)
    return text_list, flag_list

'''
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
'''


def segPos():
    pos_num = 0
    stop_word_list = loadTYC() # 停用词集
    # my_feature_set = loadFeature() # 特征集
    
    # 处理 pos
    text_list = loadDocs("./InputFile/yuliao_pos.csv")
    pos_num = len(text_list)

    text_list, flag_list = mySegment(text_list) # 分词
    text_list, flag_list = quTYC(text_list, flag_list, stop_word_list) # 去停用词
    # text_list = feature_process(text_list, my_feature_set) # 有关特征词处理

    fw = codecs.open('./InputFile/yuliao_pos.nlpresult', 'w', encoding='utf-8')
    for i in range(0, len(text_list)):
        doc = text_list[i]
        flags = flag_list[i]
        if doc[-1] != []:
            for j in range(0, len(doc) - 1):
                fw.write(doc[j] + '|')
            temp_str = ''
            for j in range(0, len(doc[-1])):
                temp_str += doc[-1][j] + '/' + flags[j] + ' '
            fw.write( temp_str.strip() + NEW_LINE )
    fw.close()

    print('pos Done')
    # 处理 pos


def segUnlabel():
    unlabel_num = 0
    stop_word_list = loadTYC() # 停用词集

    # 处理 unlabel
    text_list = loadDocs("./InputFile/yuliao_unlabel.csv")
    unlabel_num = len(text_list)

    text_list, flag_list = mySegment(text_list) # 分词
    text_list, flag_list = quTYC(text_list, flag_list, stop_word_list) # 去停用词
    # text_list = feature_process(text_list, my_feature_set) # 有关特征词处理

    fw = codecs.open('./InputFile/yuliao_unlabel.nlpresult', 'w', encoding='utf-8')
    for i in range(0, len(text_list)):
        doc = text_list[i]
        flags = flag_list[i]
        if doc[-1] != []:
            for j in range(0, len(doc) - 1):
                fw.write(doc[j] + '|')
            temp_str = ''
            for j in range(0, len(doc[-1])):
                temp_str += doc[-1][j] + '/' + flags[j] + ' '
            fw.write( temp_str.strip() + NEW_LINE )
    fw.close()

    print('unlabel Done')
    # 处理 unlabel
    
def segTest():
    test_num = 0
    stop_word_list = loadTYC() # 停用词集
    
    # 处理 test
    text_list = loadDocs("./InputFile/yuliao_test.csv")
    test_num = len(text_list)

    text_list, flag_list = mySegment(text_list) # 分词
    text_list, flag_list = quTYC(text_list, flag_list, stop_word_list) # 去停用词
    # text_list = feature_process(text_list, my_feature_set) # 有关特征词处理

    fw = open('./InputFile/yuliao_test.nlpresult', 'w', encoding='utf-8')
    for i in range(0, len(text_list)):
        doc = text_list[i]
        flags = flag_list[i]
        if doc[-1] != []:
            for j in range(0, len(doc) - 1):
                fw.write(doc[j] + '|')
            temp_str = ''
            for j in range(0, len(doc[-1])):
                temp_str += doc[-1][j] + '/' + flags[j] + ' '
            fw.write( temp_str.strip() + NEW_LINE )
    fw.close()

    print('test Done')
    # 处理 test
    
if __name__ == '__main__':
    segPos()
    segUnlabel()
    segTest()