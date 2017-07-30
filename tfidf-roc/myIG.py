import os
import codecs
import math


NEW_LINE = '\r\n'

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


if __name__ == '__main__':
    pos_list = loadNlpResult('./InputFile/yuliao_pos.nlpresult')
    pos_num = len(pos_list)
    neg_list = loadNlpResult('./InputFile/yuliao_unlabel.nlpresult')
    neg_num = len(neg_list)
    all_num = pos_num + neg_num

    P_pos = pos_num / all_num
    P_neg = neg_num / all_num

    word_set = set()
    for doc in pos_list:
        for word in doc:
            word_set.add(word)
    for doc in neg_list:
        for word in doc:
            word_set.add(word)

    word_list = list(word_set)
    word_list.sort()
    IG_list = []
    a = 0
    for word in word_list:
        IG = 0
        # pos
        wordInPos_num = 0
        for doc in pos_list:
            if word in doc:
                wordInPos_num += 1
        wordInNeg_num = 0
        for doc in neg_list:
            if word in doc:
                wordInNeg_num += 1
        P_pos_w = wordInPos_num / all_num
        P_w = (wordInPos_num + wordInNeg_num) / all_num
        if P_pos_w != 0:
            IG += P_pos_w * math.log(P_pos_w / (P_pos * P_w))

        wordNotInPos_num = 0
        for doc in pos_list:
            if word not in doc:
                wordNotInPos_num += 1
        wordNotInNeg_num = 0
        for doc in neg_list:
            if word not in doc:
                wordNotInNeg_num += 1
        P_pos_w_not = wordNotInPos_num / all_num
        P_w_not = (wordNotInPos_num + wordNotInNeg_num) / all_num
        if P_pos_w_not != 0:
            IG += P_pos_w_not * math.log(P_pos_w_not / (P_pos * P_w_not))
        # pos
        # neg
        P_neg_w = wordInNeg_num / all_num
        if P_neg_w != 0:
            IG += P_neg_w * math.log(P_neg_w / (P_neg *  P_w))

        P_neg_w_not = wordNotInNeg_num / all_num
        if P_neg_w_not != 0:
            IG += P_neg_w * math.log(P_neg_w_not / (P_neg * P_w_not))
        # neg
        IG_list.append(IG)
        print('cal IG', a, '/', len(word_list))
        a += 1

    
    # times = len(word_list) // 2 # 去词的比例
    vector_n = 3000
    times = len(word_list) - vector_n
    a = 0
    for i in range(0, times):
        min_index = IG_list.index( min(IG_list) )
        word_list.remove( word_list[min_index] )
        IG_list.remove( IG_list[min_index] )
        print('remove', a)
        a += 1
    
    
    f = codecs.open('./InputFile/feature_set.txt', 'w', 'utf-8')
    for i in range(0, len(word_list)):
        f.write( word_list[i] + ' ' + str(IG_list[i]) + NEW_LINE )
        print('write', i)
    f.close()
