import os
import codecs
import math


NEW_LINE = '\n'

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
    a = 1
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

    # f = codecs.open('./InputFile/IG.txt', 'w', 'utf-8')
    mylen = len(word_list)
    for i in range(0, len(word_list)):
        max_i = IG_list[:mylen].index(max(IG_list[:mylen]))
        word_list.append( word_list[max_i] )
        word_list.remove(word_list[max_i])
        IG_list.append( IG_list[max_i] )
        IG_list.remove(IG_list[max_i])
        mylen -= 1
        print('process word cnt#', i)
    # for i in range(0, len(word_list)):
    #     f.write( word_list[i] + '\t' + str(IG_list[i]) + NEW_LINE )
    #     print('write word cnt#', i)
    # f.close()


    mOption = input('Input vector_n: ')
    while str(mOption) != 'EXIT':
        # times = len(word_list) // 2 # 去词的比例
        vector_n = int(mOption)
        
        f = codecs.open('./InputFile/feature_set.txt', 'w', 'utf-8')
        for i in range(0, vector_n):
            f.write( word_list[i] + ' ' + str(IG_list[i]) + NEW_LINE )
            print('write', i)
        f.close()
        print('vector_n is', vector_n)
        mOption = input('Input vector_n: ')

        '''
        times = len(word_list) - vector_n
        temp_IG = []
        for ig in IG_list:
            temp_IG.append(ig)
        temp_word = []
        for w in word_list:
            temp_word.append(w)
        a = 0
        for i in range(0, times):
            min_index = temp_IG.index( min(temp_IG) )
            temp_word.remove( temp_word[min_index] )
            temp_IG.remove( temp_IG[min_index] )
            print('remove', a)
            a += 1
        
        f = codecs.open('./InputFile/feature_set.txt', 'w', 'utf-8')
        for i in range(0, len(temp_word)):
            f.write( temp_word[i] + ' ' + str(temp_IG[i]) + NEW_LINE )
            print('write', i)
        f.close()
        print('vector_n is', vector_n)
        mOption = input('Input vector_n: ')
        '''
