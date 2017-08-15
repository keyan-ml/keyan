import os
import codecs

NEW_LINE = '\n'

if __name__ == '__main__':
    pos_num = 0


    f = codecs.open('./test.txt', 'r', 'utf-8')
    content = f.read()
    f.close()
    test_list = content.split(NEW_LINE)
    test_list.remove( test_list[-1] )
    for i in range(0, len(test_list)):
        parts = test_list[i].split(' ')
        test_list[i] = parts[0]
    pos_num = test_list.count('1')
    
    f = codecs.open('./result', 'r', 'utf-8')
    content = f.read()
    f.close()
    result_list = content.split(NEW_LINE)
    result_list.remove( result_list[-1] )
    pos_result = result_list[0:pos_num]
    neg_result = result_list[pos_num:]

    TP = 0
    FN = 0
    for result in pos_result:
        if float(result) > 0:
            TP += 1
        else:
            FN += 1
    FP = 0
    TN = 0
    for result in neg_result:
        if float(result) <= 0:
            TN += 1
        else:
            FP += 1
    
    Acc = ( TP + TN ) / len(result_list)
    pos_P = TP / ( TP + FP )
    pos_R = TP / ( TP + FN )
    neg_P = TN / ( TN + FN )
    neg_R = TN / ( TN + FP )

    print( "Accuracy: %.2f%c (%d correct, %d incorrect, %d total)" % (Acc * 100, '%', TP + TN, FP + FN, len(result_list)) )
    print( "POS Precision/recall: %.2f%c/%.2f%c TP=%d FP=%d TotalPos=%d" % (pos_P * 100, '%', pos_R * 100, '%', TP, FP, TP + FN) )
    print( "NEG Precision/recall: %.2f%c/%.2f%c" % (neg_P * 100, '%', neg_R * 100, '%') )
