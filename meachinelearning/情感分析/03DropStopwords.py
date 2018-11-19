# coding:utf-8
# @Time :2018/11/19 1:13
# @Author: Mensyne
# @File :03DropStopwords.py

# 去掉停用词
import  codecs,sys

def stopWord(sourceFile,targetFile,stopkey):
    sourf = codecs.open(sourceFile,'r',encoding='utf-8')
    targetf = codecs.open(targetFile, 'w', encoding='utf-8')
    print("open source Files"+sourf)
    print("open target Files"+targetf)
    lineNum = 1
    line = sourf.realine()
    while line:
        print("----processing---",lineNum,'artilce-----')
        sentence = DropStopWord(line,stopkey)
        targetf.writelines(sentence+'\n')
        lineNum +=1
        line = sourf.readline()
    print("well done")
    sourf.close()
    targetf.close()

# 删除停用词
def DropStopWord(lien,stopkey):
    wordList = lien.split(' ')
    sentence = ''
    for word in  wordList:
        word = word.strip()
        if word not in stopkey:
            if word != '\t':
                sentence += word +" "
    return sentence.strip()

if __name__ == '__main__':
    stopkey = [w.strip() for w in codecs.open("data\stopWord.txt",'r',encoding='utf8').readlines()]
    sourceFile = '2000_neg_cut.txt'
    targetFile = '2000_neg_cut_stopword.txt'
    stopWord(sourceFile,targetFile,stopkey)

    sourceFile = '2000_pos_cut.txt'
    targetFile = '2000_pos_cut_stopword.txt'
    stopWord(sourceFile,targetFile,stopkey)





