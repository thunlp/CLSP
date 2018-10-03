# coding:utf8
'''
Input：Chinese & English word embedding files
Datasets: En-Zh Dictionary, Wordsim-240, Wordsim-297; Wordsim-353, SimLex-999
Output: Monolingual wordsim results of Chinese and English; Precision@1 and Precision@5 of Bilingual Lexicon Induction
'''
import sys
import numpy as np
from numpy import linalg
import time
import random


outputPath = sys.argv[1]
maxTestNum = eval(sys.argv[2])

zhWordVecFile = outputPath + "word-vec.zh"
enWordVecFile = outputPath + "word-vec.en"


def ReadWordVec(wordVecFile):
    '''
    Read the vectors of words from the word embedding file
    '''
    start = time.clock()
    wordVecDict = {}
    with open(wordVecFile, 'r') as file:
        num = 0
        for line in file:
            num += 1
            if num % 1000 == 0:
                print("Reading the %d-th word" % num)

            items = line.strip().split()
            if len(items) == 201:
                word = items[0]
                vec = map(eval, items[1:])
                if linalg.norm(vec) > 0:
                    wordVecDict[word] = vec / linalg.norm(vec)
    print('Word Embeddings Reading Complete, Total Number of words is: %d' % num)
    print('Time Used: %f' % (time.clock() - start))
    return wordVecDict


def EvalWordSim(wordVecDict, wordSimFile):
    '''
    Evaluate the word similarity of word embeddings according to certain human-annotated word similarity file
    '''
    with open('../data/eval_data/' + wordSimFile + '.txt') as file:
        wordSimStd = []
        wordSimPre = []
        skipPair = 0
        testPair = 0
        for line in file:
            word1, word2, valStr = line.strip().split()
            if wordVecDict.has_key(word1) and wordVecDict.has_key(word2):
                wordVec1 = wordVecDict[word1]
                wordVec2 = wordVecDict[word2]
                cosSim = np.dot(wordVec1, wordVec2)
                wordSimStd.append(eval(valStr))
                wordSimPre.append(cosSim)
                testPair += 1
            else:
                skipPair += 1
        corrCoef = np.corrcoef(wordSimStd, wordSimPre)[0, 1]
        print(wordSimFile + " Score: %f" % corrCoef)
        print('Number of Tested Pairs:%d, Number of Skipped Pairs: %d' %
              (testPair, skipPair))
    return corrCoef, testPair, skipPair


def ReadEn2zhDict(dictFile):
    '''
    Read the English to Chinese Dictionary
    '''
    en2zhDict = {}
    with open('../data/eval_data/' + dictFile, 'r') as file:
        for line in file:
            line = line.strip()
            loc = line.find('\t')
            enWord = line[:loc]
            zhWords = line[loc + 1:].split('/')
            en2zhDict[enWord.lower()] = zhWords
    print('Dictionary Reading Complete and the number of English words is: %d' %
          len(en2zhDict))
    return en2zhDict


def EvalLexiconInduction(zhWordVecDict, enWordVecDict, dictFile, testNum):
    '''
    Evaluate the bilingual lexicon induction results of bilingual word embeddings
    by calculating the translation Precision@1 and Precision@5
    '''

    # Read the English to Chinese Dictionary as the gold standard
    en2zhDict = ReadEn2zhDict(dictFile)

    precision1 = []
    precision5 = []
    wordList = []
    testNum = 0
    testWordList = en2zhDict.keys()
    random.shuffle(testWordList)  # To test the random words

    start = time.clock()
    for enWord in testWordList:
        if enWord in enWordVecDict.keys():  # Notice: we havn't make sure that the corret Chinese words are in the vocabulary
            zhWordStdList = en2zhDict[enWord]
            zhWordSimList = []

            enWordVec = enWordVecDict[enWord]
            for zhWord in zhWordVecDict.keys():
                zhWordVec = zhWordVecDict[zhWord]
                # Since we have normalized the word vectors so that we can just
                # do dot product here
                cosSim = np.dot(zhWordVec, enWordVec)
                zhWordSimList.append((zhWord, cosSim))
            zhWordSimList.sort(key=lambda x: x[1], reverse=True)
            zhWordPreList = [x[0] for x in zhWordSimList[:5]]
            precision1.append(int(zhWordPreList[0] in zhWordStdList))
            precision5.append(
                int(len(set(zhWordStdList) & set(zhWordPreList)) > 0))
            wordList.append(enWord)

            testNum += 1
            if testNum % 200 == 0:
                print("Have tested %d words, the AP@1 is %f, the AP@5 if %f" %
                      (testNum, np.mean(precision1), np.mean(precision5)))
                print('Time Used: %f' % (time.clock() - start))
                start = time.clock()
            if testNum > maxTestNum:  # 减少测试时间！
                break
    print("Tested Words Number: %d" % testNum)
    print("Average Precision@1: %f" % np.mean(precision1))
    print("Average Precision@5: %f" % np.mean(precision5))
    return precision1, precision5, testNum


print("Read the Chinese Word Embeddings")
zhWordVecDict = ReadWordVec(zhWordVecFile)

print("Read the English Word Embeddings")
enWordVecDict = ReadWordVec(enWordVecFile)

zhWordSimFiles = ['wordsim-240', 'wordsim-297']
zhWordSimResults = []
enWordSimFiles = ['wordsim-353', 'SimLex-999']
enWordSimResults = []

for wordSimFile in zhWordSimFiles:
    zhWordSimResults.append(EvalWordSim(zhWordVecDict, wordSimFile))

for wordSimFile in enWordSimFiles:
    enWordSimResults.append(EvalWordSim(enWordVecDict, wordSimFile))

[p1, p5, testNum] = EvalLexiconInduction(
    zhWordVecDict, enWordVecDict, "en2zh_dict.txt", maxTestNum)


# Print the results again
print("Chinese WordSim Results:")
for i in range(len(zhWordSimFiles)):
    print(zhWordSimFiles[i] + ': Score: %f Skipped Word Pairs: %d' % (zhWordSimResults[i][
          0], zhWordSimResults[i][2]))
print("English WordSim Results:")
for i in range(len(enWordSimFiles)):
    print(enWordSimFiles[i] + ': Score: %f Skipped Word Pairs: %d' % (enWordSimResults[i][
          0], enWordSimResults[i][2]))
print("Bilingual Lexicon Induction Results:")
print("Test Words: %d P@1: %f P@5: %f" % (testNum,
                                          np.mean(p1), np.mean(p5)))
