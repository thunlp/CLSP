# coding:utf8
'''
Input: Word vectors of source and target languages
Datasets: HowNet of English & Chinese versions, Sememe list
Method: Collaborative filtering
Output: MAP and mF1 scores of sememem prediction for  target words;
        And the specific prediction MAP and F1 scores with outputMode=1;
        And the all the results including neartest source words, predicted sememes and their scores for each target word with outputMode=2
'''
import sys
import numpy as np
from numpy import linalg
import time
import random


outputPath = sys.argv[1]
evalDataPath = sys.argv[2]
testNum = eval(sys.argv[3])
outputMode = eval(sys.argv[4])


def ReadSememeList(sememeFile):
    '''
    Read the bilingual sememes from the sememe list
    '''
    zh2enSememeDict = {}
    SememeList = []
    # 从义原列表中读取中英文义原列表
    start = time.clock()
    with open(evalDataPath + sememeFile, "r") as file:
        for line in file:
            enSememe, zhSememe = line.strip().split("|")
            SememeList.append(line.strip())
            zh2enSememeDict[zhSememe] = enSememe
    print("Sememe Reading Complete! Number of Sememes: %d" %
          len(SememeList))
    print("Time Used: %f" % (time.clock() - start))
    return zh2enSememeDict, SememeList


def ReadHowNet(HowNetFile, sememeList):
    '''
    Read the Hownet file and build the sememe dictionary for each word
    Notice that we only keep the sememes in the sememe list
    '''
    start = time.clock()
    HowNet = {}
    with open(evalDataPath + HowNetFile, "r") as file:
        for line in file:
            word, senseStr = line.strip().split("\t")
            setTmp = set([])
            for sense in senseStr.split(";"):
                for sememe in sense.strip("{}").split(","):
                    if sememe in sememeList:
                        setTmp.add(sememe)
            if setTmp:
                HowNet[word] = setTmp
    print("HowNet Reading Complete! Number of Words: %d" % len(HowNet))
    print("Time Used: %f" % (time.clock() - start))
    return HowNet


def ReadWordVec(wordVecFile, HowNet):
    '''
    Read word vectors from the word embedding file
    Notice that we only keep the words which appear in the HowNet
    '''
    start = time.clock()
    wordVecDict = {}
    num = 0
    with open(outputPath + wordVecFile, "r") as file:
        for line in file:
            num += 1
            if num % 500 == 0:
                print("Reading the %d-th word" % num)

            items = line.strip().split()
            if len(items) == 201:
                word = items[0]
                if word in HowNet.keys():
                    vec = np.array(map(eval, items[1:]))
                    if linalg.norm(vec) != 0:
                        wordVecDict[word] = vec / \
                            linalg.norm(vec)  # Normalization
    print("Word Embeddings Reading Complete! Number of Words:: %d" % num)
    print("Time Used: %f" % (time.clock() - start))
    return wordVecDict


def ReadWordFrequency(vocabFile, HowNet):
    '''
    Read word frequencies of English words from the vocabulary file
    '''
    # 读取英文词频 (存到列表中)
    start = time.clock()
    wordFreqDict = {}
    num = 0
    with open(outputPath + vocabFile, "r") as file:
        for line in file:
            num += 1
            if num % 500 == 0:
                print("Reading the %d-th word" % num)
            word, freq = line.strip().split()
            if word in HowNet.keys():
                wordFreqDict[word] = freq
    print("Word Frequency Reading Complete! Number of Words:: %d" % num)
    print("Time Used: %f" % (time.clock() - start))


def Get_AP(sememeStd, sememePre):
    '''
    Calculate the Average Precision of sememe prediction
    '''
    AP = 0
    hit = 0
    for i in range(len(sememePre)):
        if sememePre[i] in sememeStd:
            hit += 1
            AP += float(hit) / (i + 1)
    if AP == 0:
        print("Calculate AP Error")
        print("Sememe Standard:" + " ".join(sememeStd))
        print("Sememe Predicted:" + " ".join(sememePre))
        return 0
    else:
        AP /= float(hit)
    return AP


def Get_F1(sememeStdList, sememeSelectList):
    '''
    Calculate the F1 score of sememe prediction
    '''
    TP = len(set(sememeStdList) & set(sememeSelectList))
    FP = len(sememeSelectList) - TP
    FN = len(sememeStdList) - TP
    precision = float(TP) / (TP + FN)
    recall = float(TP) / (TP + FP)
    if (precision + recall) == 0:
        return 0
    F1 = 2 * precision * recall / (precision + recall)
    return F1

print("Start Reading Sememe List")
zh2enSememeDict, SememeList = ReadSememeList("sememe_1400_EnZh.txt")

print("Start Reading English HowNet")
enHowNet = ReadHowNet("HowNet_english_version.txt", SememeList)

print("Start Reading Chinese HowNet")
zhHowNet = ReadHowNet("HowNet_chinese_version.txt", SememeList)

print("Start Reading English Word Vectors")
enWordVecDict = ReadWordVec("word-vec.en", enHowNet)

print("Start Reading Chinese Word Vectors")
zhWordVecDict = ReadWordVec("word-vec.zh", zhHowNet)

print("Start Reading English Word Frequencies")
enWordFreqDict = ReadWordFrequency("vocab.en", enHowNet)

# Start Predicting Sememes
# Set hyper-parameters
K = 100  # number of nearest source words for each target word when predicting
c = 0.8  # declining coefficient
simThresh = 0.5  # threshold of chosen sememe score

start = time.clock()

testWordList = enWordVecDict.keys()
random.shuffle(testWordList)
testWordList = testWordList[:testNum]

now = 0
allResults = []
for enWord in testWordList:
    now += 1
    if now % 100 == 0:
        print("Have looked for sememes for %d English words" % now)
        print("Time Used: %f" % (time.clock() - start))

    enWordVec = enWordVecDict[enWord]
    # Sort source words according the cosine similarity
    zhWordSimList = []
    for zhWord in zhWordVecDict.keys():
        zhWordVec = zhWordVecDict[zhWord]
        cosSim = np.dot(zhWordVec, enWordVec)
        zhWordSimList.append((zhWord, cosSim))
    zhWordSimList.sort(key=lambda x: x[1], reverse=True)
    zhWordSimList = zhWordSimList[:K]

    # Calculate the score of each sememe
    sememeScore = {}
    rank = 1
    for word, cosSim in zhWordSimList:
        sememes = zhHowNet[word]
        for sememe in sememes:
            if sememe in sememeScore.keys():
                sememeScore[sememe] += cosSim * pow(c, rank)
            else:
                sememeScore[sememe] = cosSim * pow(c, rank)
        rank += 1
    # Save the sorted sememes and their scores
    sortedSememe = sorted(sememeScore.items(),
                          key=lambda x: x[1], reverse=True)

    sememePreList = [x[0] for x in sortedSememe]
    sememeStdList = list(enHowNet[enWord])
    # Calculate MAP
    AP = Get_AP(sememeStdList, sememePreList)
    # Calculate F1 score
    tmp = [x for x in sortedSememe if x[1] > simThresh]
    if tmp == []:
        # Choose the first one sememe if all the semems get scores lower than
        # the threshold
        tmp = sortedSememe[:1]
    sememeSelectList = [x[0] for x in tmp]
    F1 = Get_F1(sememeStdList, sememeSelectList)

    allResults.append([enWord, enWordFreqDict[enWord],
                       zhWordSimList, sortedSememe, AP, F1])
print("Sememe Prediction Complete")

print("mAP: %f" % np.mean([x[4] for x in allResults]))
print("mean F1: %f" % np.mean([x[5] for x in allResults]))

# Save all the results into the file
if outputMode > 0:
    with open(outputPath + "SememePreResults.txt", "w") as file:
        file.write("Word\tFrequency\tAP\tF1\n")
        for word, frequency, zhWordSimList, sortedSememe, AP, F1 in allResults:
            file.write(word + "\t" + frequency + "\t" +
                       str(AP) + "\t" + str(F1) + "\n")
            if outputMode > 1:
                file.write("\tNeartest Source Words:", zhWordSimList, "\n")
                file.write("\tSememes and Scores:", sortedSememe, "\n")
