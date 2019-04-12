import os

inFilePred = open('F:/celeba/test/results/vgg16/normcls/testpredict.txt','r')
inFileGtruth = open('F:/celeba/test/testlabel.txt','r')

allPredLine = []
predContent = inFilePred.readlines()
for predLine in predContent:
    splitPredLine = predLine.split(' ')[1:-1]
    allPredLine.append(splitPredLine)

allGtruthLine = []
GtruthContent = inFileGtruth.readlines()
for GtruthLine in GtruthContent:
    splitGtruthLine = GtruthLine[:-1].split(' ')[1:]
    allGtruthLine.append(splitGtruthLine)

totalerrorPercent = 0
for i in range(0,40):
    correctCount = 0
    for j in range(0,len(allPredLine)):
        if allPredLine[j][i] == allGtruthLine[j][i]:
            correctCount+=1
    errorPercent = 1.0 - float(correctCount)/len(allPredLine)
    totalerrorPercent += errorPercent
    print('err: '+ str(errorPercent))
print('avgError: ' + str(totalerrorPercent/40.0))
