import csv
import pandas as pd
import numpy as np
import random
import math
import operator


def loadDataset(filename, split, trainingSet = [], testSet = []):
    df = pd.read_csv(filename,encoding='utf-8',header=None)
    dataset = np.array(df)
    for x in range(len(df)):
        if random.random() < split:
            trainingSet.append(dataset[x])
        else:
            testSet.append(dataset[x])


def euclideanDistance(instance1, instance2, length):
    distance = 0
    for x in range(length):
        distance += pow(instance1[x]-instance2[x],2)
    return math.sqrt(distance)



def getNeighbors(trainingSet, testInstance, k):
    distance = []
    length = len(testInstance)-1
    for x in range(len(trainingSet)):
        dis = euclideanDistance(trainingSet[x], testInstance, length)
        distance.append((trainingSet[x],dis))
    distance.sort(key=operator.itemgetter(1))
    neighbors = []
    for x in range(k):
        neighbors.append(distance[x][0])
    return neighbors




def getResponse(neighbors):
    classvotes = {}
    for x in range(len(neighbors)):
        response = neighbors[x][-1]
        if response in classvotes:
            classvotes[response] += 1
        else:
            classvotes[response] = 1
    sorted_votes = sorted(classvotes.items(),key=operator.itemgetter(1), reverse=True)
    return sorted_votes[0][0]



def getAccuracy(testSet, predictions):
    correct = 0
    for x in range(len(testSet)):
        if testSet[x][-1] == predictions[x]:
            correct += 1
    return (correct/float(len(testSet))) * 100


def main():
    #prepare data
    trainingSet = []
    testSet = []
    split = 0.67
    loadDataset('irisdata.txt', split, trainingSet, testSet)
    print('Train set: ' + repr(len(trainingSet)))
    print('Test set: ' + repr(len(testSet)))
    #generate predictions
    predictions = []
    k = 3
    for x in range(len(testSet)):
        # trainingsettrainingSet[x]
        neighbors = getNeighbors(trainingSet, testSet[x], k)
        result = getResponse(neighbors)
        predictions.append(result)
        print ('>predicted=' + repr(result) + ', actual=' + repr(testSet[x][-1]))
    accuracy = getAccuracy(testSet, predictions)
    print('Accuracy: ' + repr(accuracy) + '%')

if __name__ == '__main__':
    main()