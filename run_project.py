"""
Uriel Mandujano
Alpha Chau
CS68 BioInformatics
"""

from parser import *
from classes import *
from random import choice

# In this file we run the ANN using the most predictive genes
# stored in predictiveGenes.txt.

def main():

    NUM_GENES = 50
    nn = Network(1, NUM_GENES )

    cases = readDataFile()
    recurrence = readDemoFile()

    # !!! file fun_mann_whitney.py needs to have been run previously
    # file identifies predictive genes and dumps to file name 
    # predictiveGenes.txt
    pred_genes = readPredictiveGenes( NUM_GENES )

    # data[patient] = [survival, [ list of gene expressions] ]
    data = combineData(cases, recurrence)
    
    overallPercent = 0.0
    numRuns = 25
    for i in range( numRuns ):
        # trainData and testData of same form as data
        newData = divideData(data)
        trainData = newData[0]
        testData = newData[1]

        # Train the Neural Net
        for patient in trainData:
            survival = trainData[patient][0]        
            expression = trainData[patient][1]
            use_genes  = []

            for g in pred_genes:
                use_genes.append( float(expression[g]) )
            
            nn.train(use_genes, survival)
        
        # Evaluate the Neural Net
        correct = 0.0
        for patient in testData:
            survival = testData[patient][0]
            expression = testData[patient][1]
            use_genes  = []

            for g in pred_genes:
                use_genes.append( float(expression[g]) )

            prediction = nn.propagate(use_genes)
            prediction = round(prediction)
            if prediction == survival:
                correct += 1
        
        overallPercent += correct / len(testData)

    print overallPercent / numRuns

# ------------------------------------------------------- #

def divideData(data):
    """
    This function divides the provided dictionary of data into 
    fifths (.2) and returns a tuple:
        tuple[0] = .8 ----> training data
        tuple[1] = .2 ----> test data
    """
    # data is of the form:
    # data[patient] = [survival, [ list of gene expressions] ]

    allPatients, total = data.keys(), len(data.keys())
    train, test = [], []
    train_len = 4 * total / 5
    test_len  = total / 5
    
    # Split the patients into train and test group
    while len( train ) < train_len:
        p = choice( allPatients )
        if p not in train:
            train.append(p)
    for p in allPatients:
        if p not in train:
            test.append(p)

    # Create dictionaries for the train and test groups
    trainData, testData = dict(), dict()

    for patient in train:
        trainData[patient] = data[patient]
    for patient in test:
        testData[patient] = data[patient]
    
    return trainData, testData

# ------------------------------------------------------- #

main()
