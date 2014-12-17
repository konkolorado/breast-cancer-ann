"""
This file contains the functions necessary to perform the Mann-Whitney
U test for feature selection.
"""

"""
- Select 50 patients randomly from all 198
- Use Mann-Whitney to choose 100 genes
    - for the 50 patients, sample w/ replacement 100 times to obtain 
        samples representative of the gene distribution

    - for each gene, remove outliers (top / bottom 5 percent)
    - calculate mean for the remaining 90% for each gene
    - obtain p values for each mean
    - include the 100 genes with the smallest p value

- Expand variables to include clinical variables
"""

from parser import *
from random import *
from scipy  import stats
from operator import itemgetter

# ------------------------------------------------------- #

def main():
    # cases[patient] = [gene_expressions]
    cases = readDataFile()
    allPatients = cases.keys()

    # randomly choose 50 patients
    patients = []
    while len(patients) < 50:
        p = choice(allPatients)

        if p not in patients:
            patients.append(p)
    
    # place each patient's expression profile into a matrix
    expressionMatrix = []
    for p in patients:
        expressionMatrix.append(cases[p])

    # for every gene
    smallestP = dict()
    #meansList = []
    for g in range(1,len(expressionMatrix[0])):
        # Bootstrp 100 times to obtain gene expressions representative of
        # the gene distribution

        randList = generateRandomNumberList(50, 100)
        
        measurementList = []
        for patient in randList:
            measurementList.append( float(expressionMatrix[patient][g]) )
        
        # remove top and bottom 5 percent data points
        measurementList = removeTopAndBottom( measurementList )
        
        # calculate mean of data points
        #mean = calcMean(measurementList)
        #meansList.append(mean)
        
        minP = 1
        for j in range(g+1, len(expressionMatrix[0])):
            comparisons = []
            for patient in randList:
                comparisons.append( float(expressionMatrix[patient][j] ) )

            # do the mann-whitney test and store test stat
            p=stats.mannwhitneyu(measurementList,comparisons,use_continuity=True)[1]
            if p < minP:
               minP = p
        
        # G is index of gene, minP is the smallest probability according to the mannwhit
        smallestP[g] = minP

    # Get smallest p-valued genes and use those as your predictors
    sorted_g = sorted(smallestP.items(), key=itemgetter(1))

    outFile = open('predictiveGenes', 'w')
    outFile.write(str(sorted_g[:100]))

# ------------------------------------------------------- #

def generateRandomNumberList(upper, number):
    """
    This function generates a list of size number of random numbers 
    between 0 and upper.
    Returns a list of random numbers.
    """
    randoms = []
    while len(randoms) < number:
        num = randrange(0,upper)
        randoms.append(num)

    return randoms

# ------------------------------------------------------- #

def removeTopAndBottom(measurementList):
    """
    This function removes the top and bottom 5 percent data points 
    in a given list.
    """
    n = len(measurementList)
    measurementList = sorted(measurementList)

    # Find 95th percentile
    p95 = int(.95 * n)
    measurementList = measurementList[:p95]
    
    # Find 5th percentile
    p05 = int(.05 * n)
    measurementList = measurementList[p05:]

    return measurementList

# ------------------------------------------------------- #

def calcMean(dataList):
    """
    This function calculates and returns the mean of a given list 
    of data.
    """
    total = 0
    length = len(dataList)

    for d in dataList:
        total += d

    return total / length 

# ------------------------------------------------------- #

# ------------------------------------------------------- #

main()
