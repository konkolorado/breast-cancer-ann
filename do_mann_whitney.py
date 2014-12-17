"""
This file contains the functions necessary to perform the Mann-Whitney
U test for feature selection.
"""

from parser import *
from random import *
from scipy  import stats
from operator import itemgetter

def find_genes():
    # cases[patient] = [gene_expressions]
    cases = readDataFile()
    allPatients = cases.keys()
    choicePatients = []

    # Randomly choose 50 patients 100 times
    for i in range(100):
        patients = dict()
        while len(patients) < 50:
            p = choice(allPatients)
            if p not in patients:
                patients[p] = cases[p]
        
        choicePatients.append(patients)

    # For each patient, for every gene, calculate p value of getting that 
    # subset of genes. Elminate the lowest and hight 5% of p values and 
    # use the mean of the remaining p values as that gene's p value.
    all_group_probabilities = []

    # choicePatients is a list of dicts
    for num, group in enumerate(choicePatients):
        group_expressions = []
        all_group_probabilities.append( [] )

        
        for patient in group:
            group_expressions.append( group[patient] )

        for g in range( 1, len( group_expressions[0] )):
            group_gene_expressions = []

            for p in range(len( group_expressions )):
                group_gene_expressions.append( group_expressions[p][g] )

            
            all_gene_expressions = []
            for p in cases:
                all_gene_expressions.append( cases[p][g] )
            
            # p of group 
            p = stats.mannwhitneyu( group_gene_expressions,\
                      all_gene_expressions, use_continuity = True)[1]

            all_group_probabilities[num].append(p)
            print num, g

    # Calculate mean for each gene across each group
    pool = dict()
    for i in range( len( all_group_probabilities[0] ) ):
        mean = 0
        for j in range( len( all_group_probabilities) ):
            mean += all_group_probabilities[j][i]
        mean = mean / len(all_group_probabilities)
        pool[i] = mean

    # Choose smallest values in pool to be gene
    sorted_pool = sorted( pool.items(), key=itemgetter(1))
    outFile = open('predictiveGenes.txt', 'w')
    outFile.write( str(sorted_pool[:100] ) )
        
find_genes()
