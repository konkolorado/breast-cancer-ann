"""
Uriel Mandujano
Alpha Chau
CS68 BioInformatics
"""

from parser import *
from classes import *

# In this file we run the ANN unsing the most predictive genes
# stored in predictiveGenes.txt.

def main():

    NUM_GENES = 100

    nn = Network(1, NUM_GENES )

    cases = readDataFile()
    recurrence = readDemoFile()
    pred_genes = readPredictiveGenes( NUM_GENES )

    # data[patient] = [survival, [ list of gene expressions] ]
    data = combineData(cases, recurrence)

    # Train the Neural Net
    for patient in data:
        survival = data[patient][0]        
        expression = data[patient][1]
        use_genes  = []

        for g in pred_genes:
            use_genes.append( float(expression[g]) )
        
        nn.train(use_genes, survival)
    
    # Evaluate the Neural Net 5 fold CV
    

main()
