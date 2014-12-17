"""
This file contains functions necessary to parse the gene expression 
data file.
"""

# ------------------------------------------------------- #

def main():
    cases = readDataFile()
    recurrence = readDemoFile()
    data = combineData(cases, recurrence)

# ------------------------------------------------------- #

def readDataFile():
    """
    This function reads the gene expression data file stored locally on 
    the computer flour. It creates a dictionary where keys are a patient's
    name and the value is a list of the patient's gene expression data.
    """

    dataFile = "/local/data.txt"
    inFile = open(dataFile, 'r')

    variables = {}
    cases     = {}
    i         = 0

    for line in inFile:
        if i == 0:
            line = line.split(',')

            for item in line:
                variables[item] = []
    
        if i != 0:
            line = line.split(',')
            name = line[0]
            cases[name] = []

            for j in range(len(line) - 1):
                cases[name].append(line[j])

        i += 1

    # Cases dictionary of the form:
    # cases[patient_name] = [expression1, expression2 .... expression22283]

    inFile.close()
    return cases

# ------------------------------------------------------- #

def readDemoFile():
    """
    This function reads the given demographics file and 
    stores patient name and breast cancer recurrence 
    with 5 years into a dictionary.
    """

    demoFile = "/local/demographics.txt"
    inFile = open(demoFile, 'r')
    
    recurrence = {}
    i = 0
    for line in inFile:
        
        if i != 0:
            line = line.split(',')
            name = line[0]
            
            years_survival = int(line[18]) / 365.0
            if years_survival < 5:
                rec = 1
            if years_survival >= 5:
                rec = 0
            
            # Recurrence dictionary of the form:
            # recurrence[patient_name] = 5 yr recurrence or not
            recurrence[name] = rec

        i += 1

    inFile.close()
    return recurrence

# ------------------------------------------------------- #

def combineData(cases, recurrence):

    patientInfo = dict()
    for patient in recurrence:
        patientInfo[patient] = [ recurrence[patient], cases[patient] ]
    
    # patientInfo[name] = [ recurrence_within_5_yrs, list_of_gene_expressions]
    return patientInfo

# ------------------------------------------------------- #

if __name__ == '__main__':
    main()
