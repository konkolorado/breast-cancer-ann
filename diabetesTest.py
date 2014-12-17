from classes import *

def main():
	# createFile()
	data = []
	trainData, trainLabel = [], []
	testData, testLabel = [], []
	dataFile = open('filtered_data.txt', 'r')
	for i, line in enumerate(dataFile):
		line = line.split(',')
		if len(line) != 5:
			continue

		for j, entry in enumerate(line):
			line[j] = int(entry)
		if i < 8000:
			trainLabel.append(line[0])
			for index in range(1, 5):
				data.append(line[index])
			trainData.append(data)
		else:
			testLabel.append(line[0])
			for index in range(1, 5):
				data.append(line[index])
			testData.append(data)
		data = []

	numFeatures = len(trainData[0])
	testNetwork = Network(1, numFeatures)

	for i, patientData in enumerate(trainData):
			race = trainLabel[i]
			testNetwork.train(patientData, race)
	
	correct, totalGuess = 0.0, 0.0
	for i in range(50):
		predictionVal = testNetwork.propagate(testData[i])
		if predictionVal < 0.5:
			prediction = 0
		else:
			prediction = 1
		
		if prediction == testLabel[i]:
			correct += 1
		totalGuess += 1
		print "Test: ", prediction
		print "Actual: ", testLabel[i]

	print "Accuracy:", correct/totalGuess


def createFile():
	"""
	This function creates a file based on the diabetes information of the first 
	10,000 patients from the Center for Clinical and Translational Research

	[2] race: 1 if White, 0 if African American
	[9] number_visit_to_hospital
	[12] number_lab_procedures
	[14] number_medications
	[21] number_diagnoses
	[47] change: 1 if present, 0 if none
	"""
	rawFile = open('diabetic_data.csv', 'r')
	newFile = open('filtered_data.txt', 'w')

	delimiter = ","
	newLine = []
	for i, line in enumerate(rawFile):
		if i == 0:
			continue
		if i == 10000:
			break
		line = line.split(',')
		if line[2] is not "?" and \
			line[9] is not "?" and \
			line[12] is not "?" and \
			line[14] is not "?" and \
			line[21] is not "?" and \
			line[47] is not "?" and \
			line[49] is not "?":
			if line[2] == "Caucasian":
				newLine.append("1")
			elif line[2] == "AfricanAmerican":
				newLine.append("0")
			newLine.append(line[9])
			newLine.append(line[12])
			newLine.append(line[14])
			newLine.append(line[21])
			if line[49] == "Ch":
				newLine.append("1")
			elif line[49] == "No":
				newLine.append("0")
			newFile.write(delimiter.join(newLine) + "\n")
			newLine = []

main()