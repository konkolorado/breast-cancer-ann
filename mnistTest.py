from classes import *
import cPickle, gzip, numpy, theano
import theano.tensor as T

def main():
	
	f = gzip.open('mnist.pkl.gz', 'rb')
	# Each set is [imagesData, labels]
	# imagesData = 784 floats between 0 (black) and 1 (white)
	# Training = 50000 images
	# Test = 10000 images
	train_set, valid_set, test_set = cPickle.load(f)
	f.close()

	"""
	# Memory management for large dataset.
	test_set_x, test_set_y = shared_dataset(test_set)
	valid_set_x, valid_set_y = shared_dataset(valid_set)
	train_set_x, train_set_y = shared_dataset(train_set)

	batch_size = 500    # size of the minibatch
	"""

	# Accessing the first minibatch of the training set
	trainData  = train_set[0][0 * 500: 1 * 500]
	trainLabel = train_set[1][0 * 500: 1 * 500]

	numFeatures = len(train_set[0][0])
	testNetwork = Network(1, numFeatures)

	for i, image in enumerate(trainData):
		truth = trainLabel[i] / 10.0
		testNetwork.train(image, truth)
	
	differences = {}
	correct, totalGuess = 0.0, 0.0
	minDiff = float("inf")
	# Run tests.
	for i in range(25):
		predictVal = testNetwork.propagate(test_set[0][i])
		# Look for best digit match per guess and measure accuracy.
		for j in range(10):
			differences[j] = abs(predictVal - j/10.0)
			for entry in differences:
				if differences[entry] < minDiff:
					minDiff = differences[entry]
					prediction = entry
		if prediction == test_set[1][i]:
			correct += 1
			print "Prediction was correct!"
		totalGuess += 1

		differences.clear()
		minDiff = float("inf")
		print "Test: ", prediction
		print "Actual: ", test_set[1][i]

	print "Accuracy: ", correct/totalGuess


	"""
	ls = [
			[0, 0, 0, 0, 0, 0],
			[0, 0, 0, 0, 0, 0],
			[0, 0, 0, 0, 0, 0],
			[0, 0, 0, 0, 0, 0],
			[0, 0, 0, 0, 0, 0],
			[0, 0, 0, 0, 0, 0],
			[0, 0, 0, 0, 0, 0],
			[0, 0, 0, 0, 0, 0],
			[0, 0, 0, 0, 0, 0],
			[1, 1, 1, 1, 1, 1],
			[1, 1, 1, 1, 1, 1],
			[1, 1, 1, 1, 1, 1],
			[1, 1, 1, 1, 1, 1],
			[1, 1, 1, 1, 1, 1],
			[1, 1, 1, 1, 1, 1],
			[1, 1, 1, 1, 1, 1],
			[1, 1, 1, 1, 1, 1],
			[1, 1, 1, 1, 1, 1],
			[1, 1, 1, 1, 1, 1],
			[1, 1, 1, 1, 1, 1],
			[1, 1, 1, 1, 1, 1],
			[1, 1, 1, 1, 1, 1],
			[1, 1, 1, 1, 1, 1],
			[1, 1, 1, 1, 1, 1],
			[1, 1, 1, 1, 1, 1],
			[1, 1, 1, 1, 1, 1],
			[1, 1, 1, 1, 1, 1],
			[1, 1, 1, 1, 1, 1],
			[1, 1, 1, 1, 1, 1],
			[1, 1, 1, 1, 1, 1],
			[1, 1, 1, 1, 1, 1],
			[1, 1, 1, 1, 1, 1],
			[1, 1, 1, 1, 1, 1],
			[1, 1, 1, 1, 1, 1],
			[1, 1, 1, 1, 1, 1],
			[1, 1, 1, 1, 1, 1],
			[1, 1, 1, 1, 1, 1],
			[1, 1, 1, 1, 1, 1],
			[0, 0, 0, 0, 0, 0]
		  ]
		
	test = [0, 0, 0, 0, 0, 0]

	testNetwork = Network(1, 6)
	for i in ls:
		for j in range(10):
			if i[0] == 0:
				truth = 0
			if i[0] == 1:
				truth = 1
			testNetwork.train(i, truth)
	print "prop" ,testNetwork.propagate(test)
	"""

def shared_dataset(data_xy):
    """ Function that loads the dataset into shared variables

    The reason we store our dataset in shared variables is to allow
    Theano to copy it into the GPU memory (when code is run on GPU).
    Since copying data into the GPU is slow, copying a minibatch everytime
    is needed (the default behaviour if the data is not in a shared
    variable) would lead to a large decrease in performance.
    """
    data_x, data_y = data_xy
    shared_x = theano.shared(numpy.asarray(data_x, dtype=theano.config.floatX))
    shared_y = theano.shared(numpy.asarray(data_y, dtype=theano.config.floatX))
    # When storing data on the GPU it has to be stored as floats
    # therefore we will store the labels as ``floatX`` as well
    # (``shared_y`` does exactly that). But during our computations
    # we need them as ints (we use labels as index, and if they are
    # floats it doesn't make sense) therefore instead of returning
    # ``shared_y`` we will have to cast it to int. This little hack
    # lets us get around this issue
    return shared_x, T.cast(shared_y, 'int32')

main()