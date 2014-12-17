from random import uniform
from math import exp
import numpy as np

# ----------------------------------------------------------------- #

class Neuron(object):
        """
        Represents a generic neuron in the Artificial Neural Network (ANN).
        """
	# General node if bias is not set; Bias node if bias is set.
	def __init__(self, bias = 0):
		self.incomingEdges = []
		self.outgoingEdges = []
		self.output = bias
		self.bias   = bias
			
	def addIncoming(self, incoming):
		self.incomingEdges.append(incoming)
	
	def addOutgoing(self, outgoing):
		self.outgoingEdges.append(outgoing)
	
	# The activation function of the ANN.
	def sigmoid(self, x):
		return 1.0 / ( 1.0 + np.exp(-x) )
	
	def calcOutput(self):
		# Ignore if bias node.
		if self.bias:
			return
		# Calculate the output of this node based on incoming nodes & weights.
		total = 0
		incoming = self.incomingEdges
		for i in incoming:
			src = i.src
			dst = i.dst
			
			if src.bias:
				biasVal = i.weight * src.output  
			if not src.bias:
				total += i.weight * src.output
		
		self.output = self.sigmoid(biasVal + total)
				
class InputNeuron(Neuron):
	"""
	Represents an input neuron that should output the input of the ANN. 
	"""
	def __init__(self):
		Neuron.__init__(self)
	
	def input(self, value):
		self.output = value
		
class OutputNeuron(Neuron):
	"""
	Represents an output neuron that should output the final output of the ANN.
	"""
	def __init__(self):
		Neuron.__init__(self)
		
class HiddenNeuron(Neuron):
	"""
	Represents a hidden neuron in the ANN.
	"""
	def __init__(self):
		Neuron.__init__(self)

# ----------------------------------------------------------------- #

class Edge(object):
	"""
	Represents a weighted connection between nodes in the ANN.
	"""
	# Initial weights are selected randomly.
	def __init__(self, src, dst):
		self.src = src
		self.dst = dst
		self.weight = uniform(-1, 1)
	
	def adjustWeight(self, delta):
		self.weight += delta

# ----------------------------------------------------------------- #
	
class Layer(object):
	"""
	Represents a layer of nodes in the ANN.
	"""
	def __init__(self, numNeuron, type):
		"""
		Type represents layer type:
			1 = Hidden
			2 = Input
			3 = Output
			
		"""
		BIAS = -1
		
		self.neurons = []
		if type == 1:
			for i in range(numNeuron):
				self.neurons.append( HiddenNeuron() )
			self.neurons.append( Neuron(BIAS) )		 # Add bias node
		elif type == 2:
			for i in range(numNeuron):
				self.neurons.append( InputNeuron() )
			self.neurons.append( Neuron(BIAS) )		 # Add bias node
		elif type == 3:
			for i in range(numNeuron):
				self.neurons.append( OutputNeuron() )
	
	def getSize(self):
		return len(self.neurons)
	
# ----------------------------------------------------------------- #
	
class Network(object):
	"""
	Represents a neural network that learns through Back-Propagation.
	"""

	NUM_ON = 1
	NUM_HN = 15
	LEARNING_RATE = 0.3
	INPUT_TYPE = 2
	OUTPUT_TYPE = 3
	
	def __init__(self, numHiddenLayer, numFeatures):
		
		# Create nodes and layers.		
		self.inputLayer = Layer( numFeatures, Network.INPUT_TYPE)
		self.outputLayer = Layer(Network.NUM_ON, Network.OUTPUT_TYPE)
		
		self.hiddenLayers = []
		for i in range(numHiddenLayer):
			self.hiddenLayers.append( Layer(Network.NUM_HN, 1) )
			
		# Connect nodes between layers.
		inNeurons = self.inputLayer.neurons
		outNeurons = self.outputLayer.neurons
		
		# NOTE: If using multiple hidden layers, must connect neurons within 
		# consecutive hidden layers to each other here. We didn't.
		for h in self.hiddenLayers:
			hNeurons = h.neurons

		for n in inNeurons:
			for j in hNeurons:
				edge = Edge(n,j)
				j.addIncoming(edge)
				n.addOutgoing(edge)
		for j in hNeurons:
			for o in outNeurons:
				edge = Edge(j,o)
				o.addIncoming(edge)
				j.addOutgoing(edge)
				
	def propagate(self, example):
		"""
		This function performs the feed-forward operation of an example, i.e.
		a vector of features, through the ANN and returns an output value.  
		"""

		# Push input vector into the input layer
		for i, neuron in enumerate(self.inputLayer.neurons):
			if neuron.bias:
				continue
			if not neuron.bias:
				neuron.input(example[i])
		
		# Calculate new outputs for the hidden layers
		for hlayer in self.hiddenLayers:
			for neuron in hlayer.neurons:
				neuron.calcOutput()
		
		# Calculate the output for the output neuron
		for neuron in self.outputLayer.neurons:
			neuron.calcOutput()
		
		# Return output layer's output
		return self.outputLayer.neurons[0].output
		
	def train(self, example, answer):
		"""
		This function trains the ANN given an example (a vector of features) and 
		an answer (the desired label for the example) using Back-Propagation.
		"""
		# Determine prediction and error.
		prediction = self.propagate(example)
		error = prediction*(1-prediction) * (answer-prediction)
		
		# Back-Propagation.
		# Adjust edge weights between hidden nodes and output.
		incoming = self.outputLayer.neurons[0].incomingEdges
		for i in incoming:
			prevOutput = i.dst.output
			deltaWeight = prevOutput*error
			i.adjustWeight(deltaWeight*Network.LEARNING_RATE)
		
		# Adjust edge weights of hidden layers.
		for neuron in self.hiddenLayers[0].neurons:
			# Sum hidden-output connections.
			totalError = 0
			outgoing = neuron.outgoingEdges
			for edge in outgoing:
				totalError += edge.weight*error
			# Adjust edge weights from input to hidden.
			incoming = neuron.incomingEdges
			for edge in incoming:
				prevOutput = neuron.output
				# Based on derivative of activation function.
				deltaHidden = prevOutput * (1 - prevOutput)
				deltaHidden *= totalError
				deltaWeight = edge.src.output * deltaHidden
				edge.adjustWeight(deltaWeight*Network.LEARNING_RATE)
		
		return prediction
		
# ----------------------------------------------------------------- #
