from random import uniform
from math import exp

# ----------------------------------------------------------------- #

class Neuron(object):
	
	def __init__(self, bias = 0):
		self.incomingEdges = []
		self.outgoingEdges = []
		self.output = bias
		self.bias   = bias
			
	def addIncoming(self, incoming):
		self.incomingEdges.append(incoming)
	
	def addOutgoing(self, outgoing):
		self.outgoingEdges.append(outgoing)
		
	def sigmoid(self, x):
		return 1.0 / ( 1.0 + exp(-x) )
	
	def calcOutput(self):
		if self.bias:
			return

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
	def __init__(self):
		Neuron.__init__(self)
	
	def input(self, value):
		self.output = value
		
class OutputNeuron(Neuron):
	def __init__(self):
		Neuron.__init__(self)
		
class HiddenNeuron(Neuron):
	def __init__(self):
		Neuron.__init__(self)

# ----------------------------------------------------------------- #

class Edge(object):
	def __init__(self, src, dst):
		self.src = src
		self.dst = dst
		self.weight = uniform(-1, 1)
	
	def adjustWeight(self, delta):
		self.weight += delta

# ----------------------------------------------------------------- #
	
class Layer(object):
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
	NUM_ON = 1
	NUM_HN = 10
	LEARNING_RATE = 0.5
	
	def __init__(self, numHiddenLayer, numFeatures):
		
		self.inputLayer = Layer( numFeatures, 2)
		self.outputLayer = Layer(Network.NUM_ON, 3)
		
		self.hiddenLayers = []
		for i in range(numHiddenLayer):
			self.hiddenLayers.append( Layer(Network.NUM_HN, 1) )
			
		inNeurons = self.inputLayer.neurons
		outNeurons = self.outputLayer.neurons
		
		for h in self.hiddenLayers:
			hneurons = h.neurons

		# If using multiple hidden layers, must connect neurons within 
		# consecutive hdden layers to each other here. We didn't.	
			
		for n in inNeurons:
			for j in hneurons:
				edge = Edge(n,j)
				j.addIncoming(edge)
				n.addOutgoing(edge)
		for j in hneurons:
			for o in outNeurons:
				edge = Edge(j,o)
				o.addIncoming(edge)
				j.addOutgoing(edge)
				
	def propagate(self, example):
		
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
		prediction = self.propagate(example)
		
		error = prediction*(1-prediction) * (answer-prediction)
		
		# Back-Propagation.
		# Adjust edge weights between hidden and output
		incoming = self.outputLayer.neurons[0].incomingEdges
		for i in incoming:
			prevOutput = i.dst.output
			deltaWeight = prevOutput*error
			i.adjustWeight(deltaWeight*Network.LEARNING_RATE)
		
		# Adjust edge weights of hidden layers
		for neuron in self.hiddenLayers[0].neurons:
			# Sum hidden-output connections
			totalError = 0
			outgoing = neuron.outgoingEdges
			for edge in outgoing:
				totalError += edge.weight*error
			# Adjust edge weights from input to hidden
			incoming = neuron.incomingEdges
			for edge in incoming:
				prevOutput = neuron.output
				deltaHidden = prevOutput * (1 - prevOutput)
				deltaHidden *= totalError
				deltaWeight = edge.src.output * deltaHidden
				edge.adjustWeight(deltaWeight*Network.LEARNING_RATE)
		
		return prediction
		
# ----------------------------------------------------------------- #
