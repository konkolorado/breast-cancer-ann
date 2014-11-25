"""
Alpha Chau
Uriel Mandujano
CS68 Final Project
"""

# Definition of the Neuron Class
from random import uniform 
from math import *

def activationFunction(x):
    return 1.0 / (1.0 + exp(-x))

# ------------------------------------------------------- #

class Node(object):
    
    def __init__(self):
        self.lastOutput = None
        self.lastInput  = None
        self.error      = None
        self.outgoingEdges = []
        self.incomingEdges = []
        self.addBias()

    def addBias(self):
        self.incomingEdges.append( Edge(BiasNode(), self) )
    
    # Need to store node's output because of duplicate
    # calls on a node and for use in training.
    def evaluate(self, inputVector):
        if self.lastOutput is not None:
            return self.lastOutput

        self.lastInput = []
        weightedSum = 0

        for edge in self.incomingEdges:
            theInput = edge.src.evaluate(inputVector)
            self.lastInput.append(theInput)
            weightedSum += edge.weight * theInput

        self.lastOutput = activationFunction(weightedSum)
        self.evaluateCache = self.lastOutput
        return self.lastOutput

    def getError(self, label):
        """
        Gets the error for a given node in the network.
        Error for output nodes are retrieved using label.
        Error for input nodes are ignored.
        """
        if self.error is not None:
            return self.error

        if self.outgoingEdges == []:  # This is an output node.
            self.error = label - self.lastOutput
        else:
            self.error = sum([edge.weight * edge.tgt.getError(label)
                                for edge in self.outgoingEdges])
        return self.error

    def updateWeights(self, learningRate):
        """
        Update the weights of a node and its successors.
        (Assumes self is not an InputNode.)
        Updated nodes have None for error, lastOutput, and lastInput.
        """
        if (self.error is not None and self.lastOutput is not None \
                and self.lastInput is not None):

            for i, edge in enumerate(self.incomingEdges):
                edge.weight += (learningRate * self.lastOutput * \
                        (1 - self.lastOutput) * self.error * self.lastInput[i])

            for edge in self.outgoingEdges:
                edge.tgt.updateWeights(learningRate)

            self.error = None
            self.lastInput = None
            self.lastOutput = None

    def clearEvaluateCache(self):
        if self.lastOutput is not None:
            self.lastOutput = None
            for edge in self.incomingEdges:
                edge.src.clearEvaluateCache()

# ------------------------------------------------------- #

class InputNode(Node):
    """
    Nodes that evaluate to the value of a specific index of the input.
    """

    def __init__(self, index):
        Node.__init__(self)
        self.index = index

    def evaluate(self, inputVector):
        self.lastOutput = inputVector[self.index]
        return self.lastOutput

    def updateWeights(self, learningRate):
        for edge in self.outgoingEdges:
            edge.tgt.updateWeights(learningRate)

    def getError(self, label):
        for edge in self.outgoingEdges:
            edge.tgt.getError(label)

    def addBias(self):
        pass

    def clearEvaluateCache(self):
        self.lastOutput = None

# ------------------------------------------------------- #

class BiasNode(InputNode):

    def __init__(self):
        Node.__init__(self)

    def evaluate(self, inputVector):
        return 1.0

# ------------------------------------------------------- #

class Edge(object):
    
    def __init__(self, src, tgt):
        self.weight = uniform(0, 1)
        self.src = src
        self.tgt = tgt
        
        src.outgoingEdges.append(self)
        tgt.incomingEdges.append(self)

# ------------------------------------------------------- #

class Network(object):
    
    def __init__(self):
        self.inputNodes = []
        self.outputNode = None

    def evaluate(self, inputVector):
        self.outputNode.clearEvaluateCache()
        return self.outputNode.evaluate(inputVector)

    def propagateError(self, label):
        for node in self.inputNodes:
            node.getError(label)

    def updateWeights(self, learningRate):
        for node in self.inputNodes:
            node.updateWeights(learningRate)

    def train(self, labelledExamples, learningRate=0.9, maxIter=10000):
        while maxIter > 0:
            for example, label in labelledExamples:
                output = self.evaluate(example)
                self.propagateError(label)
                self.updateWeights(learningRate)

                maxIter -= 1

# ------------------------------------------------------- #

if __name__ == '__main__':
    neurons = []
    for i in range(10):
        neurons.append( Neuron(1) )
        #print neurons[i]
    print Layer(1000,9)

    

