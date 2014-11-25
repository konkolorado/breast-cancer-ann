"""
Uriel Mandujano & Alpha Chau

test.py - Test file for Artificial Neural Network
"""

from neuralNetwork import *

def basicTest():
    network = Network()
    inputNodes = [ InputNode(i) for i in range(3) ]
    hiddenNodes = [ Node() for i in range(3) ]
    outputNode = Node()

    # weight are all randomized.
    for inputNode in inputNodes:
        for node in hiddenNodes:
            Edge(inputNode, node)
        
    for node in hiddenNodes:
        Edge(node, outputNode)

    network.outputNode = outputNode
    network.inputNodes.extend(inputNodes)

    labelledExamples = [((0,0,0), 1),
                      ((0,0,1), 0),
                      ((0,1,0), 1),
                      ((0,1,1), 0),
                      ((1,0,0), 1),
                      ((1,0,1), 0),
                      ((1,1,0), 1),
                      ((1,1,1), 0)]

    network.train(labelledExamples, maxIter=5000)

    # test for consistency
    for number, isEven in labelledExamples:
        print "Error for %r is %0.4f. Output was:%0.4f" \
        % (number, isEven - network.evaluate(number), network.evaluate(number))

if __name__ == '__main__':
    basicTest()
