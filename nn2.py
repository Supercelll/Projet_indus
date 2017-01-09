#!/usr/bin/env python
# -----------------------------------------------------------------------------
# Multi-layer perceptron
# Copyright (C) 2011  Nicolas P. Rougier
#
# Distributed under the terms of the BSD License.
# -----------------------------------------------------------------------------
# This is an implementation of the multi-layer perceptron with retropropagation
# learning.
# -----------------------------------------------------------------------------
import numpy as np


def sigmoid(x):
    ''' Sigmoid like function using tanh '''
    return np.tanh(x)


def dsigmoid(x):
    ''' Derivative of sigmoid above '''
    return 1.0-x**2


class MLP:
    ''' Multi-layer perceptron class. '''

    def __init__(self, *args):
        ''' Initialization of the perceptron with given sizes.  '''

        self.shape = args
        n = len(args)

        # Build layers
        self.layers = []
        # Input layer (+1 unit for bias)
        self.layers.append(np.ones(self.shape[0]+1))
        # Hidden layer(s) + output layer
        for i in range(1, n-1):  # (1,n) à l'origine et sans le +1 dans les couches ni présence de la couche sortie
            self.layers.append(np.ones(self.shape[i]+1))  # Tentative de mettre 1 biais par couche cachée
        self.layers.append(np.ones(self.shape[-1]))  # Couche sortie

        # Build weights matrix (randomly between -0.25 and +0.25)
        self.weights = []
        for i in range(n-1):
            self.weights.append(np.zeros((self.layers[i].size,
                                         self.layers[i+1].size)))

        # dw will hold last change in weights (for momentum)
        self.dw = [0, ]*len(self.weights)

        # Reset weights
        self.reset()

    def reset(self):
        ''' Reset weights '''

        for i in range(len(self.weights)):
            Z = np.random.random((self.layers[i].size, self.layers[i+1].size))
            self.weights[i][...] = (2*Z-1)*0.25

    def propagate_forward(self, data):
        ''' Propagate data from input layer to output layer. '''

        # Set input layer
        self.layers[0][0:-1] = data

        # Propagate from layer 0 to layer n-1 using sigmoid as activation function
        for i in range(1, len(self.shape)):
            # Propagate activity
            self.layers[i][...] = sigmoid(np.dot(self.layers[i-1], self.weights[i-1]))  # [...] en [0:-1] pour biais

        # Return output
        return self.layers[-1]

    def propagate_backward(self, target, lrate=0.1, momentum=0.1):
        ''' Back propagate error related to target using lrate. '''

        deltas = []

        # Compute error on output layer
        error = target - self.layers[-1]
        delta = error*dsigmoid(self.layers[-1])
        deltas.append(delta)

        # Compute error on hidden layers
        for i in range(len(self.shape)-2, 0, -1):
            delta = np.dot(deltas[0], self.weights[i].T)*dsigmoid(self.layers[i])
            deltas.insert(0, delta)

        # Update weights
        for i in range(len(self.weights)):
            layer = np.atleast_2d(self.layers[i])
            delta = np.atleast_2d(deltas[i])
            dw = np.dot(layer.T, delta)
            self.weights[i] += lrate*dw + momentum*self.dw[i]
            self.dw[i] = dw

        # Return error
        return (error**2).sum()


def map_to_input(map, start, end):
    input = []
    for i in range(len(map)):
        for j in range(len(map[0])):
            if (i, j) == start:
                input.append(1)
            elif (i, j) == end:
                input.append(1)
            elif map[i][j].blocked:
                input.append(-1)
            else:
                input.append(0)
    return input


def map_to_result(map, start, end, path):
    result = []
    for i in range(len(map)):
        for j in range(len(map[0])):
            if (i, j) == start:
                result.append(1)
            elif (i, j) == end:
                result.append(1)
            elif (i, j) in path and (i, j) != start and (i, j) != end:
                result.append(1)
            elif map[i][j].blocked:
                result.append(-1)
            else:
                result.append(0)
    return result


def output_to_chemin(output, map_width):  # besoin de l'entrée start/end
    # pour ordonner le print
    temp = []
    list_coord = []
    for i in range(len(output)):
        if abs(output[i] - 1) < 0.00000001:  # marge d'erreur car le réseau ne renverra jamais 1 exactement
            temp.append(i)
    for pos in temp:
        list_coord.append((pos//map_width, pos % map_width))  # on reforme les coordonnées avec la position
        # dans le vecteur ligne
    # On réordonne pour avoir le même format que le chemin sortie par astar

    def get_key(item):
        return item[0]  # Pour sort selon la première valeur
    sorted(list_coord, key=get_key)

    return list_coord


def learn_path_format(network, samples, epochs=2500, lrate=.1, momentum=0.1):  # Version affichant des résultats
    #  lisibles plutôt qu'une énorme liste
    # Train
    for i in range(epochs):
        n = np.random.randint(samples.size)
        network.propagate_forward(samples['input'][n])
        network.propagate_backward(samples['output'][n], lrate, momentum)
    # Test
    for i in range(samples.size):
        o = network.propagate_forward(samples['input'][i])
        o_displayed = output_to_chemin(o, 10)   # WARNING: nombre = MAP_HEIGHT!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        print(i, samples['input'][i], o_displayed, )
        print('expected: ', output_to_chemin(samples['output'][i], 10))  # ici aussi


def modif(output, err):
    for k in range(len(output)):
        if output[k] > 1-err:
            output[k] = 1
        elif output[k] < -1+err:
            output[k] = -1
        else:
            output[k] = 0


def learn_path_format2(network, samples, epochs=2500, lrate=.1, momentum=0.1):  # Version affichant des résultats
    #  lisibles plutôt qu'une énorme liste
    #  Ajout du traitement de la sortie afin de remplacer tout par -1 0 1
    # Train
    for i in range(epochs):
        n = np.random.randint(samples.size)
        network.propagate_forward(samples['input'][n])
        network.propagate_backward(samples['output'][n], lrate, momentum)
    # Test
    for i in range(samples.size):
        o = network.propagate_forward(samples['input'][i])
        print(o)
        modif(o, 0.00000001)               # 0.1 = marge d'erreur/validation, faire attention à ce que ça corresponde avec
        # la valeur de output_to_chemin
        print(o)
        o_displayed = output_to_chemin(o, 18)  # WARNING: nombre = MAP_HEIGHT!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        print(i, samples['input'][i], o_displayed, )
        print('expected: ', output_to_chemin(samples['output'][i], 18))  # ici aussi

# -----------------------------------------------------------------------------
if __name__ == '__main__':
    import matplotlib
    import matplotlib.pyplot as plt

    def learn(network, samples, epochs=2500, lrate=.1, momentum=0.1):
        # Train
        for i in range(epochs):
            n = np.random.randint(samples.size)
            network.propagate_forward(samples['input'][n])
            network.propagate_backward(samples['output'][n], lrate, momentum)
        # Test
        for i in range(samples.size):
            o = network.propagate_forward(samples['input'][i])
            print(i, samples['input'][i], '%.2f' % o[0],)
            print('(expected %.2f)' % samples['output'][i])


    network = MLP(2, 2, 1)
    samples = np.zeros(4, dtype=[('input',  float, 2), ('output', float, 1)])

    # Example 1 : OR logical function
    # -------------------------------------------------------------------------
    print("Learning the OR logical function")
    network.reset()
    samples[0] = (0, 0), 0
    samples[1] = (1, 0), 1
    samples[2] = (0, 1), 1
    samples[3] = (1, 1), 1
    learn(network, samples)

    # Example 2 : AND logical function
    # -------------------------------------------------------------------------
    print("Learning the AND logical function")
    network.reset()
    samples[0] = (0, 0), 0
    samples[1] = (1, 0), 0
    samples[2] = (0, 1), 0
    samples[3] = (1, 1), 1
    learn(network, samples)

    # Example 3 : XOR logical function
    # -------------------------------------------------------------------------
    print ("Learning the XOR logical function")
    network.reset()
    samples[0] = (0, 0), 0
    samples[1] = (1, 0), 1
    samples[2] = (0, 1), 1
    samples[3] = (1, 1), 0
    learn(network, samples)

    # Example 4 : Learning sin(x)
    # -------------------------------------------------------------------------
    print("Learning the sin function")
    network = MLP(1, 10, 1)
    samples = np.zeros(500, dtype=[('x',  float, 1), ('y', float, 1)])
    samples['x'] = np.linspace(0, 1, 500)
    samples['y'] = np.sin(samples['x']*np.pi)

    for i in range(10000):
        n = np.random.randint(samples.size)
        network.propagate_forward(samples['x'][n])
        network.propagate_backward(samples['y'][n])

    plt.figure(figsize=(10, 5))
    # Draw real function
    x, y = samples['x'], samples['y']
    plt.plot(x, y, color='b', lw=1)
    # Draw network approximated function
    for i in range(samples.shape[0]):
        y[i] = network.propagate_forward(x[i])
    plt.plot(x, y, color='r', lw=3)
    plt.axis([0, 1, 0, 1])
    plt.show()

    # --------------------------------------------------------------------------
    # Zone pour simu lidar
    # Test pathfinding
    def learn_path(network, samples, epochs=2500, lrate=.1, momentum=0.1):
        # Train
        for i in range(epochs):
            n = np.random.randint(samples.size)
            network.propagate_forward(samples['input'][n])
            network.propagate_backward(samples['output'][n], lrate, momentum)
        # Test
        for i in range(samples.size):
            o = network.propagate_forward(samples['input'][i])
            print(i, samples['input'][i], o,)
            print('expected: ', samples['output'][i])

    def map_to_input(map, start, end):
        input = []
        for i in range(map):
            for j in range(map[0]):
                if (i, j) == start:
                    input.append(1)
                elif (i, j) == end:
                    input.append(1)
                elif map[i][j].blocked:
                    input.append(-1)
                else:
                    input.append(0)
        return input

    def map_to_result(map, start, end, path):
        result = []
        for i in range(map):
            for j in range(map[0]):
                if (i, j) == start:
                    result.append(1)
                elif (i, j) == end:
                    result.append(1)
                elif (i, j) in path:
                    result.append(1)
                elif map[i][j].blocked:
                    result.append(-1)
                else:
                    result.append(0)
        return result

    def output_to_chemin(output, map_width):  # besoin de l'entrée start/end
        # pour ordonné le print
        temp = []
        list_coord = []
        for i in range(len(output)):
            if output[i] == 1:
                temp.append(i)
        for k in temp:
            list_coord.append((k//map_width, k % map_width))  # on reforme les coordonnées avec la position
            # dans le vecteur ligne
        # On réordonne pour avoir le même format que le chemin sortie par astar

        def get_key(item):
            return item[0]  # Pour sort selon la première valeur
        sorted(list_coord, key=get_key)

        return list_coord


    def learn_path_format(network, samples, epochs=2500, lrate=.1, momentum=0.1):  # Version affichant des résultats
        #  lisibles plutôt qu'une énorme liste
        # Train
        for i in range(epochs):
            n = np.random.randint(samples.size)
            network.propagate_forward(samples['input'][n])
            network.propagate_backward(samples['output'][n], lrate, momentum)
        # Test
        for i in range(samples.size):
            o = network.propagate_forward(samples['input'][i])
            o_displayed = output_to_chemin(o, 100)   # WARNING: 100 = MAP_WIDTH!!!!!!!!!!!!!!!!!!!!!!!!!!!!
            print(i, samples['input'][i], o_displayed, )
            print('expected: ', output_to_chemin(samples['output'][i], 100))

    network = MLP(25, 50, 50, 25)
    samples = np.zeros(1, dtype=[('input',  float, 25), ('output', float, 25)])
    init = []
    result = []
    init.append([0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0])  # taille 5x5 de base
    result.append([0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0])

    samples[0] = init[0], result[0]
    # print(samples)

    learn_path(network, samples)
    # # Train
    # n = np.random.randint(samples.size)
    # network.propagate_forward(samples['input'][n])
    # network.propagate_backward(samples['output'][n], 0.1, 0.1)
    #
    # # Test
    # for i in range(samples.size):
    #     o = network.propagate_forward(samples['input'][i])
    #     print(i, samples['input'][i], o, )
    #     print('(expected )', samples['output'][i])

