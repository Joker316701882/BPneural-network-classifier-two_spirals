# -*- coding: utf-8 -*-
"""
Created on Wed Nov 09 10:42:41 2016

@author: Joker
"""

import numpy as np
import data
import matplotlib.pyplot as plt

dataset,label = data.create_dataset()

class Network(object):

    def __init__(self, m,epochs):
        self.epochs = epochs
        self.sizes = (2,m,1)
        self.num_layers = 3
        self.biases_l1 = np.random.normal(loc=0,scale = np.sqrt(1.0/2.0),size = (m,1))
        self.biases_l2 = np.random.normal(loc=0,scale = np.sqrt(1.0/2.0),size = (1,1))
        self.weights_l1 = np.random.normal(loc=0,scale = np.sqrt(1.0/2.0),size = (m,2))
        self.weights_l2 = np.random.normal(loc=0,scale = np.sqrt(1.0/2.0),size = (1,m))
        self.x = []
        self.y = []
        
    def feedforward(self, x):
        """Return the output of the network if ``a`` is input."""
        a1 = tanh(np.dot(self.weights_l1,x.reshape(2,1)) + self.biases_l1)
        a2 = sigmoid(np.dot(self.weights_l2,a1) + self.biases_l2)
        return a2
    
    def GD(self,dataset,label,eta):
        for i in range(self.epochs):
            gd_weights_l1 = np.zeros(self.weights_l1.shape)
            gd_weights_l2 = np.zeros(self.weights_l2.shape)
            gd_biases_l1 = np.zeros(self.biases_l1.shape)
            gd_biases_l2 = np.zeros(self.biases_l2.shape)
            for x,y in zip(dataset,label):
                delta_b_l1,delta_w_l1,delta_b_l2,delta_w_l2 = self.backprop(x,y)
                gd_biases_l1 = gd_biases_l1 + delta_b_l1
                gd_biases_l2 = gd_biases_l2 + delta_b_l2
                gd_weights_l1 = gd_weights_l1 + delta_w_l1
                gd_weights_l2 = gd_weights_l2 + delta_w_l2
#            print 'w1:',gd_weights_l1
#            print 'w2:',gd_weights_l2
#            print 'b1:',gd_biases_l1
#            print 'b2:',gd_biases_l2
            self.weights_l1 = self.weights_l1-(float(eta)/len(dataset))*gd_weights_l1
            self.weights_l2 = self.weights_l2-(float(eta)/len(dataset))*gd_weights_l2
            self.biases_l1 = self.biases_l1-(float(eta)/len(dataset))*gd_biases_l1
            self.biases_l2 = self.biases_l2-(float(eta)/len(dataset))*gd_biases_l2
            print 'time i:',i
            self.x.append(i)
            temp = self.accuracy(dataset,label)
            self.y.append(temp)
            print 'the accuracy is :',temp       

    def backprop(self,x,y):
        x = x.reshape(2,1)
        y = y.reshape(1,1)
        activation = x
        activations = [x]
        zs = []
            #calculate hidden layer
        z = (np.dot(self.weights_l1,activation))+self.biases_l1   
        zs.append(z)
        activation = tanh(z)
        activations.append(activation)
            #calculate output layer
        z = np.dot(self.weights_l2,activation)+self.biases_l2
        zs.append(z)
        activation = sigmoid(z)
        activations.append(activation)
        #backpropgation
            #calculate output layer
        delta_l2 = self.derivative(activations[2], y)  #delta_l2.shape = (1,1)
        nabla_b_l2 = delta_l2
        nabla_w_l2 = (activations[1]*delta_l2).transpose()   #nabla_w_l2.shape = (16,1)
            #calculate hidden layer
        tp = tanh_prime(zs[0])   #tp.shape = (16,1)
        delta_l1 = np.dot(self.weights_l2.transpose() , delta_l2) * tp
        nabla_b_l1 = delta_l1
        nabla_w_l1 = np.dot(delta_l1,activations[0].transpose())
        return nabla_b_l1,nabla_w_l1,nabla_b_l2,nabla_w_l2
    
    def derivative(self, a, y):
        return a-y

    def accuracy(self,dataset,label):
        result = []
        for i in range(len(dataset)):
#            print self.feedforward(dataset[i]),'  ',label[i]
            a = 0 if self.feedforward(dataset[i])<0.5 else 1
            y = label[i].reshape(1,1)
            if a == y:
                result.append(1)
            else:
                result.append(0)
        accuracy = float(sum(result))/float(len(result))
        return accuracy
    
    
    def plot(self):
        plt.plot(self.x,self.y)
        plt.show()
    
def sigmoid(z):
    return 1.0/(1.0+np.exp(-z))

def sigmoid_prime(z):
    return sigmoid(z)*(1-sigmoid(z))

def tanh(z):
    return (np.exp(z)-np.exp(-z))/(np.exp(z)+np.exp(-z))

def tanh_prime(z):
    return (1-tanh(z)*tanh(z))