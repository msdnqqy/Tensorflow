"""

神经网络与深度学习.pdf
"""

import numpy as np;
import random
from tensorflow.examples.tutorials.mnist import input_data


class NetWork(object):
    def __init__(self,size=[2,3,1]):
        #初始化weights & biases
        #size[0]=输入层
        self.weights=np.array([np.random.randn(y,x) for x,y in zip(size[:-1],size[1:])])
        self.biases=np.array([np.random.randn(y,1) for y in size[1:]])

        self.num_layers=len(size)
        self.size=size


    """
    train_data,循环次数，批数大小，学习率，测试数据
    """
    def SGD(self,train_data,epochs,mini_batch_size,eta,test_data=None):
        for i in range(epochs):

            #生成结构数据
            x_data=[train_data[k:k+mini_batch_size] for k in range(0,len(train_data),mini_batch_size)]

            for xs in x_data:
                self.update_mini_batch(xs,eta)
            
            if test_data is not None:
                acc=self.evaluate(test_data)
                print('acc:{0}/{1}'.format(acc,len(test_data)))
            
            print('epochos:{i} complete')


    def update_mini_batch(self,train_data,lr):
        # nabla_b=np.zeros(self.biases.shape)
        # nabla_w=np.zeros(self.weights.shape)

        nabla_b=np.array([np.zeros(b.shape) for b in self.biases])
        nabla_w=np.array([np.zeros(w.shape) for w in self.weights])

        # 进行求导求梯度
        for (x,y) in train_data:
            delta_nabla_b,delta_nabla_w=self.backprop(x,y)
            nabla_b=[nb+dnb for nb,dnb in zip(nabla_b,delta_nabla_b)]
            nabla_w=[nw+dnw for nw,dnw in zip(nabla_w,delta_nabla_w)]


        # self.weights-=(lr/len(train_data))*nabla_w
        # self.biases-=(lr/len(train_data))*nabla_b
        self.weights=[w-(lr/len(train_data))*nw for w,nw in zip(self.weights,nabla_w)]
        self.biases=[b-(lr/len(train_data))*nb for b,nb in zip(self.biases,nabla_b)]


    #求梯度
    def backprop(self,x,y):
        # nabla_b,nabla_w=np.zeros(self.biases.shape),np.zeros(self.weights.shape)
        nabla_b=np.array([np.zeros(b.shape) for b in self.biases])
        nabla_w=np.array([np.zeros(w.shape) for w in self.weights])

        activation=x;
        activations=[x]
        zs=[]
        for b,w in zip(self.biases,self.weights):
            z=np.dot(w,activation)+b
            zs.append(z)
            activation=self.sigmoid(z)
            activations.append(activation)

        
        delta=self.cost_derivative(activations[-1],y)*self.sigmoid_prime(zs[-1])
        nabla_b[-1]=delta
        nabla_w[-1]=np.dot(delta,activations[-2].transpose())

        # 对每一层求导
        for l in range(2,self.num_layers):
            z=zs[-l]
            sp=self.sigmoid_prime(z)
            delta=np.dot(self.weights[-l+1].transpose(),delta)*sp
            nabla_b[-l]=delta
            nabla_w[-l]=np.dot(delta,activations[-l-1].transpose())

        return (nabla_b,nabla_w)

    #损失函数
    def cost_derivative(self,output_activations,y):
        return (output_activations-y)

    def sigmoid(self,z):
        return 1.0/(1.0+np.exp(-z))

    def sigmoid_prime(self,z):
        return self.sigmoid(z)*(1-self.sigmoid(z))

    def evaluate(self,test_data):
        test_results=[(np.argmax(self.feedforward(x)),np.argmax(y)) for (x,y) in test_data]

        return sum(int(x==y) for (x,y) in test_results)

    def feedforward(self,a):
        for w,b in zip(self.weights,self.biases):
            a=self.sigmoid(np.dot(w,a)+b)
        return a


if __name__ == "__main__":

    mnist=input_data.read_data_sets('./mnist',one_hot=True)
    train_data=[]
    test_data=[]
    for i in range(6000):
        train_data.append((mnist.train.images[i][:,np.newaxis],mnist.train.labels[i][:,np.newaxis]))
        test_data.append((mnist.test.images[i][:,np.newaxis],mnist.test.labels[i][:,np.newaxis]))


    train_data=np.array(train_data)
    test_data=np.array(test_data)


    net=NetWork([784,30,10])
    net.SGD(train_data,30,10,0.5,test_data=test_data)

