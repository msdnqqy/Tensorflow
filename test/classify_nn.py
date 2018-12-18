"""
tf官方神经网络层
用回归做例子
需要做归一化，避免nn问题的发生
"""

import tensorflow as tf;
import numpy as np;
import pandas as pd;
import matplotlib.pyplot as plt;

x_data=np.linspace(-1,1,200)[:,np.newaxis]
noise=np.random.normal(0,0.1,size=x_data.shape)
y_data=np.power(x_data,2)+noise

#创建神经网络输入+标签的placeholder
xs=tf.placeholder(tf.float32,[None,1])
ys=tf.placeholder(tf.float32,[None,1])

#创建神经网络层
l1=tf.layers.dense(xs,10,tf.nn.tanh)
prediction=tf.layers.dense(l1,1)

#设置损失函数和训练过程
loss=tf.losses.mean_squared_error(ys,prediction)
optimizer=tf.train.GradientDescentOptimizer(0.5)
train_step=optimizer.minimize(loss)

init=tf.global_variables_initializer()
sess=tf.Session()
sess.run(init)

#画出原始数据图形
plt.scatter(x_data,y_data)
plt.ion()
plt.show()

for i in range(800):
    _,lossing,pred,l1test=sess.run([train_step,loss,prediction,l1],{xs:x_data,ys:y_data})
    if i%20==0:
        print('lossing:',lossing)
        plt.cla()
        plt.scatter(x_data,y_data)
        plt.plot(x_data,pred,'r-',lw=5)
        plt.pause(0.1)            


plt.ioff()
plt.show()