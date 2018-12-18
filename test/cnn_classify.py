"""
RNN分类器
"""

import tensorflow as tf;
import numpy as np;
import pandas as pd;

batch_size=64
time_step=28
input_size=28
lr=0.01

from tensorflow.examples.tutorials.mnist import input_data
import matplotlib.pyplot as plt;

mnist=input_data.read_data_sets('./mnist',one_hot=True)
print('mnist.train.images[0].shape:',mnist.train.images[0].shape,'\nmnist.train.labels[0].shape:',mnist.train.labels[0].shape)

tf_x=tf.placeholder(tf.float32,[None,time_step*input_size])
image=tf.reshape(tf_x,[-1,time_step,input_size])

tf_y=tf.placeholder(tf.float32,[None,10])

#定义rnn
rnn_cell=tf.contrib.rnn.BasicLSTMCell(num_units=64)
outputs,(h_c,h_n)=tf.nn.dynamic_rnn(rnn_cell,image,
        initial_state=None,dtype=tf.float32,time_major=False)


output=tf.layers.dense(outputs[:,-1,:],10)

loss=tf.losses.softmax_cross_entropy(onehot_labels=tf_y,
                                logits=output)
train_step=tf.train.AdamOptimizer(lr).minimize(loss)

acc=tf.metrics.accuracy(labels=tf.argmax(tf_y,axis=1),
                predictions=tf.argmax(output,axis=1))[1]

sess=tf.Session()
sess.run([tf.global_variables_initializer(),
        tf.local_variables_initializer()])

for i in range(1201):
    b_x,b_y=mnist.train.next_batch(batch_size)
    _,loss_=sess.run([train_step,loss],{tf_x:b_x,tf_y:b_y})

    if i%50==0:
        acc_=sess.run([acc],{tf_x:mnist.test.images[0:2000],tf_y:mnist.test.labels[0:2000]})
        print('i:{0},loss:{1},acc:{2}'.format(i,loss_,acc_))