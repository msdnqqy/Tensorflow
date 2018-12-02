"""
tensorflow 分类器
识别手写数字
发现问题：loss 返回nan
原因：在wx+b时候出现结果为负值，log（<0)=nan
"""

import numpy as np;
import tensorflow as tf;
from tensorflow.examples.tutorials.mnist import input_data

"""
添加神经网络层
"""
def add_layer(inputs,in_size,out_size,activation_function=None):
    weights=tf.Variable(tf.random_normal([in_size,out_size]))
    biases=tf.Variable(tf.zeros([1,out_size])+0.1)

    wx_plus_b=tf.matmul(inputs,weights)+biases

    if activation_function is None:
        output=wx_plus_b
    else:
        output=activation_function(wx_plus_b)

    return output

"""
计算准确度
"""
def compute_accuracy(v_xs, v_ys):
    global prediction
    y_pre = sess.run(prediction, feed_dict={xs: v_xs})
    correct_prediction = tf.equal(tf.argmax(y_pre,1), tf.argmax(v_ys,1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    result = sess.run(accuracy, feed_dict={xs: v_xs, ys: v_ys})
    return result

#获取训练数据
mnist=input_data.read_data_sets('MNIST_data',one_hot=True)

#真实数据 & 标签
xs=tf.placeholder(tf.float32,[None,784])
ys=tf.placeholder(tf.float32,[None,10])

#使用softmax作为激励函数
prediction=add_layer(xs,784,10,activation_function=tf.nn.softmax)

#这里使用ys×log（prediction）；log函数在底<1时候单调递增，在x<1时，log（x）<0
loss=tf.reduce_mean(-tf.reduce_sum(ys*tf.log(prediction),reduction_indices=[1]))
train_step=tf.train.AdagradOptimizer(0.1).minimize(loss)

init=tf.global_variables_initializer()
sess=tf.Session()
sess.run(init)

for i in range(1001):

    #SGD,每次拿100条数据训练
    batch_xs,batch_ys=mnist.train.next_batch(100)
    sess.run(train_step,feed_dict={xs:batch_xs,ys:batch_ys})
    
    #打印训练数据
    if i%50==0:
        print('------------------------step:',i)
        # print('pre:',sess.run(tf.log(prediction),feed_dict={xs:batch_xs,ys:batch_ys}))
        # print('sum:',sess.run(-tf.reduce_sum(ys*tf.log(prediction)),feed_dict={xs:batch_xs,ys:batch_ys}))
        print('loss:{0}'.format(sess.run(loss,feed_dict={xs:batch_xs,ys:batch_ys})))
        print(compute_accuracy(mnist.test.images, mnist.test.labels))