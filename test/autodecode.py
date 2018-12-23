import tensorflow as tf
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
import matplotlib.pyplot as plt

batch_size=64
lr=0.002
n_test_img=5

mnist=input_data.read_data_sets('./mnist',one_hot=True)

tf_x=tf.placeholder(tf.float32,[None,28*28])

#定义encoder层
en0=tf.layers.dense(tf_x,128,tf.nn.tanh)
en1=tf.layers.dense(en0,64,tf.nn.tanh)
en2=tf.layers.dense(en1,12,tf.nn.tanh)
encoded=tf.layers.dense(en2,3)

#定义decoder
de0=tf.layers.dense(encoded,12,tf.nn.tanh)
de1=tf.layers.dense(de0,64,tf.nn.tanh)
de2=tf.layers.dense(de1,128,tf.nn.tanh)
decoder=tf.layers.dense(de2,28*28,tf.nn.sigmoid)

loss=tf.losses.mean_squared_error(labels=tf_x,predictions=decoder)
train=tf.train.AdamOptimizer(lr).minimize(loss)

view_data=mnist.test.images[:n_test_img]
f,a=plt.subplots(2,n_test_img,figsize=(5,2))
plt.ion()
for i in range(n_test_img):
    a[0][i].imshow(np.reshape(view_data[i],(28,28)),cmap='gray')
    a[0][i].set_xticks(());a[0][i].set_yticks(())



sess=tf.Session()
sess.run(tf.global_variables_initializer())
for step in range(8000):
    b_x,b_y=mnist.train.next_batch(batch_size)
    _,encode,decode,loss_=sess.run([train,encoded,decoder,loss],{tf_x:b_x})

    if step%100==0:
        print('loss:{0}'.format(loss_))
        decode_=sess.run(decoder,{tf_x:view_data})
        for i in range(n_test_img):
            a[1][i].clear()
            a[1][i].imshow(np.reshape(decode_[i],(28,28)),cmap='gray')
            a[1][i].set_xticks(())
            a[1][i].set_yticks(())

        plt.draw();plt.pause(0.01);

plt.ioff();
plt.show()
