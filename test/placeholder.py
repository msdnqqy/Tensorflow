"""
placeholder传值
1，placeholder（type，like）
2.feed_dict{name：value...}
"""
import tensorflow as tf;
import numpy as np;

input1=tf.placeholder(tf.float32)
input2=tf.placeholder(tf.float32)

mul=tf.multiply(input1,input2)#tf.mul被废弃，已经使用multiply代替

with tf.Session() as sess:
    print(sess.run(mul,feed_dict={input1:[4.],input2:[5.]}))