import numpy as np;
import tensorflow as tf;

"""
基础tensorflow的使用
"""
x_data=np.random.rand(100).astype(np.float32)
y_data=x_data*0.1+0.3

Weights=tf.Variable(tf.random_uniform([1],-1.0,1.0))
biase=tf.Variable(tf.zeros([1]))

y=Weights*x_data+biase

loss=tf.reduce_mean(tf.square(y-y_data))#定义损失函数
optimizer=tf.train.GradientDescentOptimizer(0.5)#定义优化器
train=optimizer.minimize(loss)#定义训练过程为减少loss

init=tf.global_variables_initializer()#初始化变量

sess=tf.Session()
sess.run(init)

for step in range(201):
    sess.run(train)
    if step%20==0:
        print(step,sess.run(Weights),sess.run(biase))


