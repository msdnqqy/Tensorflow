"""
rnn回归
"""

import tensorflow as tf;
import numpy as np;
import matplotlib.pyplot as plt;

time_step=10
input_size=1
cell_size=32
lr=0.02

steps=np.linspace(0,2*np.pi,100,dtype=np.float32)
x_np=np.sin(steps)
y_np=np.cos(steps)

plt.plot(steps,y_np,'r-',label='target(cos)')
plt.plot(steps,x_np,'b-',label='input(sin)')

plt.legend(loc='best')
plt.show()

#定义rnn的输入输出
tf_x=tf.placeholder(tf.float32,[None,time_step,input_size])
tf_y=tf.placeholder(tf.float32,[None,time_step,input_size])

rnn_cell=tf.contrib.rnn.BasicRNNCell(num_units=cell_size)
init_s=rnn_cell.zero_state(batch_size=1,dtype=tf.float32)

outpus,final_s=tf.nn.dynamic_rnn(rnn_cell,tf_x,initial_state=init_s,time_major=False)

#定义全连接层
outs2D=tf.reshape(outpus,[-1,cell_size])
net_outs2D=tf.layers.dense(outs2D,input_size)

outs=tf.reshape(net_outs2D,[-1,time_step,input_size])

loss=tf.losses.mean_squared_error(labels=tf_y,predictions=outs)
train_op=tf.train.AdamOptimizer(lr).minimize(loss)

plt.figure(1,figsize=(12,5))
plt.ion()

sess=tf.Session()
sess.run(tf.global_variables_initializer())

for step in range(60):
    start,end=step*np.pi,(step+1)*np.pi

    steps=np.linspace(start,end,time_step)
    x=np.sin(steps)[np.newaxis,:,np.newaxis]
    y=np.cos(steps)[np.newaxis,:,np.newaxis]
    if 'final_s_' not in globals():
        feed_dict={tf_x:x,tf_y:y}
    else:
        feed_dict={tf_x:x,tf_y:y,init_s:final_s_}

    _,pred_,final_s_=sess.run([train_op,outs,final_s],feed_dict)

    plt.plot(steps,y.flatten(),'r-')
    plt.plot(steps,pred_.flatten(),'b-')
    plt.pause(0.05)


plt.ioff()
plt.show()