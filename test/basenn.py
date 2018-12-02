import tensorflow as tf;
import numpy as np;
import matplotlib.pyplot as plt

"""
添加一个神经网络层
input=输入
in_size=输入属性数
out_size=神经元个数
activation_function=激活函数
"""
def add_layer(input,in_size,out_size,activation_function):
    Weights=tf.Variable(tf.random_normal([in_size,out_size]))
    biases=tf.Variable(tf.zeros([1,out_size])+0.1)

    Wx_plus_b=tf.matmul(input,Weights)+biases

    if activation_function is None:
        output=Wx_plus_b
    else:
        output=activation_function(Wx_plus_b)

    return output

x_data=np.linspace(-1,1,300)[:,np.newaxis]
noise=np.random.normal(0,0.05,x_data.shape)
y_data=np.square(x_data)+noise-0.5

fig=plt.figure()
ax=fig.add_subplot(1,1,1)
ax.scatter(x_data,y_data)
plt.ion()
plt.show()

xs=tf.placeholder(tf.float32,[None,1])
ys=tf.placeholder(tf.float32,[None,1])
l1=add_layer(xs,1,10,tf.nn.relu)
prediction=add_layer(l1,10,1,None)

loss=tf.reduce_mean(tf.reduce_sum(np.square(ys-prediction),reduction_indices=[1]))
train_step=tf.train.GradientDescentOptimizer(0.1).minimize(loss)

init=tf.global_variables_initializer()
sess=tf.Session()
sess.run(init)

for i in range(2000):
    sess.run(train_step,feed_dict={xs:x_data,ys:y_data})
    if i%50==0:
        print(sess.run(loss,feed_dict={xs:x_data,ys:y_data}))
        predict=sess.run(prediction,feed_dict={xs:x_data,ys:y_data})

        try:
            ax.lines.remove(lines[0])
        except Exception:
            pass
        lines=ax.plot(x_data,predict,'r-',lw=5)
        plt.pause(0.1)