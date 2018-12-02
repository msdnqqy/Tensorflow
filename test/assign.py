"""
tf assign用法
"""

import numpy as np;
import tensorflow as tf;

state=tf.Variable(0,name='state')
one=tf.constant(1)

new_value=tf.add(state,one)
update=tf.assign(state,new_value)

init=tf.global_variables_initializer()

sess=tf.Session()
sess.run(init)
for i in range(3):
    sess.run(update)
    print(sess.run(state))
