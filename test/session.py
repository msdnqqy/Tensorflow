"""
tensorflow session会话控制
"""

import tensorflow as tf;
import numpy as np;

matrix1=tf.constant(np.random.randn(1,2).astype(np.float32))
matrix2=tf.constant(np.random.randint(10,size=(2,3)).astype(np.float32))
matrix=tf.matmul(matrix1,matrix2)

sess=tf.Session()
result1=sess.run(matrix)
print(result1)

with tf.Session() as sess:
    result2=sess.run(matrix)
    print(result2)