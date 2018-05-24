#!/usr/bin/python3
# -*- coding:utf-8 -*-
# author: zhaofeng-shu33
import tensorflow as tf
import numpy as np
import pdb
if __name__ == '__main__':
    X = np.array([0,0,1,1,0,1,1,0], dtype=np.float32).reshape(4,2)
    Y = np.array([0,0,1,1], dtype = np.float32)


    w1 = tf.Variable(tf.constant([[-0.62,1.08],[1.17, 0.41]]))
    b1 = tf.Variable(tf.zeros([2]))
    w2 = tf.Variable(tf.constant([[1.22], [-1.11]]))
    b2 = tf.Variable(tf.zeros([1]))

    out1 = tf.tanh(tf.add(tf.matmul(X, w1), b1))

    out2 = tf.sigmoid(tf.add(tf.matmul(out1, w2), b2)) 
    error = tf.scalar_mul(-1.0, tf.tensordot(Y, tf.log(out2), axes = 1) + tf.tensordot(tf.constant(1.0) - Y,  tf.log(1.0 - out2), axes = 1))
    cross_entropy_loss = tf.reduce_mean(error/4)

    gd = tf.train.GradientDescentOptimizer(0.01)
    train = gd.minimize(cross_entropy_loss)



    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    print(sess.run(cross_entropy_loss))
    print(sess.run(gd.compute_gradients(cross_entropy_loss)))
    for i in range(10):
        _, loss,predict = sess.run([train,cross_entropy_loss,out2])
    acc = ((predict.reshape(4) > 0.5) == Y).mean()    
    print([loss, acc])






