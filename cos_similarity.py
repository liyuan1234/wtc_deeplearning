#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec 14 14:49:29 2018

@author: liyuan
"""

import tensorflow as tf

tf.reset_default_graph()

v1 = tf.placeholder(tf.float32,shape = [None,100])
v2 = tf.placeholder(tf.float32,shape = [None,100])

newv1 = v1/tf.norm(v1,axis = 1,keepdims = True)
newv2 = v2/tf.norm(v2,axis = 1,keepdims = True)

norm = tf.norm(v1,axis = 1, keepdims = True)

cos_d = tf.losses.cosine_distance(newv1,newv2,axis = 1)
print(cos_d)
x = np.ones([5,100])*np.array([[1],[2],[3],[4],[5]])
y = np.ones([5,100])*5

sess = tf.Session()

out = sess.run([cos_d], feed_dict = {v1:x,v2:y})
print(out)







