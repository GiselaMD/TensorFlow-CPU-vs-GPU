#!/usr/bin/env python

import sys
import time
import numpy as np
import tensorflow as tf
import os

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID" 

LOGDIR = '/tmp/matmul/'
N = 10
config = tf.ConfigProto(allow_soft_placement = True, log_device_placement = True) 

with tf.device('/cpu:0'): # CHANGE DEVICE HERE
    A = tf.Variable(tf.random_normal((N*1024,N*1024)), name="A")
    B = tf.Variable(tf.random_normal((N*1024,N*1024)), name="B")
    C = tf.matmul(A, B)
    tf.summary.histogram("matriz", A)   #if you want to see the tensors with tensorboard, 
    tf.summary.histogram("matriz", B)   #run: tensorboard --logdir /"LOGDIR path"
    tf.summary.histogram("matriz", C)   


with tf.Session(config=config) as s:
    s.run(tf.global_variables_initializer())
    writer = tf.summary.FileWriter(LOGDIR)  #tensorboard
    writer.add_graph(s.graph)               #tensorboard
    t0 = time.time()
    s.run(C)
    t1 = time.time()
    print(t1-t0)
    

    tmin = 100
    for _ in range(10):
        t0 = time.time()
        s.run(c)
        t1 = time.time()
        tmin = min(t1-t0, tmin)

print(tmin)

