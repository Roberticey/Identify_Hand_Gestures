import cv2
import time
import scipy
import os
import math
import numpy as np
import h5py
import matplotlib.pyplot as plt
import scipy
from PIL import Image
from scipy import ndimage
import tensorflow as tf
from tensorflow.python.framework import ops
from cnn_utils import *
import webbrowser


def create_placeholders(n_H0, n_W0, n_C0, n_y):
    """
    Creates the placeholders for the tensorflow session.

    Arguments:
    n_H0 -- scalar, height of an input image
    n_W0 -- scalar, width of an input image
    n_C0 -- scalar, number of channels of the input
    n_y -- scalar, number of classes

    Returns:
    X -- placeholder for the data input, of shape [None, n_H0, n_W0, n_C0] and dtype "float"
    Y -- placeholder for the input labels, of shape [None, n_y] and dtype "float"
    """

    ### START CODE HERE ### (≈2 lines)
    X = tf.placeholder(tf.float32, [None, n_H0, n_W0, n_C0])
    Y = tf.placeholder(tf.float32, [None, n_y])
    ### END CODE HERE ###

    return X, Y


def forward_propagation(X, parameters):
    """
    Implements the forward propagation for the model:
    CONV2D -> RELU -> MAXPOOL -> CONV2D -> RELU -> MAXPOOL -> FLATTEN -> FULLYCONNECTED

    Note that for simplicity and grading purposes, we'll hard-code some values
    such as the stride and kernel (filter) sizes.
    Normally, functions should take these values as function parameters.

    Arguments:
    X -- input dataset placeholder, of shape (input size, number of examples)
    parameters -- python dictionary containing your parameters "W1", "W2"
                  the shapes are given in initialize_parameters

    Returns:
    Z3 -- the output of the last LINEAR unit
    """

    # Retrieve the parameters from the dictionary "parameters"
    W1 = parameters['W1']
    W2 = parameters['W2']

    ### START CODE HERE ###
    # CONV2D: stride of 1, padding 'SAME'
    Z1 = tf.nn.conv2d(X, W1, strides=[1, 1, 1, 1], padding='SAME')
    # RELU
    A1 = tf.nn.relu(Z1)
    # MAXPOOL: window 8x8, stride 8, padding 'SAME'
    P1 = tf.nn.max_pool(A1, ksize=[1, 8, 8, 1], strides=[1, 8, 8, 1], padding='SAME')
    # CONV2D: filters W2, stride 1, padding 'SAME'
    Z2 = tf.nn.conv2d(P1, W2, strides=[1, 1, 1, 1], padding='SAME')
    # RELU
    A2 = tf.nn.relu(Z2)
    # MAXPOOL: window 4x4, stride 4, padding 'SAME'
    P2 = tf.nn.max_pool(A2, ksize=[1, 4, 4, 1], strides=[1, 4, 4, 1], padding='SAME')
    # FLATTEN
    F = tf.contrib.layers.flatten(P2)
    # FULLY-CONNECTED without non-linear activation function (not not call softmax).
    # 6 neurons in output layer. Hint: one of the arguments should be "activation_fn=None"
    Z3 = tf.contrib.layers.fully_connected(F, 6, activation_fn=None)
    ### END CODE HERE ###

    return Z3



model_saver = tf.train.Saver()
with tf.Session(graph=graph_cnn) as sess:
    model_saver.restore(sess, "saved_models/CNN_New.ckpt")
    print("Model restored.")
    print('Initialized')
    graph = tf.get_default_graph()
    parameters = graph.get_tensor_by_name("parameters")

    capture = cv2.VideoCapture(0)
    pred = []
    while (True):
        ret, frame = capture.read()
        b = cv2.resize(frame, (64, 64), fx=0, fy=0, interpolation=cv2.INTER_CUBIC)
        cv2.imshow('video', b)
        X, Y = create_placeholders(64, 64, 3, 6)
        Z3 = forward_propagation(X, parameters)
        prediction = sess.run(tf.argmax(Z3, 1), feed_dict={X: [b]})
        pred += [np.squeeze(prediction)]

        if cv2.waitKey(1) == 27:
            break

        time.sleep(0.5)

        if len(pred)==10:
            do = max(pred,key=pred.count)
            if do == 0:
                os.system('shutdown -s') # shut-down
            elif do == 1:
                os.system("shutdown /r /t 1") # restart
            elif do == 2:
                os.system("shutdown -l")  #log-out
            elif do == 3:
                exit()    #exit program
            elif do == 4:
                beep = lambda x: os.system("echo -n '\a';sleep 0.2;" * x)
                beep(10)
                # make a beeeping sound
            else:
                url = "http://www.google.com/"
                chrome_path = '/usr/bin/google-chrome %s'
                webbrowser.get(chrome_path).open(url)
                # open google

    capture.release()
    cv2.destroyAllWindows()