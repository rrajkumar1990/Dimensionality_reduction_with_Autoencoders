# -*- coding: utf-8 -*-
"""
Created on Fri Jun 21 12:17:58 2019

@author: rajkumar.rajasekaran
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs


#create a random classification dataset with 200 sample records and 3 features and 2 labels(centers)
data= make_blobs(n_samples=200, n_features=3, centers=2, random_state=101)

#assigning the features 
train =data[0]

#lets scale the input features using MImmaxscaler
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(X= train) 

data_x = scaled_data[:,0]
data_y = scaled_data[:,1]
data_z = scaled_data[:,2]

#to visuvalise lets open and see the 3d plot of input data as we have three features
from mpl_toolkits.mplot3d import Axes3D
fig = plt.figure()
ax = fig.add_subplot(111,projection='3d')
ax.scatter(data_x,data_y,data_z,c=data[1])
plt.show()




#lets start with our Linear autoencoder for dimensionality reduction
import tensorflow as tf
from tensorflow.contrib.layers import fully_connected


num_inputs = 3  # 3 features
num_hidden = 2  # 2 reducing the three features to 2 in the hidden layer so 2 neurons
num_outputs = num_inputs # Must be true for an autoencoder!

learning_rate = 0.01

#placeholders
X= tf.placeholder(tf.float32,shape=[None, num_inputs])

hidden_layer = fully_connected(X,num_hidden,activation_fn=None) # the activation funxtions are none as this is a Linear autoenocoders
output_layer=  fully_connected(hidden_layer,num_outputs,activation_fn=None)# the activation funxtions are none as this is a Linear autoenocoders

loss = tf.reduce_mean(tf.square(output_layer -X))


#using adamoptimizer as we adam  uses weighed average of weights its always good to use Adam
optimizer = tf.train.AdamOptimizer(learning_rate)
train  = optimizer.minimize( loss)


init = tf.global_variables_initializer()


#lets start our session
with tf.Session() as sess :
    sess.run(init)
    for  i in range(1000):
        sess.run(train,feed_dict={X:scaled_data})
    result =hidden_layer.eval(feed_dict={X:scaled_data})



#lets print the shape of our result variable where we have stored the hidden neuorns values
print(result.shape)
#200,2


#visualising the reduced features
plt.scatter(result[:,0],result[:,1],c=data[1])
plt.show()




                  
