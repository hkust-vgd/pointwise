import tensorflow as tf
import numpy as np
import datetime
import json
from collections import namedtuple

def parse_arguments(json_file):
    with open(json_file) as f:
        arg = json.load(f)
    return arg

def tic():
    global _start_time
    _start_time = datetime.datetime.now()

def toc():
    _stop_time = datetime.datetime.now()
    ms = (_stop_time - _start_time).total_seconds() * 1000
    (t_sec, t_msec) = divmod(ms, 1000)
    (t_min, t_sec) = divmod(t_sec, 60)
    (t_hour, t_min) = divmod(t_min, 60)
    print('Elapsed: {} h : {} m : {} s : {} ms'.format(t_hour, t_min, t_sec, t_msec))

def leaky_relu(x, alpha=0.1, name="LeakyReLU"):
    """ LeakyReLU.
    Modified version of ReLU, introducing a nonzero gradient for negative
    input.
    Arguments:
    	x: A `Tensor` with type `float`, `double`, `int32`, `int64`, `uint8`,
    		`int16`, or `int8`.
    	alpha: `float`. slope.
    	name: A name for this activation op (optional).
    Returns:
    	A `Tensor` with the same type as `x`.
    References:
    	Rectifier Nonlinearities Improve Neural Network Acoustic Models,
    	Maas et al. (2013).
    Links:
    	[http://web.stanford.edu/~awni/papers/relu_hybrid_icml2013_final.pdf]
    	(http://web.stanford.edu/~awni/papers/relu_hybrid_icml2013_final.pdf)
    """
    # If incoming Tensor has a scope, this op is defined inside it
    i_scope = ""
    if hasattr(x, 'scope'):
        if x.scope:
            i_scope = x.scope
    with tf.name_scope(i_scope + name) as scope:
        x = tf.nn.relu(x)
        m_x = tf.nn.relu(-x)
        x -= alpha * m_x

    x.scope = scope
    return x

def sort_point_cloud_xyz(batch_data):
    """ Sort the point clouse base on coordinate with piority x -> y -> z
        Input:
          BxNxK array, original batch of point clouds. First three channels are XYZ. 
        Return:
          BxNxK array, sorted batch of point clouds
    """
    sorted_data = np.zeros(batch_data.shape, dtype=np.float32)
    num_channels = batch_data.shape[2]
    for k in range(batch_data.shape[0]):
        shape_pc = batch_data[k, ...]
        shape_pc = shape_pc.reshape((-1, num_channels))
        shape_pc = shape_pc[shape_pc[:,2].argsort()] # sort the least significant field first.
        shape_pc = shape_pc[shape_pc[:,1].argsort(kind='mergesort')] # from now we need to use a stable sort algorithm
        shape_pc = shape_pc[shape_pc[:,0].argsort(kind='mergesort')]
        sorted_data[k, ...] = shape_pc
        #print("sorted data: ")
        #print(sorted_data)
    return sorted_data

def sort_point_cloud_xyz2(batch_data, batch_attributes):
    """ Sort the point clouse base on coordinate with piority x -> y -> z. 
        Also change the order of the attributes of each point accordingly.
        Input:
          BxNxK array, original batch of point clouds. First three channels are XYZ. 
          BxNxM array, original batch of point attributes.
        Return:
          BxNxK array, sorted batch of point clouds
          BxNxM array, sorted batch of point attributes
    """
    sorted_data = np.zeros(batch_data.shape, dtype=batch_data.dtype)
    sorted_attr = np.zeros(batch_attributes.shape, dtype=batch_attributes.dtype)
    
    for k in range(batch_data.shape[0]):
        shape_pc = batch_data[k, ...]
        shape_attr = batch_attributes[k, ...]
    
        # sort the least significant field first 
        idx = shape_pc[:,2].argsort()  # return index
        shape_pc = shape_pc[idx]  # swap rows
        shape_attr = shape_attr[idx]

        # from now we need to use a stable sort algorithm
        idx = shape_pc[:,1].argsort(kind='mergesort')        
        shape_pc = shape_pc[idx] 
        shape_attr = shape_attr[idx]

        idx = shape_pc[:,0].argsort(kind='mergesort')
        shape_pc = shape_pc[idx]
        shape_attr = shape_attr[idx]

        sorted_data[k, ...] = shape_pc
        sorted_attr[k, ...] = shape_attr        

    return sorted_data, sorted_attr
