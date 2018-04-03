import tensorflow as tf

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
