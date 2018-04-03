import tensorflow as tf
import numpy as np
import selu
import util

import sys
import os
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
conv3p_module = tf.load_op_library(BASE_DIR + '/tf_ops/conv3p/tf_conv3p.so')

def conv3p(points_tensor, input_tensor, kernel_tensor, stride_tensor, voxel_size):
    return conv3p_module.conv3p(points_tensor, input_tensor, kernel_tensor, stride_tensor, voxel_size);

@tf.RegisterGradient('Conv3p')
def _conv3p_grad(op, grad_from_next_layer):
    """The derivatives for convolution.
    Args:
        op: the convolution op.
        grad_from_next_layer: the tensor representing the gradient w.r.t. the output
    Returns:
        the gradients w.r.t. the point tensor, input tensor, and the filter
    """
    points = op.inputs[0]
    input = op.inputs[1]
    filter = op.inputs[2]
    stride = op.inputs[3]
    voxel_size = op.inputs[4]

    input_grad, filter_grad = conv3p_module.conv3p_grad(grad_from_next_layer, points, input, filter, stride, voxel_size)
    return [None, input_grad, filter_grad, None, None]

def layer(idx, points_tensor, input_tensor, voxel_size, filter_shape, stride_shape, is_training=True):
    filter_tensor = tf.get_variable("filter{}".format(idx), filter_shape)
    stride_tensor = tf.constant(stride_shape)
    conv = conv3p(points_tensor, input_tensor, filter_tensor, stride_tensor, voxel_size);
    relu = selu.selu(conv)
    return relu 
    
class PointConvNet:
    def __init__(self, num_class):
        self.num_class = num_class

    def model(self, points_tensor, input_tensor, is_training=True):
        """
        Arguments:
            points_tensor: [b, n, 3] point cloud
            input_tensor: [b, n, channels] extra data defined for each point
        """        
        in_channels = input_tensor.get_shape()[2].value

        voxel_size = tf.constant([0.1])
        net1 = layer(1, points_tensor, input_tensor, voxel_size, [3, 3, 3, in_channels, 9], [1, 1, 1])
        net2 = layer(2, points_tensor, net1, voxel_size, [3, 3, 3, 9, 9], [2, 2, 2])
        net3 = layer(3, points_tensor, net2, voxel_size, [3, 3, 3, 9, 9], [3, 3, 3])
        net4 = layer(4, points_tensor, net3, voxel_size, [3, 3, 3, 9, 9], [4, 4, 4])
        concat = tf.concat([net1, net2, net3, net4], axis=2)
        net = layer(5, points_tensor, concat, voxel_size, [3, 3, 3, 36, self.num_class], [1, 1, 1])
        return net

    def loss(self, logits, labels):
        """
        Arguments:
            logits: prediction with shape [batch_size, num_class]
            labels: ground truth scalar labels with shape [batch_size]
        """
        onehot_labels = tf.one_hot(labels, depth=self.num_class)
        e = tf.losses.softmax_cross_entropy(onehot_labels, logits)

        #e = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=labels)
        #e = tf.reduce_mean(e)
        return e
