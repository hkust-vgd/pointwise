import tensorflow as tf
from tensorflow.python.framework import ops
import sys
import os
import numpy as np
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)
print(BASE_DIR)

conv3p_module = tf.load_op_library('./tf_conv3p.so')

def conv3p(points_tensor, input_tensor, kernel_tensor, stride_tensor, voxel_size):
    return conv3p_module.conv3p(points_tensor, input_tensor, kernel_tensor, stride_tensor, voxel_size)

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

class Conv3pTest(tf.test.TestCase):
    def test(self):
        pass

    def test_grad(self):
        import time
        np.random.seed(100)
        b = 6
        n = 16
        points = np.random.random((b, n ,3)).astype('float64') * 0.05 - 0.025
        input = np.random.random((b, n, 8)).astype('float64') * 0.05 - 0.025
        kernel = np.random.random((3,3,3,8,2)).astype('float64') * 0.05 - 0.025
        stride = np.array([2, 2, 2])
        voxel_size = np.array([0.05])

        # GPU or CPU test
        force_gpu=True

        with self.test_session(force_gpu=force_gpu):
            points_tensor = tf.constant(points)
            input_tensor = tf.constant(input)
            kernel_tensor = tf.constant(kernel)
            stride_tensor = tf.constant(stride, dtype='int32')
            voxel_size_tensor = tf.constant(voxel_size, dtype='float64')
            output_tensor = conv3p(points_tensor, input_tensor, kernel_tensor, stride_tensor, voxel_size_tensor)

            print("---- Test gradient with respect to input -------")
            err = tf.test.compute_gradient_error(input_tensor, (b, n, 8), output_tensor, (b, n, 2))
            print(err)
            self.assertLess(err, 1e-3)

            print("---- Test gradient with respect to filter -------")
            err = tf.test.compute_gradient_error(kernel_tensor, (3,3,3,8,2), output_tensor, (b, n, 2))
            print(err)
            self.assertLess(err, 1e-3)

if __name__=='__main__':
    tf.test.main()
