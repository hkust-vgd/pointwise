import tensorflow as tf
from tensorflow.python.framework import ops
import sys
import os
import timeit
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)
print(BASE_DIR)

conv3p_module = tf.load_op_library('./tf_conv3p.so')

if __name__=='__main__':
    import numpy as np

    np.random.seed(100)
    b = 32
    n = 2048 # only at 20480 points grid starts to show speed up
    points = np.random.random((b, n ,3)).astype('float64') * 2.0 - 1.0
    input = np.random.random((b, n, 8)).astype('float64') * 2.0 - 1.0
    kernel = np.random.random((3,3,3,8,2)).astype('float64') * 2.0 - 1.0
    voxel_size = np.array([0.05])
    stride = np.array([2, 2, 2])

    """
    np.random.seed(100)
    b = 6
    n = 16
    points = np.random.random((b, n ,3)).astype('float64') * 0.05 - 0.025
    input = np.random.random((b, n, 8)).astype('float64') * 0.05 - 0.025
    kernel = np.random.random((3,3,3,8,2)).astype('float64') * 0.05 - 0.025
    voxel_size = np.array([0.05])
    stride = np.array([2, 2, 2])
    """
    # Create network    
    with tf.device('/cpu:0'):
    #with tf.device('/gpu:0'):
        config = tf.ConfigProto(
    #        device_count = {'GPU': 0},
            log_device_placement = False
        )

        points_tensor = tf.constant(points)
        input_tensor = tf.constant(input);
        kernel_tensor = tf.constant(kernel)
        voxel_size_tensor = tf.constant(voxel_size, dtype='float64')
        stride_tensor = tf.constant(stride, dtype='int32')
        output_tensor = conv3p_module.conv3p(points_tensor, input_tensor, kernel_tensor, stride_tensor, voxel_size_tensor);

        with tf.Session(config=config) as sess:
            tic = timeit.default_timer()
            #for _ in range(100):
            ret = sess.run(output_tensor)
            toc = timeit.default_timer()
            print('Time: ', toc - tic)
            print(ret.shape, ret.dtype)
            print(ret[b//2, n//2, :])
