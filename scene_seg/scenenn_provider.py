import util

import random
import numpy as np
import tensorflow as tf
import h5py
import os
import sys

#all_files = ['005', '011', '014', '015', '016', '021', '025', '032', '036', '038', '041', '045', '047', '049', '052', '054', '057', '061', '062', '065', '066', '069', '071', '073', '074', '076', '078', '080', '082', '084', '086', '087', '089', '093', '096', '098', '109', '201', '202', '206', '207', '209', '213', '217', '223', '225', '227', '231', '234', '237', '240', '243', '246', '249', '251', '252', '255', '260', '263', '265', '270', '272', '273', '276', '279', '286', '294', '308', '522', '527', '609', '613', '614', '621', '623', '700']

train_files = ['005', '014', '015', '016', '025', '036', '038', '041', '045', '047', '052', '054', '057', '061', '062', '066', '071', '073', '078', '080', '084', '087', '089', '096', '098', '109', '201', '202', '209', '217', '223', '225', '227', '231', '234', '237', '240', '243', '249', '251', '255', '260', '263', '265', '270', '276', '279', '286', '294', '308', '522', '609', '613', '614', '623', '700']
test_files = ['011', '021', '065', '032', '093', '246', '086', '069', '206', '252', '273', '527', '621', '076', '082', '049', '207', '213', '272', '074']

class ScenennProvider:
    """
    Data consumer is designed to feed batch data for training. It should not contain any logic about data generation.
    """
    def __init__(self, data_root="", batch_size=1, sort_cloud=False, sort_method="xyz", training=True):
        self.batch_size = batch_size
       
        self.sort_cloud = sort_cloud
        self.sort_method = sort_method 

        self.training = training
        if self.training:
            files = train_files
        else:
            files = test_files 
        
        for file in files:
            file_path = data_root + '/' + 'scenenn_seg_' + file + '.hdf5'
            
            if not os.path.isfile(file_path):
                print('{} not existed. Skip.'.format(file_path))
                continue
            
            print('Loading ', file_path)
            with h5py.File(file_path, 'r') as f:
                data = np.array(f['data'])
                label = np.array(f['label'])
                if not hasattr(self, 'data'):
                    self.data = data
                    self.label = label

                    self.num_points = data.shape[1]
                    self.num_channels = data.shape[2]

                    print('Num points: ', self.num_points)
                    print('Num channels: ', self.num_channels)

                elif data.shape[0] > 0:
                    self.data = np.concatenate((self.data, data))
                    self.label = np.concatenate((self.label, label))
                else:
                    print('{} not have data. Skip.'.format(file_path))

        self.batch_points = np.zeros([self.batch_size, self.num_points, 3], dtype='float')
        self.batch_input = np.ones([self.batch_size, self.num_points, self.num_channels], dtype='float')
        self.batch_label = np.zeros([self.batch_size, self.num_points], dtype='uint8')
        
        self.num_batches = self.data.shape[0] // self.batch_size
        print('Num batches: ', self.num_batches)

        self.next_epoch()

    def next_epoch(self):
        self.cur_batch = 0

        # Reshuffle for each epoch
        self.permutation = np.arange(0, self.data.shape[0])
        if self.training:            
            np.random.shuffle(self.permutation)
            self.data = self.data[self.permutation, ...]
            self.label = self.label[self.permutation, ...]

    def has_next_batch(self):
        # Still have data in the current train file?
        if self.cur_batch + 1 < self.num_batches:
            return True

        return False

    def next_batch(self):
        self.cur_batch += 1
        return True

    def get_batch_point_cloud(self):
        start_idx = self.cur_batch * self.batch_size
        end_idx = (self.cur_batch + 1) * self.batch_size

        points_data = self.data[start_idx:end_idx, :, :]
        label_data = self.label[start_idx:end_idx, :]

        if self.sort_cloud:
            points_data, label_data = util.sort_point_cloud_xyz2(points_data, label_data)

        self.batch_points[:, :, :] = points_data[:, :, 0:3]
        self.batch_input[:, :, :] = points_data[:, :, 0:self.num_channels]
        self.batch_label[:] = label_data[:]

        #cloud = pe.PointCloud(self.batch_points[0, :, :].astype('float32'))
        #pe.Visualizer.show(cloud)

        return self.batch_points, self.batch_input, self.batch_label
