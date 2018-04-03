import os
import sys
import numpy as np
import h5py
import util

# Dataset availability check
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, 'data/indoor3d_sem_seg_hdf5_4096')
if not os.path.exists(DATA_DIR):
    print('Data folder {0} not found. Please run download.py.' % DATA_DIR)
    sys.exit()

def getDataFiles(list_filename):
    return [line.rstrip() for line in open(list_filename)]

def load_h5(h5_filename):
    f = h5py.File(h5_filename)
    data = f['data'][:]
    label = f['label'][:]
    return (data, label)

class S3DISProvider:
   
    def __init__(self, batch_size=1, training=True, sort_cloud=False):
        ALL_FILES = getDataFiles(os.path.join(DATA_DIR, 'all_files.txt'))
        room_filelist = getDataFiles(os.path.join(DATA_DIR, 'room_filelist.txt')) # test data

        # Load ALL data into the memory
        data_batch_list = []
        label_batch_list = []
        for h5_filename in ALL_FILES:
            data_batch, label_batch = load_h5(os.path.join(DATA_DIR, h5_filename))
            data_batch_list.append(data_batch)
            label_batch_list.append(label_batch)
        data_batches = np.concatenate(data_batch_list, 0)
        label_batches = np.concatenate(label_batch_list, 0)
        print(data_batches.shape)
        print(label_batches.shape)

        test_area = 6
        test_area = 'Area_'+str(test_area)
        train_idxs = []
        test_idxs = []
        for i,room_name in enumerate(room_filelist):
            if test_area in room_name:
                test_idxs.append(i)
            else:
                train_idxs.append(i)

        train_data = data_batches[train_idxs,...]
        train_label = label_batches[train_idxs]
        test_data = data_batches[test_idxs,...]
        test_label = label_batches[test_idxs]
        print(train_data.shape, train_label.shape)
        print(test_data.shape, test_label.shape)

        self.training = training
        self.sort_cloud = sort_cloud

        self.batch_size = batch_size
        self.num_points = train_data.shape[1]
        self.num_channels = train_data.shape[2]

        if self.training:
            self.current_data = train_data 
            self.current_label = train_label
        else:
            self.current_data = test_data 
            self.current_label = test_label

        self.num_batches = self.current_data.shape[0] // self.batch_size

        self.batch_points = np.zeros([self.batch_size, self.num_points, 3], dtype='float')
        self.batch_input = np.ones([self.batch_size, self.num_points, self.num_channels], dtype='float')
        self.batch_label = np.zeros([self.batch_size, self.num_points], dtype='uint8')
    
        self.next_epoch()

    def next_epoch(self):
        self.cur_batch = 0

        # Reshuffle for each epoch
        self.permutation = np.arange(0, self.current_data.shape[0])
        if self.training:            
            np.random.shuffle(self.permutation)
            self.current_data = self.current_data[self.permutation, ...]
            self.current_label = self.current_label[self.permutation, ...]

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

        points_data = self.current_data[start_idx:end_idx, :, :]
        label_data = self.current_label[start_idx:end_idx, :]

        if self.sort_cloud:
            points_data, label_data = util.sort_point_cloud_xyz2(points_data, label_data)

        self.batch_points[:, :, :] = points_data[:, :, 0:3]
        self.batch_input[:] = points_data[:]
        self.batch_label[:] = label_data[:]

        #cloud = pe.PointCloud(self.batch_points[0, :, :].astype('float32'))
        #pe.Visualizer.show(cloud)

        return self.batch_points, self.batch_input, self.batch_label
