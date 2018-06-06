import os
import sys
import numpy as np
import h5py
import util

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)

def shuffle_data(data, labels):
    """ Shuffle data and labels.
        Input:
          data: B,N,... numpy array
          label: B,... numpy array
        Return:
          shuffled data, label and shuffle indices
    """
    idx = np.arange(len(labels))
    np.random.shuffle(idx)
    return data[idx, ...], labels[idx], idx


def rotate_point_cloud(batch_data):
    """ Randomly rotate the point clouds to augument the dataset
        rotation is per shape based along up direction
        Input:
          BxNx3 array, original batch of point clouds
        Return:
          BxNx3 array, rotated batch of point clouds
    """
    rotated_data = np.zeros(batch_data.shape, dtype=np.float32)
    for k in range(batch_data.shape[0]):
        rotation_angle = np.random.uniform() * 2 * np.pi
        cosval = np.cos(rotation_angle)
        sinval = np.sin(rotation_angle)
        rotation_matrix = np.array([[cosval, 0, sinval],
                                    [0, 1, 0],
                                    [-sinval, 0, cosval]])
        shape_pc = batch_data[k, ...]
        rotated_data[k, ...] = np.dot(shape_pc.reshape((-1, 3)), rotation_matrix)
    return rotated_data


def rotate_point_cloud_by_angle(batch_data, rotation_angle):
    """ Rotate the point cloud along up direction with certain angle.
        Input:
          BxNx3 array, original batch of point clouds
        Return:
          BxNx3 array, rotated batch of point clouds
    """
    rotated_data = np.zeros(batch_data.shape, dtype=np.float32)
    for k in range(batch_data.shape[0]):
        #rotation_angle = np.random.uniform() * 2 * np.pi
        cosval = np.cos(rotation_angle)
        sinval = np.sin(rotation_angle)
        rotation_matrix = np.array([[cosval, 0, sinval],
                                    [0, 1, 0],
                                    [-sinval, 0, cosval]])
        shape_pc = batch_data[k, ...]
        rotated_data[k, ...] = np.dot(shape_pc.reshape((-1, 3)), rotation_matrix)
    return rotated_data


def jitter_point_cloud(batch_data, sigma=0.01, clip=0.05):
    """ Randomly jitter points. jittering is per point.
        Input:
          BxNx3 array, original batch of point clouds
        Return:
          BxNx3 array, jittered batch of point clouds
    """
    B, N, C = batch_data.shape
    assert(clip > 0)
    jittered_data = np.clip(sigma * np.random.randn(B, N, C), -1*clip, clip)
    jittered_data += batch_data
    return jittered_data

def getDataFiles(list_filename):
    return [line.rstrip() for line in open(list_filename)]

def load_h5(h5_filename):
    f = h5py.File(h5_filename)
    data = f['data'][:]
    label = f['label'][:]
    return (data, label)

def loadDataFile(filename):
    return load_h5(filename)

def load_h5_data_label_seg(h5_filename):
    f = h5py.File(h5_filename)
    data = f['data'][:]
    label = f['label'][:]
    seg = f['pid'][:]
    return (data, label, seg)


def loadDataFile_with_seg(filename):
    return load_h5_data_label_seg(filename)

def sort_point_cloud_morton(points):
    import libpluie as pe
    batch_size = points.shape[0]
    num_points = points.shape[1]
    pc = np.zeros((num_points, 3))
    for b in range(batch_size):
        pc[:, :] = points[b, :, 0:3]
        indices = pe.PointCloud.arg_sort_morton(pe.PointCloud(pc))
        # permute 
        points[b, :, :] = points[b, indices, :]
    return points 



class DataConsumer:
    """
    A wrapper to ModelNet40 provider
    """
    def __init__(self, file='', num_points=4096, num_channels=1, batch_size=1, test=False, sort_cloud=False, sort_method="xyz"):
        self.TRAIN_FILES = getDataFiles(os.path.join(BASE_DIR, file))
        self.test = test
        self.sort_cloud = sort_cloud
        self.sort_method = sort_method

        self.batch_size = batch_size
        self.num_points = num_points
        self.num_channels = num_channels

        self.batch_points = np.zeros([self.batch_size, self.num_points, 3], dtype='float')
        self.batch_input = np.ones([self.batch_size, self.num_points, self.num_channels], dtype='float')
        self.batch_label = np.zeros([self.batch_size], dtype='uint8')

        self.next_epoch()

    def next_epoch(self):
        self.cur_batch = 0
        self.fn = 0

        # Reshuffle for each epoch
        self.train_file_idxs = np.arange(0, len(self.TRAIN_FILES))
        if not self.test:
            np.random.shuffle(self.train_file_idxs)

        self.load_current_data()

    def has_next_batch(self):
        """
        Check global batch across train files
        """

        # Still have data in the current train file?
        if self.cur_batch + 1 < self.num_batches:
            return True

        # No data in the current train file, but have any in the next train file?
        if self.has_next_train_file():
            self.next_train_file()
            if self.has_next_batch():
                return True

        return False

    def next_batch(self):
        self.cur_batch += 1
        return True

    def has_next_train_file(self):
        if self.fn + 1 < len(self.TRAIN_FILES):
            return True
        return False

    def load_current_data(self):
        current_data, current_label = loadDataFile(self.TRAIN_FILES[self.train_file_idxs[self.fn]])
        current_data = current_data[:, 0:self.num_points, :]
        if not self.test:
            current_data, current_label, _ = shuffle_data(current_data, np.squeeze(current_label))
        current_label = np.squeeze(current_label)

        # TODO: divisible by batch size?

        self.current_data = current_data
        self.current_label = current_label

        self.num_batches = current_data.shape[0] // self.batch_size

    def next_train_file(self):
        self.cur_batch = 0
        self.fn += 1

        self.load_current_data()

    def get_batch_point_cloud(self):
        start_idx = self.cur_batch * self.batch_size
        end_idx = (self.cur_batch + 1) * self.batch_size

        # Augment batched point clouds by rotation and jittering during training
        if not self.test:
            rotated_data = rotate_point_cloud(self.current_data[start_idx:end_idx, :, :])
            jittered_data = jitter_point_cloud(rotated_data)
        else:
            jittered_data = self.current_data[start_idx:end_idx, :, :]

        if self.sort_cloud:
            if self.sort_method == "xyz":
                jittered_data = util.sort_point_cloud_xyz(jittered_data)
            elif self.sort_method == "morton":
                jittered_data = sort_point_cloud_morton(jittered_data)
            else:
                print('Unknown sort_method=', sort_method)

        label_data = self.current_label[start_idx:end_idx]

        self.batch_points[:] = jittered_data[:]
        self.batch_input[:] = jittered_data[:]
        self.batch_label[:] = label_data[:]

        #cloud = pe.PointCloud(self.batch_points[0, :, :].astype('float32'))
        #pe.Visualizer.show(cloud)

        return self.batch_points, self.batch_input, self.batch_label
