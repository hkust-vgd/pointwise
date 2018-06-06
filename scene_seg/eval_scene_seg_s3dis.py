import tensorflow as tf
import numpy as np
import importlib
import os.path
import sys
import timeit
import datetime

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.dirname(BASE_DIR)
sys.path.append(PARENT_DIR)
import s3dis_provider as provider
import util

def eval(args):
    
    max_epoch = args["max_epoch"]
    start_learning_rate = args["start_learning_rate"]
    decay_steps = args["decay_steps"]
    decay_rate = args["decay_rate"]
    momentum = args["momentum"]

    sort_cloud = args["sort_cloud"]
    snapshot_interval = args["snapshot_interval"]
    batch_size = args["batch_size"]

    if len(sys.argv) > 1:
        start_epoch = int(sys.argv[1])
    else:
        start_epoch = 0

    model_name = args["scene_seg"]["model"]
    model = importlib.import_module(model_name)
    print('Using model ', model_name)

    d = provider.S3DISProvider(batch_size=batch_size, sort_cloud=sort_cloud, training=False)
    num_points = d.num_points
    num_channels = d.num_channels
    num_class = 13

    print('Categories: ', num_class)
    print('Sort cloud: ', sort_cloud)

    with tf.device('/cpu:0'): # For now CPU only
        config = tf.ConfigProto(
            device_count = {'GPU': 0},
            log_device_placement = False
        )
        with tf.Session(config=config) as session:
            network = model.PointConvNet(num_class)

            batch_points_placeholder = tf.placeholder(tf.float32, shape=(batch_size, num_points, 3))
            batch_input_placeholder = tf.placeholder(tf.float32, shape=(batch_size, num_points, num_channels))
            batch_label_placeholder = tf.placeholder(tf.int32, shape=(batch_size, num_points))

            with tf.variable_scope("pointcnn") as scope:
                batch_prediction_placeholder = network.model(batch_points_placeholder, batch_input_placeholder, is_training=False)

            # Save to disk
            saver = tf.train.Saver(max_to_keep=None)

            weight_file = "./{}_snapshot_{}.tf".format(model_name, start_epoch)            
            saver.restore(session, weight_file)
            print('Weight file {} restored'.format(weight_file))
            
            # one epoch
            total_correct = 0
            total_seen = 0
            total_correct_class = np.zeros((num_class))
            total_seen_class = np.zeros((num_class))

            num_seen_batches = 0
            batch_time = 0
            
            while True:
                points, points_data, gt_label = d.get_batch_point_cloud()

                tic = timeit.default_timer()
                feed_dict = {
                    batch_points_placeholder : points,
                    batch_input_placeholder : points_data,
                    batch_label_placeholder : gt_label
                }
                [pred_val] = session.run([batch_prediction_placeholder], feed_dict=feed_dict)
                toc = timeit.default_timer()
                batch_time += toc - tic

                pred_label = np.argmax(pred_val, axis=2)

                correct = np.sum(pred_label == gt_label)
                total_correct += correct
                total_seen += batch_size * num_points
                                
                for i in range(batch_size):
                    for j in range(num_points):
                        l = gt_label[i, j]
                        total_seen_class[l] += 1
                        total_correct_class[l] += (pred_label[i, j] == l)

                num_seen_batches += 1

                print('Batch {} / {}'.format(num_seen_batches, d.num_batches))                
                print('Current accuracy   : %f' % (correct / float(batch_size * num_points)))
                print('Average batch time : ', batch_time / num_seen_batches)

                if not d.has_next_batch():
                    break
                d.next_batch()                
            
            mean_accuracy = (total_correct / float(total_seen))
            mean_class_accuracy = np.mean(total_correct_class / total_seen_class)
            print('Mean accuracy      : %f' % mean_accuracy)
            print('Avg class accuracy : %f' % mean_class_accuracy)

if __name__ == "__main__":
    args = util.parse_arguments("../param.json")
    eval(args)
