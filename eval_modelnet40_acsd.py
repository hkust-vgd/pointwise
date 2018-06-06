import tensorflow as tf
import numpy as np
import modelnet_provider as modelnet
import importlib
import util
import sys
import datetime 

def eval(args):

    num_points = 2048   # ModelNet40
    num_channels = 3

    model_name = args["model"]
    model = importlib.import_module(model_name)
    print('Using model ', model_name)

    sort_cloud = args['sort_cloud']
    sort_method= args['sort_method']
    if len(sys.argv) > 1:
        epoch = int(sys.argv[1])
    else:
        epoch = 0

    batch_size = args["batch_size"]
    d = modelnet.DataConsumer(file='data/modelnet40_ply_hdf5_2048/test_files.txt', batch_size=batch_size, num_points=num_points, num_channels=num_channels, test=True, sort_cloud=sort_cloud, sort_method=sort_method)
    num_class = 40

    print('Categories: ', num_class)
    print('Sort cloud: ', sort_cloud)
    
    use_gpu = args["use_gpu"]
    if use_gpu:
        str_device = '/gpu:0'
    else:
        str_device = '/cpu:0'
    print('Using device', str_device)
    
    with tf.Graph().as_default(), tf.device(str_device):

        if not use_gpu:
            config = tf.ConfigProto(
                device_count = {'GPU': 0},
                log_device_placement = False
            )
        else:
            config = tf.ConfigProto(
                allow_soft_placement = True,
                log_device_placement = False
            )
            
        with tf.Session(config=config) as session:
            network = model.PointConvNet(num_class)

            batch_points_placeholder = tf.placeholder(tf.float32, shape=(batch_size, num_points, 3))
            batch_input_placeholder = tf.placeholder(tf.float32, shape=(batch_size, num_points, num_channels))
            batch_label_placeholder = tf.placeholder(tf.int32, shape=(batch_size))

            with tf.variable_scope("pointcnn") as scope:
                batch_prediction_placeholder = network.model(batch_points_placeholder, batch_input_placeholder, is_training=False)

            saver = tf.train.Saver(max_to_keep=None)
            file = "./{}_snapshot_{}.tf".format(model_name, epoch)
            print('Testing with model ', file)
            saver.restore(session, file)

            total_correct = 0
            total_seen = 0
            total_correct_class = np.zeros((num_class))
            total_seen_class = np.zeros((num_class))

            num_batches = 0

            while True:
                points, points_data, gt_label = d.get_batch_point_cloud()

                feed_dict = {
                    batch_points_placeholder : points,
                    batch_input_placeholder : points_data,
                    batch_label_placeholder : gt_label
                }
                [pred_val] = session.run([batch_prediction_placeholder], feed_dict=feed_dict)

                pred_label = np.argmax(pred_val, axis=1)

                correct = np.sum(pred_label == gt_label)
                total_correct += correct
                total_seen += batch_size

                for i in range(batch_size):
                    l = gt_label[i]
                    total_seen_class[l] += 1
                    total_correct_class[l] += (pred_label[i] == l)

                num_batches += 1
                print('Current accuracy  : %f' % (correct / float(batch_size)))
                
                if not d.has_next_batch():
                    break
                d.next_batch()
                

            mean_accuracy = (total_correct / float(total_seen))
            mean_class_accuracy = (np.mean(total_correct_class / total_seen_class))
            print('Mean accuracy      : %f' % mean_accuracy)
            print('Avg class accuracy : %f' % mean_class_accuracy)

            with open("test_info_{}.csv".format(model_name), 'a') as f:
                f.write(str(datetime.datetime.now().strftime("%c")) + ', ' + \
                        str(epoch) + ', ' + str(mean_accuracy) + ', ' + str(mean_class_accuracy) + '\n')


if __name__ == "__main__":
    args = util.parse_arguments("param.json")
    eval(args)
