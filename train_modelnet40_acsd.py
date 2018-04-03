import tensorflow as tf
import numpy as np
import modelnet_provider as modelnet
import importlib
import util
import os.path
import sys
import timeit
import datetime

def train(args):

    num_points = 2048   # ModelNet40
    num_channels = 3

    max_epoch = args["max_epoch"]
    start_learning_rate = args["start_learning_rate"]
    decay_steps = args["decay_steps"]
    decay_rate = args["decay_rate"]
    momentum = args["momentum"]

    sort_cloud = args["sort_cloud"]
    sort_method = args["sort_method"]
    snapshot_interval = args["snapshot_interval"]
    batch_size = args["batch_size"]

    if len(sys.argv) > 1:
        start_epoch = int(sys.argv[1])
    else:
        start_epoch = 0

    model_name = args["model"]
    model = importlib.import_module(model_name)
    print('Using model ', model_name)

    d = modelnet.DataConsumer(file='data/modelnet40_ply_hdf5_2048/train_files.txt', batch_size=batch_size, num_points=num_points, num_channels=num_channels, sort_cloud=sort_cloud, sort_method=sort_method)
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
                batch_prediction_placeholder = network.model(batch_points_placeholder, batch_input_placeholder, is_training=True)

            loss = network.loss(batch_prediction_placeholder, batch_label_placeholder)

            # Track the global training progress
            global_step = tf.Variable(0, name='global_step', trainable=False)
            # Adaptive learning rate
            learning_rate = tf.train.exponential_decay(start_learning_rate, global_step,
                                                       decay_steps, decay_rate, staircase=True)

            optimizer = tf.train.MomentumOptimizer(learning_rate, momentum)
            train_op = optimizer.minimize(loss, global_step=global_step)

            # Save to disk
            saver = tf.train.Saver(max_to_keep=None)

            weight_file = "{}_snapshot_{}.tf".format(model_name, start_epoch)
            if start_epoch > 0 and os.path.isfile(weight_file+".index"):
                saver.restore(session, weight_file)
                print('Weight file {} restored'.format(weight_file))
                start_epoch += 1
            else:
                print('Train from scratch');
                init_op = tf.global_variables_initializer()
                session.run(init_op)
                start_epoch = 0

            # Live statistics
            tf.summary.scalar('learning_rate', learning_rate)
            tf.summary.scalar('loss', loss)

            # Write some statistics to file during training
            merged = tf.summary.merge_all()
            train_writer = tf.summary.FileWriter('log', session.graph)

            for epoch in range(start_epoch, max_epoch):
                print('# Epoch %d' % epoch)

                # one epoch
                total_correct = 0
                total_seen = 0
                total_correct_class = np.zeros((num_class))
                total_seen_class = np.zeros((num_class))
                total_loss = 0

                num_batches = 0
                batch_time = 0
                
                while True:
                    points, points_data, gt_label = d.get_batch_point_cloud()

                    tic = timeit.default_timer()
                    feed_dict = {
                        batch_points_placeholder : points,
                        batch_input_placeholder : points_data,
                        batch_label_placeholder : gt_label
                    }
                    _, summary, loss_val, pred_val = session.run([train_op, merged, loss, batch_prediction_placeholder], feed_dict=feed_dict)
                    toc = timeit.default_timer()
                    batch_time += toc - tic

                    pred_label = np.argmax(pred_val, axis=1)

                    correct = np.sum(pred_label == gt_label)
                    total_correct += correct
                    total_seen += batch_size
                    total_loss += loss_val

                    for i in range(batch_size):
                        l = gt_label[i]
                        total_seen_class[l] += 1
                        total_correct_class[l] += (pred_label[i] == l)

                    num_batches += 1
                    
                    print('Current loss       : %f' % loss_val)
                    print('Current accuracy   : %f' % (correct / float(batch_size)))
                    print('Average batch time : ', batch_time / num_batches)
                    
                    if not d.has_next_batch():
                        break
                    d.next_batch()

                mean_loss = (total_loss / float(num_batches))
                mean_accuracy = (total_correct / float(total_seen))
                mean_class_accuracy = (np.mean(total_correct_class / total_seen_class))
                print('Mean loss          : %f' % mean_loss)
                print('Mean accuracy      : %f' % mean_accuracy)
                print('Avg class accuracy : %f' % mean_class_accuracy)
                train_writer.add_summary(summary)

                if epoch % snapshot_interval == 0 or epoch == max_epoch - 1:
                    snapshot_file = "{}_snapshot_{}.tf".format(model_name, epoch)
                    saver.save(session, snapshot_file)
                    print('Model %s saved' % snapshot_file)

                global_step_val, learning_rate_val = session.run([global_step, learning_rate])
                print('Global step    : ', global_step_val)
                print('Learning rate  : ', learning_rate_val)

                with open("train_info_{}.csv".format(model_name), 'a') as f:
                    f.write(str(datetime.datetime.now().strftime("%c")) + ', ' + \
                            str(epoch) + ', ' + str(mean_loss) + ', ' + str(mean_accuracy) + ', ' + str(mean_class_accuracy) + '\n')

                d.next_epoch()

if __name__ == "__main__":
    args = util.parse_arguments("param.json")
    train(args)
