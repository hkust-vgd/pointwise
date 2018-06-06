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
import scenenn_provider as provider
import util

data_root = '/d/scenenn_seg/'

# for moving ply of each scene into a separate folder 
scenes = ['005', '011', '014', '015', '016', '021', '025', '032', '036', '038', '041', '045', '047', '049', '052', '054', '057', '061', '062', '065', '066', '069', '071', '073', '074', '076', '078', '080', '082', '084', '086', '087', '089', '093', '096', '098', '109', '201', '202', '206', '207', '209', '213', '217', '223', '225', '227', '231', '234', '237', '240', '243', '246', '249', '251', '252', '255', '260', '263', '265', '270', '272', '273', '276', '279', '286', '294', '308', '522', '527', '609', '613', '614', '621', '623', '700']
all_count = [652, 659, 381, 656, 433, 549, 307, 287, 854, 864, 697, 621, 658, 462, 481, 713, 381, 1209, 640, 816, 272, 610, 705, 186, 73, 881, 277, 362, 564, 441, 599, 1220, 644, 743, 670, 712, 125, 907, 784, 1689, 506, 135, 558, 635, 561, 704, 395, 415, 516, 513, 408, 260, 473, 319, 625, 492, 129, 492, 711, 586, 809, 945, 490, 697, 659, 821, 778, 1413, 429, 1735, 1149, 390, 693, 1258, 1467, 762]
test_files = ['011', '021', '065', '032', '093', '246', '086', '069', '206', '252', '273', '527', '621', '076', '082', '049', '207', '213', '272', '074']

bag = {}
for s, c in zip(scenes, all_count):
    bag[s] = c
pieces_count = [bag[s] for s in test_files]


def eval(args):
    
    max_epoch = args["max_epoch"]
    start_learning_rate = args["start_learning_rate"]
    decay_steps = args["decay_steps"]
    decay_rate = args["decay_rate"]
    momentum = args["momentum"]

    sort_cloud = args["sort_cloud"]
    snapshot_interval = args["snapshot_interval"]
    #batch_size = args["scene_seg"]["batch_size"]
    #batch_size = 1
    batch_size = 32

    if len(sys.argv) > 1:
        start_epoch = int(sys.argv[1])
    else:
        start_epoch = 0

    model_name = args["scene_seg"]["model"]
    model = importlib.import_module(model_name)
    print('Using model ', model_name)

    d = provider.ScenennProvider(data_root=data_root, batch_size=batch_size, sort_cloud=sort_cloud, training=False)
    num_points = d.num_points
    num_channels = d.num_channels
    num_class = 41

    print('Categories: ', num_class)
    print('Sort cloud: ', sort_cloud)

    #use_gpu = args["scene_seg"]["use_gpu"]
    use_gpu = False
    #use_gpu = True
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
            
            coords = np.zeros((num_points, 3))
            normals = np.zeros((num_points, 3))
            nyu_gt_labels = np.zeros(num_points, dtype=np.int32)
            nyu_predicted_labels = np.zeros(num_points, dtype=np.int32)

            cur_scene = 0
            cur_scene_id = test_files[cur_scene]
            cur_pieces = pieces_count[cur_scene]
            #gt_mesh = pe.SMesh()
            #predict_mesh = pe.SMesh()

            export_idx = 0
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

                # Visualize segmentation
                if not os.path.isdir('output'):
                    os.mkdir('output')                    
                for b in range(batch_size):
                    #print('Predicted labels of point cloud', b)

                    # continuous buffer 
                    coords[:, :] = points_data[b, :, 9:12]
                    normals[:, :] = points_data[b, :, 3:6]
                    nyu_gt_labels[:] = gt_label[b, :]
                    nyu_predicted_labels[:] = pred_label[b, :]

                    #predict_piece = pe.SMesh.from_xyz_normal_nyu(coords, normals, nyu_predicted_labels, num_points)
                    #gt_piece =  pe.SMesh.from_xyz_normal_nyu(coords, normals, nyu_gt_labels, num_points)
                    #predict_mesh.append(predict_piece)
                    #gt_mesh.append(gt_piece)

                    #pe.Visualizer.save("output/{}_predict_{}.ply".format(cur_scene_id, export_idx), predict_piece)
                    #print('Ground truth labels of point cloud', b)
                    #pe.Visualizer.save("output/{}_gt_{}.ply".format(cur_scene_id, export_idx), gt_piece)
                
                    export_idx += 1

                    if (export_idx >= cur_pieces):
                        if not os.path.isdir('output/{}'.format(cur_scene_id)):
                            os.mkdir('output/{}'.format(cur_scene_id))
                        #pe.Visualizer.save("output/{}/{}_predict.ply".format(cur_scene_id, cur_scene_id), predict_mesh)
                        #pe.Visualizer.save("output/{}/{}_gt.ply".format(cur_scene_id, cur_scene_id), gt_mesh)
                
                        cur_scene += 1
                        cur_scene_id = test_files[cur_scene]
                        cur_pieces = pieces_count[cur_scene]
                        export_idx = 0
                        #gt_mesh = pe.SMesh()
                        #predict_mesh = pe.SMesh()
                        

                #sys.exit(0)
                

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

            table_class_accuracy = total_correct_class / total_seen_class
            with open('test_perclass_{}_{}.csv'.format(model_name, start_epoch), 'w') as g:
                for u in range(table_class_accuracy.shape[0]):
                    g.write('{},{}\n'.format(u, table_class_accuracy[u]))






if __name__ == "__main__":
    args = util.parse_arguments("../param.json")
    eval(args)
