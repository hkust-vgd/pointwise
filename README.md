# Pointwise Convolutional Neural Networks

This is the release of the code for the paper `Pointwise Convolutional Neural Networks' in CVPR 2018. 

## Usage

The code is tested in latest Ubuntu 18.04 LTS with CUDA 9.2 and Tensorflow 1.9. 

First, we need to compile the convolution operator as follows:

    cd tf_ops/conv3p/
    chmod 777 tf_conv3p_compile.sh
    ./tf_conv3p_compile.sh -a

The result is a dynamic library file named `tf_conv3p.so`. The Python training and evaluation code loads this library for pointwise convolution. 
By default, the library contains both a CPU and a GPU implementation of the convolution operator. The `use_gpu` flag in `param.json` can be set to `true` to enable the convolution on the GPU. 

To train object classification, execute 

    python train_modelnet40_acsd.py [epoch]

To evaluate, execute 

    python eval_modelnet40_acsd.py [epoch] 
   
By default, `epoch` is 0 if it is not passed as a parameter to the above command. During training, the network is saved after each epoch. You can resume the training if the network was saved before. Just pass the epoch number to the training command.

Similar code structure is adopted for scene segmentation task. For this task, we also provide a re-implementation of PointNet in PyTorch based on the open source implementation by [fxia22](https://github.com/fxia22/pointnet.pytorch).

## Training Data 

- [ModelNet40](https://shapenet.cs.stanford.edu/media/modelnet40_ply_hdf5_2048.zip): 450 MB.
- [SceneNN Segmentation](https://drive.google.com/open?id=1PkP6m2mxCfCh_u3SnzLZod7J1zpzD6t4): 5.5 GB. With 76 scenes, re-annotated with NYU-D v2 40 classes. 56 scenes for training and 20 scenes for testing.  
- [S3DIS Segmentation](https://shapenet.cs.stanford.edu/media/indoor3d_sem_seg_hdf5_data.zip): 1.6 GB.

## Troubleshooting 

If you are using Tensorflow 1.4, you might want to try compiling with `tf_conv3p_compile_tf14.sh` instead. It fixes some include paths due to `nsync_cv.h`, and set the flag `_GLIBCXX_USE_CXX11_ABI=0` to make it compatible to libraries compiled with GCC version earlier than 5.1. 

## Performance

As this is a custom convolution operator we built with minimum optimization tricks that we know, you might find it running more slowly than those Tensorflow built-in operators. 
Despite that, the experiments were done on NVIDIA GTX 1070, GTX 1080, and Titan X (first generation) without big issues. 

It will take hours or 1-2 days depending on your setup to finish training for object recognition. For scene segmentation, it might take longer. 

## Dependencies

This code includes the following third party libraries and data:

- Scaled exponential linear units (SeLU) for self-normalization in neural network.

- ModelNet40 data from PointNet

- Some other utility code from PointNet

- h5py

## Citation 

Please cite our [paper](https://arxiv.org/abs/1712.05245)
  
    @inproceedings{hua-pointwise-cvpr18,
        title = {Pointwise Convolutional Neural Networks},
        author = {Binh-Son Hua and Minh-Khoi Tran and Sai-Kit Yeung},
        booktitle = {Computer Vision and Pattern Recognition (CVPR)},
        year = {2018}
    }

if you find this useful for your work. 

## Future work 

We made this simple operator with the hope that existing techniques in 2D image understanding tasks can be brought to 3D in a more straightforward manner. More research along this direction is encouraged. 

Please contact the authors at binhson.hua@gmail.com if you have any queries. 
