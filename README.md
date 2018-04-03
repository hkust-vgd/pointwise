# Pointwise Convolutional Neural Networks

This is the pre-release of the code for the paper `Pointwise Convolutional Neural Networks' in CVPR 2018. 
This code is under further revision, basically just cleaning up and removing unnecessary dependencies. 
For now, please kindly only use the code for education or non-commercial purposes.

## Dependencies

- Scaled exponential linear units (SeLU) for self-normalization in neural network.

- ModelNet40 data from PointNet

- Some other utility code from PointNet


## Usage

Please install Tensorflow and a C++ compiler to compile the convolution operator to a .so file. 
To start, you might want to simply compile the convolution on the CPU, and make sure it work properly first. We experimented with GPU implementation, and also multi-GPU implementation. 

See train_modelnet40_acsd.py as an example how to train object recognition network. Run eval_modelnet40_acsd.py to evaluate the accuracy. 
Similar code structure is adopted for scene segmentation task. 

As this is a custom convolution operator we built with minimum optimization tricks that we know, you will find it running more slowly than those Tensorflow built-in operators. 
Despite that, the experiments were done on NVIDIA GTX 1070, GTX 1080, and Titan X (first generation) without big issues. 

It will take hours or 1-2 days depending on your setup to finish training for object recognition. For scene segmentation, it might take longer. 


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
