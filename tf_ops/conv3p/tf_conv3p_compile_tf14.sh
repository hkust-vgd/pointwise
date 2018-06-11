#!/bin/bash

# use -a option to enable atrous 

TF_INC=$(python3 -c 'import tensorflow as tf; print(tf.sysconfig.get_include())')
TF_LIB=$(python3 -c 'import tensorflow as tf; print(tf.sysconfig.get_lib())')

# non-atrous convolution
FILE_CUDA=tf_conv3p_naive_ctxmem
FILE_CPU=tf_conv3p_grid

FLAGS=
while getopts ":a" opt; do
  case $opt in
    a) 
    # atrous convolution
    FLAGS=-DATROUS=1    
    FILE_CUDA=tf_conv3p_atrous
    FILE_CPU=tf_conv3p_atrous
    ;;    
    \?) echo "Usage: cmd [-a]"
    ;;
  esac
done

OUT=tf_conv3p.so 

# This flag is only required for compatibility if your Tensorflow is built with gcc < 5.1 or has ABI tag disabled
ABI_FLAG=-D_GLIBCXX_USE_CXX11_ABI=0

# Both CUDA and CPU code in a single library: compile each implementation, and register op and link both into the same shared library
g++ -std=c++11 -c $FILE_CPU.cpp -o ${OUT}_cpu.o -fPIC -I $TF_INC -I $TF_INC/external/nsync/public -g -O3 -fopenmp -DCONV_OPENMP $ABI_FLAG
/usr/local/cuda/bin/nvcc -std=c++11 -c $FILE_CUDA.cu -o ${OUT}_cuda.o -c -g -O3 -DGOOGLE_CUDA=1 -Xcompiler -fPIC -use_fast_math -I /usr/local/cuda/include -I $TF_INC -I $TF_INC/external/nsync/public $ABI_FLAG
g++ -std=c++11 -c register_op.cpp -o register_op.o -fPIC -I $TF_INC -I $TF_INC/external/nsync/public -g -O3 $ABI_FLAG $FLAGS

# Linking. From Tensorflow 1.4 somehow we need tensorflow_framework
g++ -shared -o $OUT register_op.o ${OUT}_cuda.o ${OUT}_cpu.o -lcudart -L /usr/local/cuda/lib64/ -L$TF_LIB -ltensorflow_framework -fopenmp

# Clean up
rm ${OUT}_cpu.o 
rm ${OUT}_cuda.o

