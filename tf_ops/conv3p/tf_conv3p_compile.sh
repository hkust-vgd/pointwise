#/bin/bash

# use -a option to enable atrous 

TF_INC=$(python3 -c 'import tensorflow as tf; print(tf.sysconfig.get_include())')

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

# with CUDA
#/usr/local/cuda-8.0/bin/nvcc $FILE.cu -std=c++11 -o $FILE.so -shared -g -O3 -DGOOGLE_CUDA=1 -Xcompiler -fPIC -use_fast_math -I /usr/local/cuda-8.0/include -I $TF_INC  -lcudart -L /usr/local/cuda-8.0/lib64/ -D_GLIBCXX_USE_CXX11_ABI=0 

# with OpenMP
#g++ -std=c++11 $FILE.cpp -o $FILE.so -shared -fPIC -I $TF_INC -g -O2 -fopenmp -DCONV_OPENMP -D_GLIBCXX_USE_CXX11_ABI=0

# no OpenMP
#g++ -std=c++11 $FILE.cpp -o $FILE.so -shared -fPIC -I $TF_INC -g -O2 -D_GLIBCXX_USE_CXX11_ABI=0
#g++ -std=c++11 $FILE.cpp -o $FILE.so -shared -fPIC -I $TF_INC -g -O0 -D_GLIBCXX_USE_CXX11_ABI=0

# Both CUDA and CPU code in a single library: compile each implementation, and register op and link both into the same shared library
g++ -std=c++11 -c $FILE_CPU.cpp -o ${OUT}_cpu.o -fPIC -I $TF_INC -g -O3 -fopenmp -DCONV_OPENMP -D_GLIBCXX_USE_CXX11_ABI=0
/usr/local/cuda-8.0/bin/nvcc -std=c++11 -c $FILE_CUDA.cu -o ${OUT}_cuda.o -c -g -O3 -DGOOGLE_CUDA=1 -Xcompiler -fPIC -use_fast_math -I /usr/local/cuda-8.0/include -I $TF_INC  -D_GLIBCXX_USE_CXX11_ABI=0
g++ -std=c++11 -c register_op.cpp -o register_op.o -fPIC -I $TF_INC -g -O3 -D_GLIBCXX_USE_CXX11_ABI=0 $FLAGS
g++ -shared -o $OUT register_op.o ${OUT}_cuda.o ${OUT}_cpu.o -lcudart -L /usr/local/cuda-8.0/lib64/
rm ${OUT}_cpu.o 
rm ${OUT}_cuda.o

