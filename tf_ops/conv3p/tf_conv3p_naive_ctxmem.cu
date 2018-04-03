#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "tensorflow/core/framework/common_shape_fns.h"

#include <cuda_runtime.h>

#include <vector>
#include <iostream>

#include <chrono>
#include <type_traits>


using namespace tensorflow;

const int kThreadsPerBlock = 256;

#define min(a, b) ((a) > (b))? (b): (a)
#define max(a, b) ((a) > (b))? (a): (b)

__host__ __device__ __forceinline__ int divUp(int total, int grain)
{
    return (total + grain - 1) / grain;
}

#if !defined(__CUDA_ARCH__) || __CUDA_ARCH__ >= 600

  #else
  static __inline__ __device__ double atomicAdd(double *address, double val) {
    unsigned long long int* address_as_ull = (unsigned long long int*)address;
    unsigned long long int old = *address_as_ull, assumed;
    if (val==0.0)
      return __longlong_as_double(old);
    do {
      assumed = old;
      old = atomicCAS(address_as_ull, assumed, __double_as_longlong(val +__longlong_as_double(assumed)));
    } while (assumed != old);
    return __longlong_as_double(old);
  }


#endif

void cudaErrorCheck(int line) {
  cudaError_t error = cudaGetLastError();
  if(error != cudaSuccess)
  {
    // print the CUDA error message and exit
    printf("CUDA error at line %d: %s\n", line, cudaGetErrorString(error));
    exit(-1);
  }
}


class timer
{
    // alias our types for simplicity
    using clock = typename std::conditional< std::chrono::high_resolution_clock::is_steady,
                                                  std::chrono::high_resolution_clock,
                                                  std::chrono::steady_clock >::type ;
    using time_point_type   = std::chrono::time_point < clock, std::chrono::milliseconds > ;
public:
    // default constructor that stores the start time
    timer()
    {
        start = std::chrono::time_point_cast<std::chrono::milliseconds>(clock::now());
    }

    // gets the time elapsed from construction.
    float getTimePassed()
    {
        // get the new time
        auto end = clock::now();

        // return the difference of the times
        return (end - start).count() / 1e9f;
    }

private:
    time_point_type start;
};


struct GpuAlloc {
    
    GpuAlloc(OpKernelContext *context) {
        this->context = context;
    }

	void alloc(void**data, int bytes) {
		// We use Tensorflow's allocate_temp to allocate memory instead of using cudaMalloc/cudaFree directly. 
        // Using cudaMalloc/cudaFree is totally ok for a single GPU case, but can cause some race conditions in multi-GPU case, 
        // resulting in long delays in memory allocation and free. 
        
        int num_elems = std::ceil(bytes / 8.0);
        		
        // Have to keep the tensors until end of op to avoid memory crash.
        tmp_tensors.push_back(Tensor());
        Tensor &tmp = tmp_tensors.back();
        TensorShape shape({num_elems});
        OP_REQUIRES_OK(context, context->allocate_temp(DataTypeToEnum<double>::value, shape, &tmp));

        double *buf = &(tmp.flat<double>()(0));
        *data = (void*)buf;
	}
	void free(void **data) {		
		*data = NULL;
	}
	void zero(void *data, int bytes) {
		cudaMemset(data, 0, bytes);
	}
    OpKernelContext *context;
    std::vector<Tensor> tmp_tensors;

};

/*
struct GpuAlloc {
    GpuAlloc(OpKernelContext *context) {
        
    }

	void alloc(void **data, int bytes) {
		cudaMalloc(data, bytes);
	}
	void free(void **data) {
		cudaFree(*data);
		*data = NULL;
	}
	void zero(void *data, int bytes) {
		cudaMemset(data, 0, bytes);
	}
};*/

/**
 * This array wraps the data provided by a device pointer, or from a device memory allocation.
 * 
 * It refers to CUDA APIs to allocate memory from a host function call. 
 * Accessing values of the array must be done on the device. 
 */
template <typename T>
struct Array {
	T *data;
	int capacity;
	int size;

	Array() {        
		data = NULL;
		capacity = 0;
		size = 0;        
	}

	/**
	 * Wrap an input array
	 */
	Array(T *data, int n) {
		this->data = data;
		capacity = n;
		size = n;
	}

	void alloc(GpuAlloc *allocator, int new_capacity) {
        if (new_capacity > capacity) {
            this->free(allocator);
		    allocator->alloc((void**)&data, new_capacity * sizeof(T));
		    this->capacity = new_capacity;
        }
        this->size = 0;
	}

	void resize(GpuAlloc *allocator, int size) {
		if (size > capacity) {
			this->free(allocator);
			this->alloc(allocator, size);
		}
		this->size = size;
	}

	void free(GpuAlloc *allocator) {
		if (! data || capacity == 0) return;
		allocator->free((void**)&data);
        this->data = NULL;
		this->capacity = 0;
		this->size = 0;
	}

	__device__ void zero() {
        for (int i = 0; i < size; ++i) data[i] = 0;
    }

	__host__ void zero(GpuAlloc *allocator) {		
        allocator->zero((void*)data, size * sizeof(T));		
	}

	__device__ T& operator[](int i) {
		return data[i];
	}

    __device__ T operator[](int i) const {
		return data[i];
	}

	__device__ void append(T value) {
		data[size] = value;
		size++;
	}

	__device__ void clear() {
		size = 0;
	}
};

struct Query {
	__device__ virtual void operator()(int ii, int f, int fsize) = 0;
};

template <typename T>
struct Grid {
    Array<T> points;
	Array<int> neighbor_count;  // for each point, for each filter, for each cell of the filter, store how many points in that cell.

	T voxel_size;
	int filter_x, filter_y, filter_z, filter_count;

	Grid() {}

	Grid(Array<T> points, T voxel_size, int filter_x, int filter_y, int filter_z) {
        this->points = points;		
		this->voxel_size = voxel_size;
		this->filter_x = filter_x;
		this->filter_y = filter_y;
		this->filter_z = filter_z;
		this->filter_count = filter_x * filter_y * filter_z;
	}
	
	void alloc(GpuAlloc *allocator) {
		neighbor_count.resize(allocator, points.size * filter_count);
		neighbor_count.zero(allocator);
	}

	__device__ void neighbor_brute_force(int i, T x, T y, T z, int filter_x, int filter_y, int filter_z, T voxel_size,
				  			 			 Query &query) 
	{
		// Center the filter at the current point
		T xmin = x - filter_x * 0.5 * voxel_size;
		T xmax = x + filter_x * 0.5 * voxel_size;
		T ymin = y - filter_y * 0.5 * voxel_size;
		T ymax = y + filter_y * 0.5 * voxel_size;
		T zmin = z - filter_z * 0.5 * voxel_size;
		T zmax = z + filter_z * 0.5 * voxel_size;

		for (int j = 0; j < points.size; ++j) {

			T vx = points[3 * j + 0];
			T vy = points[3 * j + 1];
			T vz = points[3 * j + 2];

			if (vx < xmin || vx > xmax || vy < ymin || vy > ymax || vz < zmin || vz > zmax) continue;

			// Determine which cell
			int fx = min(filter_x - 1, (int)((vx - xmin) / voxel_size));
			int fy = min(filter_y - 1, (int)((vy - ymin) / voxel_size));
			int fz = min(filter_z - 1, (int)((vz - zmin) / voxel_size));

			int f = (fz * filter_y + fy) * filter_x + fx;

			// Good point
			query(j, f, neighbor_count[i * filter_count + f]);
		}
	}

	__device__ void build_neighbor_count(int i) {

			T x = points[3 * i + 0];
			T y = points[3 * i + 1];
			T z = points[3 * i + 2];

			// Center the filter at the current point
			T xmin = x - filter_x * 0.5 * voxel_size;
			T xmax = x + filter_x * 0.5 * voxel_size;
			T ymin = y - filter_y * 0.5 * voxel_size;
			T ymax = y + filter_y * 0.5 * voxel_size;
			T zmin = z - filter_z * 0.5 * voxel_size;
			T zmax = z + filter_z * 0.5 * voxel_size;

			for (int j = 0; j < points.size; ++j) {

                T vx = points[3 * j + 0];
                T vy = points[3 * j + 1];
                T vz = points[3 * j + 2];

                if (vx < xmin || vx > xmax || vy < ymin || vy > ymax || vz < zmin || vz > zmax) continue;

                // Determine which cell
                int fx = min(filter_x - 1, (int)((vx - xmin) / voxel_size));
                int fy = min(filter_y - 1, (int)((vy - ymin) / voxel_size));
                int fz = min(filter_z - 1, (int)((vz - zmin) / voxel_size));

                int f = (fz * filter_y + fy) * filter_x + fx;

                neighbor_count[i * filter_count + f]++;
						
			}
		
	}

	void free(GpuAlloc *allocator) {
		neighbor_count.free(allocator);
	}

};

template <typename T>
__global__ void kernelBuildNeighborCount(int batch_size, int num_points, Array< Grid<T> > grids) {
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	int b = idx / num_points;
	int i = idx % num_points;
	if (b >= batch_size) return;

	grids[b].build_neighbor_count(i);
}

template <typename T>
struct ForwardQuery : public Query {
    __device__ ForwardQuery(int i, const T *points,  const T *input, const T *filter, T *output, 
							int batch_size, int num_points, int filter_x, int filter_y, int filter_z, 
							int filter_c_in, int filter_c_out, T voxel_size) 
    : i(i), points(points), 
    input(input),
    filter(filter),
    output(output),
    batch_size(batch_size), num_points(num_points), filter_x(filter_x), filter_y(filter_y), filter_z(filter_z), 
    filter_c_in(filter_c_in), filter_c_out(filter_c_out), voxel_size(voxel_size)	
    {
		
    }

    __device__ void operator()(int ii, int f, int fsize) {            
		
		T inv_fsize = 1.0 / fsize;
		#pragma unroll
		for (int c = 0; c < filter_c_out; ++c) {
			#pragma unroll
			for (int k = 0; k < filter_c_in; ++k) {

				// Get filter weight
				T w = filter[(f * filter_c_in + k) * filter_c_out + c];

				output[i * filter_c_out + c] += w * input[ii * filter_c_in + k] * inv_fsize;
			}
		}      
    }

	int i;
    const T *points; const T *input; const T *filter; T *output; 
    int batch_size; int num_points; int filter_x; int filter_y; int filter_z; 
    int filter_c_in; int filter_c_out; float voxel_size;	
};

template <typename T>
__global__ void kernelForward(const T *points_flat, const T *input_flat, const T *filter, T *output_flat, 
							int batch_size, int num_points, int filter_x, int filter_y, int filter_z, 
							int filter_c_in, int filter_c_out, T voxel_size, 
							Array< Grid<T> > grids ) {
	
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	
	int b = idx / num_points;
	int i = idx % num_points;
	if (b >= batch_size) return;
	
	const T *points = points_flat + b * num_points * 3;       // XYZ as input
	const T *input = input_flat + b * num_points * filter_c_in;
	T *output = output_flat + b * num_points * filter_c_out;

	Grid<T> &grid = grids[b]; 

	T x = points[3 * i + 0];
	T y = points[3 * i + 1];
	T z = points[3 * i + 2];

	ForwardQuery<T> query(i, points, input, filter, output, batch_size, num_points, filter_x, filter_y, filter_z, 
					      filter_c_in, filter_c_out, voxel_size);
	
	grid.neighbor_brute_force(i, x, y, z, filter_x, filter_y, filter_z, voxel_size, query);
}

template <typename T> 
struct GradientQuery : Query {
	int j; T x, y, z;
	const T *grad_from_next_tensor;
	const T *points; const T *input; const T *filter; T *grad_input; T *grad_filter_thread_arr; 
    int batch_size; int num_points; int filter_x; int filter_y; int filter_z; int filter_count;
    int filter_c_in; int filter_c_out; float voxel_size;
	Array<int> &neighbor_count;

	__device__ GradientQuery(int j, T x, T y, T z, /* the point where we start the neighbor query */
							const T *grad_from_next_tensor, 
							const T *points, const T *input, const T *filter, T *grad_input, T *grad_filter_thread_arr,
							int batch_size, int num_points, int filter_x, int filter_y, int filter_z, 
							int filter_c_in, int filter_c_out, float voxel_size, Array<int> &neighbor_count) 
	: j(j), x(x), y(y), z(z),
	grad_from_next_tensor(grad_from_next_tensor),
    points(points), 
    input(input),
    filter(filter),
    grad_input(grad_input), grad_filter_thread_arr(grad_filter_thread_arr),
    batch_size(batch_size), num_points(num_points), filter_x(filter_x), filter_y(filter_y), filter_z(filter_z), 
    filter_c_in(filter_c_in), filter_c_out(filter_c_out), voxel_size(voxel_size), 
	neighbor_count(neighbor_count)
	{
		filter_count = filter_x * filter_y * filter_z;
	}

	__device__ void operator()(int ii, int f_ii, int fsize_ii) {
		
		// Take i as center
		T kx = points[3 * ii + 0];
		T ky = points[3 * ii + 1];
		T kz = points[3 * ii + 2];

		T xmin = kx - filter_x * 0.5 * voxel_size;
		T ymin = ky - filter_y * 0.5 * voxel_size;
		T zmin = kz - filter_z * 0.5 * voxel_size;

		// Check which cell the point pj is in w.r.t the point pi
		int fx = min(filter_x - 1, (int)((x - xmin) / voxel_size));
		int fy = min(filter_y - 1, (int)((y - ymin) / voxel_size));
		int fz = min(filter_z - 1, (int)((z - zmin) / voxel_size));

		int filter_index = (fz * filter_y + fy) * filter_x + fx;
		int count = neighbor_count[ii * filter_count + filter_index];
		if (count == 0) return; // FIXME: non-symmetric neighbor issue

		// For all types of filters
		#pragma unroll
		for (int c = 0; c < filter_c_out; ++c) {
			int out_index = ii * filter_c_out + c;
			T dL_dxi = grad_from_next_tensor[out_index];
			T dL_dxi_div_count = dL_dxi / (T)count;

			#pragma unroll
			for (int k = 0; k < filter_c_in; ++k) {
				int weight_index = (filter_index * filter_c_in + k) * filter_c_out + c;
				
				int in_index = j * filter_c_in + k;

				// Update the gradient of an input xi
				T w_as = filter[weight_index];
				grad_input[in_index] += dL_dxi_div_count * w_as;

				// Update the gradient of a filter weight
				T dxi_dw = input[in_index];
				
				atomicAdd(&grad_filter_thread_arr[weight_index], dL_dxi_div_count * dxi_dw);
			}
		}		
	}
};

template <typename T>
__global__ void kernelGradient(const T *grad_from_next_tensor_flat, 
								const T *points_flat,  const T *input_flat, const T *filter, 
								T *grad_input_flat, T *grad_filter_thread_arr, 
								int batch_size, int num_points, int filter_x, int filter_y, int filter_z, 
								int filter_c_in, int filter_c_out, T voxel_size, 
								Array< Grid<T> > grids) {
	
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	
	int b = idx / num_points;
	int j = idx % num_points;	
	if (b >= batch_size) return;
	
	const T *grad_from_next_tensor = grad_from_next_tensor_flat + b * num_points * filter_c_out;
	const T *points = points_flat + b * num_points * 3;       // XYZ as input
	const T *input = input_flat + b * num_points * filter_c_in;
	T *grad_input = grad_input_flat + b * num_points * filter_c_in;

	Grid<T> &grid = grids[b]; 
	
	T x = points[3 * j + 0];
	T y = points[3 * j + 1];
	T z = points[3 * j + 2];

	GradientQuery<T> query(j, x, y, z, grad_from_next_tensor, 
						   points, input, filter, grad_input, grad_filter_thread_arr, batch_size, num_points, filter_x, filter_y, filter_z, 
					       filter_c_in, filter_c_out, voxel_size, grid.neighbor_count);

	grid.neighbor_brute_force(j, x, y, z, filter_x, filter_y, filter_z, voxel_size, query);
}

template <typename T>
class Conv3pOp : public OpKernel {
 public:
  explicit Conv3pOp(OpKernelConstruction* context) : OpKernel(context) {
  }

  void Compute(OpKernelContext* context) override {

  	// Point tensor is of the following dimensions:
    // [ batch, num_points, 3 ]
    const Tensor& points_tensor = context->input(0);
    OP_REQUIRES(context, points_tensor.dims()==3, errors::InvalidArgument("Conv3p expects (batch_size, num_points, 3) points shape"));
    int batch_size = points_tensor.shape().dim_size(0);
    
	int device; 
	cudaGetDevice(&device);

    int num_points = points_tensor.shape().dim_size(1);
    auto points_flat = points_tensor.flat<T>();

    // Input tensor is of the following dimensions:
    const Tensor& input_tensor = context->input(1);
    OP_REQUIRES(context, input_tensor.shape().dim_size(0) == points_tensor.shape().dim_size(0), errors::InvalidArgument("Conv3p expects points and input tensor to have the same batch size"));
    OP_REQUIRES(context, input_tensor.shape().dim_size(1) == points_tensor.shape().dim_size(1), errors::InvalidArgument("Conv3p expects points and input tensor to have the same number of points"));
    int num_channels_in = input_tensor.shape().dim_size(2);
    auto input_flat = input_tensor.flat<T>();

    // Input filter is of the following dimensions:
    // [ filter_z, filter_y, filter_x, in_channels, out_channels]
    const Tensor& filter_tensor = context->input(2);
    int filter_z = filter_tensor.shape().dim_size(0);
    int filter_y = filter_tensor.shape().dim_size(1);
    int filter_x = filter_tensor.shape().dim_size(2);
    int filter_c_in = filter_tensor.shape().dim_size(3);
    int filter_c_out = filter_tensor.shape().dim_size(4);
    OP_REQUIRES(context, filter_c_in == num_channels_in, errors::InvalidArgument("Conv3p expects filter channels to be matched with input channels"));
    
    auto filter_flat = filter_tensor.flat<T>();    

    const Tensor& voxel_tensor = context->input(3);
    OP_REQUIRES(context, voxel_tensor.shape().dim_size(0) == 1, errors::InvalidArgument("Conv3p expects voxel tensor to have dimension 1."));
    const T *voxel_flat = &(voxel_tensor.flat<T>()(0));
	T voxel_size;	
	cudaMemcpy(&voxel_size, voxel_flat, sizeof(T), cudaMemcpyDeviceToHost);
	
    // Create output tensor
    Tensor* output_tensor = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(0, TensorShape{batch_size, num_points, filter_c_out},
                                                     &output_tensor));
    auto output_flat = output_tensor->flat<T>();
    cudaMemset(&(output_flat(0)), 0, sizeof(T) * batch_size * num_points * filter_c_out);

    GpuAlloc allocator(context);

	//timer t1;
	std::vector<Grid<T> > cpu_grids(batch_size);
	for (int b = 0; b < batch_size; ++b) {
		const T *points = &(points_flat(0)) + b * num_points * 3;
		cpu_grids[b] = Grid<T>(Array<T>((T*)points, num_points), voxel_size, filter_x, filter_y, filter_z);
        cpu_grids[b].alloc(&allocator);
    }	
	Array< Grid<T> > grids;
	grids.resize(&allocator, batch_size);
	cudaMemcpy(grids.data, cpu_grids.data(), sizeof(Grid<T>) * batch_size, cudaMemcpyHostToDevice);
	//std::cout << "device " << device << "t1: " << t1.getTimePassed() << std::endl;

	//timer t2;
	{
		dim3 blocks(divUp(batch_size * num_points, kThreadsPerBlock));
		dim3 threads(kThreadsPerBlock);
		kernelBuildNeighborCount<<<blocks, threads>>>(batch_size, num_points, grids);
	}
	//cudaDeviceSynchronize();
	//std::cout << "device " << device << "t2: " << t2.getTimePassed() << std::endl;
	
	//timer t3;
	{
	// Now parallelize over all batches and all points
	dim3 blocks(divUp(batch_size * num_points, kThreadsPerBlock));
	dim3 threads(kThreadsPerBlock);
	
	kernelForward<<<blocks, threads>>>(&(points_flat(0)), &(input_flat(0)), &(filter_flat(0)), &(output_flat(0)), 
										batch_size, num_points, filter_x, filter_y, filter_z, filter_c_in, filter_c_out, voxel_size, 
										grids);
	}
	//cudaDeviceSynchronize();
	//std::cout << "device " << device << "t3: " << t3.getTimePassed() << std::endl;
	
	//timer t4;
	for (int b = 0; b < batch_size; ++b) {
		cpu_grids[b].free(&allocator);
	}
	grids.free(&allocator);
	//cudaDeviceSynchronize();
	//std::cout << "device " << device << "t4: " << t4.getTimePassed() << std::endl;

  }
};

#define REGISTER_GPU_KERNEL(T)                                  \
  REGISTER_KERNEL_BUILDER(                                      \
      Name("Conv3p").Device(DEVICE_GPU).TypeConstraint<T>("T"), \
      Conv3pOp<T>);
TF_CALL_float(REGISTER_GPU_KERNEL);
TF_CALL_double(REGISTER_GPU_KERNEL);
#undef REGISTER_GPU_KERNEL

////////////////////////////////////////////////////////////////////////////////
template <typename T>
class Conv3pGradOp : public OpKernel {
 public:
  explicit Conv3pGradOp(OpKernelConstruction* context) : OpKernel(context) {
  }

  void Compute(OpKernelContext* context) override {

    /**1. Setting things up **/
    // get the gradient tensor (from the later tensor)
    const Tensor& grad_from_next_tensor = context->input(0);
    auto grad_from_next_tensor_flat = grad_from_next_tensor.flat<T>();

    // get other inputs
    const Tensor& points_tensor = context->input(1);
    auto points_flat = points_tensor.flat<T>();

    const Tensor& input_tensor = context->input(2);
    auto input_flat = input_tensor.flat<T>();

    // infos about the inputs
    int batch_size = points_tensor.shape().dim_size(0);
    int num_points = points_tensor.shape().dim_size(1);

    // get the filters tensor (which include weights)
    const Tensor& filter_tensor = context->input(3);
    auto filter_flat = filter_tensor.flat<T>();
    
    const Tensor& voxel_tensor = context->input(4);
    OP_REQUIRES(context, voxel_tensor.shape().dim_size(0) == 1, errors::InvalidArgument("Conv3p expects voxel tensor to have dimension 1."));
    const T *voxel_flat = &(voxel_tensor.flat<T>()(0));
	T voxel_size;
	cudaMemcpy(&voxel_size, voxel_flat, sizeof(T), cudaMemcpyDeviceToHost);

    // dimensional infos for the filters tensor
    int filter_z = filter_tensor.shape().dim_size(0);
    int filter_y = filter_tensor.shape().dim_size(1);
    int filter_x = filter_tensor.shape().dim_size(2);
    int filter_c_in = filter_tensor.shape().dim_size(3);
    int filter_c_out = filter_tensor.shape().dim_size(4);
    int num_weights = filter_z * filter_y * filter_x * filter_c_in * filter_c_out;
    
    // Get shape of the grad tensors
    TensorShape grad_input_shape = input_tensor.shape();
    TensorShape grad_filter_shape = filter_tensor.shape();

    // Create the output tensor for the gradient of the inputs
    // How many points * number of input channel = how many gradients.
    Tensor* grad_input = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(0, grad_input_shape, &grad_input));
    auto grad_input_flat = grad_input->flat<T>();
    cudaMemset(&(grad_input_flat(0)), 0, sizeof(T) * input_tensor.shape().dim_size(0)*input_tensor.shape().dim_size(1)*input_tensor.shape().dim_size(2));

    // a) First we need to check if the size of the grad tensor and the number of points are compitable.
    OP_REQUIRES(context, grad_from_next_tensor.shape().dim_size(0) == batch_size, errors::InvalidArgument("backprop grad tensor has wrong size for dim 0"));
    OP_REQUIRES(context, grad_from_next_tensor.shape().dim_size(1) == num_points, errors::InvalidArgument("backprop grad tensor has wrong size for dim 1"));
    OP_REQUIRES(context, grad_from_next_tensor.shape().dim_size(2) == filter_c_out, errors::InvalidArgument("backprop grad tensor has wrong size for dim 2"));

    Tensor* grad_filter = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(1, grad_filter_shape, &grad_filter));
    auto grad_filter_flat = grad_filter->flat<T>();
    cudaMemset(&(grad_filter_flat(0)), 0, sizeof(T) * num_weights);
    
    GpuAlloc allocator(context);

	std::vector<Grid<T> > cpu_grids(batch_size);
	for (int b = 0; b < batch_size; ++b) {
		const T *points = &(points_flat(0)) + b * num_points * 3;
		cpu_grids[b] = Grid<T>(Array<T>((T*)points, num_points), voxel_size, filter_x, filter_y, filter_z);
        cpu_grids[b].alloc(&allocator);
    }	
	Array< Grid<T> > grids;
	grids.resize(&allocator, batch_size);
	cudaMemcpy(grids.data, cpu_grids.data(), sizeof(Grid<T>) * batch_size, cudaMemcpyHostToDevice);

	{
		dim3 blocks(divUp(batch_size * num_points, kThreadsPerBlock));
		dim3 threads(kThreadsPerBlock);
		kernelBuildNeighborCount<<<blocks, threads>>>(batch_size, num_points, grids);
	}

	// Now parallelize over all batches and all points
	{
		dim3 blocks(divUp(batch_size * num_points, kThreadsPerBlock));
		dim3 threads(kThreadsPerBlock);
		kernelGradient<<<blocks, threads>>>(&(grad_from_next_tensor_flat(0)), &(points_flat(0)), &(input_flat(0)), &(filter_flat(0)),
											&(grad_input_flat(0)), &(grad_filter_flat(0)), // grad_filter_thread_arr.data,
											batch_size, num_points, filter_x, filter_y, filter_z, filter_c_in, filter_c_out, voxel_size, 
											grids);
	}
	
	for (int b = 0; b < batch_size; ++b) {
		cpu_grids[b].free(&allocator);
	}
	grids.free(&allocator);
	
    /**2. Compute gradient of the input**/
    // dL_dxj = (sum over xi that have xj as a neighbor)dL/dxi * w_as
    // w_as is the weight associated xi and xj. w_as = 0 if xj not contribute to xi.
    // we take avantaged of if xj is a neighbor of xi, then xi is also a neighbor of xi
    // or that we have symmetric neighborhood, then
    // dL_dxj = (sum over all xi that are neightbor of xj)dL/dxi * w_as

    // /**3. Compute gradient of the filter**/
    // Reminder: grad_from_next_tensor contain dL/dx, with xi is an output of the forward pass.
    // Reminder: we can calculate dL/dw = (sum over i)dL/dxi * dx/dw, with w is a weight that connect x to the next layer
    // =>
    // b) We do this by go through all the points and accumulate the gradients
    // Create the output tensor for the gradient of the filter

  }
};

#define REGISTER_GPU_KERNEL(T)                                        \
  REGISTER_KERNEL_BUILDER(                                            \
      Name("Conv3pGrad").Device(DEVICE_GPU).TypeConstraint<T>("T"),   \
      Conv3pGradOp<T>);
TF_CALL_float(REGISTER_GPU_KERNEL);
TF_CALL_double(REGISTER_GPU_KERNEL);
#undef REGISTER_GPU_KERNEL
