#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "tensorflow/core/framework/common_shape_fns.h"

#include <vector>
#include <cstdio>
#include <ctime>
#include <cstring>
#include <cstdlib>
#include <cmath>
#include <iostream>

#include <chrono>
#include <thread>
#include <type_traits>
#include <thread>

#ifdef CONV_OPENMP
#include <omp.h>
#endif

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
    long /*milliseconds*/ getTimePassed()
    {
        // get the new time
        auto end = clock::now();

        // return the difference of the times
        return (end - start).count();
    }

private:
    time_point_type start;
};

template <class T>
struct CpuAlloc {
	static void alloc(T **data, int size) {
		*data = new T[size];
	}
	static void free(T **data) {
		delete [] *data;
		*data = NULL;
	}
	static void zero(T *data, int size) {
		memset(data, 0, sizeof(T) * size);
	}
};

/*
template <typename T>
struct GpuAlloc {
	static void alloc(T **data, int size) {
		*data = (T*)cudaMalloc(sizeof(T) * size);
	}
	static void free(T **data) {
		cudaFree(*data);
		*data = NULL;
	}
	static void zero(T **data, int size) {
		cudaMemset(*data, 0, sizeof(T) * size);
	}
};*/

template <template<class> class Alloc, typename T>
struct Array {
	T *data;
	int capacity;
	int size;

	Array() {
		data = NULL;
		capacity = 0;
		size = 0;
	}

    Array(const Array &a) {
        *this = a;
        std::cout << "Unexpected array copy" << std::endl;
    }

	/**
	 * Wrap an input array
	 */
	Array(T *data, int n) {
		this->data = data;
		capacity = n;
		size = n;
	}

	void alloc(int new_capacity) {
        if (new_capacity > capacity) {
            this->free();
		    Alloc<T>::alloc(&data, new_capacity);
		    this->capacity = new_capacity;
        }
        this->size = 0;
	}

	void resize(int size) {
		if (size > capacity) {
			this->free();
			this->alloc(size);
		}
		this->size = size;
	}

	void free() {
		if (! data || capacity == 0) return;
		Alloc<T>::free(&data);
		this->capacity = 0;
		this->size = 0;
	}

	void zero() {
		Alloc<T>::zero(data, capacity);
	}

	T& operator[](int i) {
		return data[i];
	}

    T operator[](int i) const {
		return data[i];
	}

	void append(T value) {
		data[size] = value;
		size++;
	}

	void clear() {
		size = 0;
	}
};

template <template<class> class Alloc, typename T>
struct Grid {
	Array< Alloc, int> cell;			// cell index of each point

	Array< Alloc, int> cell_sorted;	    // index of the points in cell order
	Array< Alloc, int> cell_count;		// number of points in each cell
	Array< Alloc, int> cell_start;

    Array<Alloc, T> points;

	int dim_x, dim_y, dim_z;
	T radius;
	T vmin_x, vmin_y, vmin_z, vmax_x, vmax_y, vmax_z;

    Grid(const Grid &g) {
        *this = g;
        std::cout << "Unexpected grid copy" << std::endl;
    }

	Grid(Array<Alloc, T> points, T radius) {
		// Assume data is normalized in [-1, 1]
        this->points = points;		
		this->radius = radius;

		// Find bounding box 
		vmin_x = vmin_y = vmin_z = 1e6f;
		vmax_x = vmax_y = vmax_z = -1e6f;
		for (int i = 0; i < points.size; ++i) {
			T x = points[3 * i + 0];
			T y = points[3 * i + 1];
			T z = points[3 * i + 2];

			vmin_x = std::min(vmin_x, x);
			vmin_y = std::min(vmin_y, y);
			vmin_z = std::min(vmin_z, z);

			vmax_x = std::max(vmax_x, x);
			vmax_y = std::max(vmax_y, y);
			vmax_z = std::max(vmax_z, z);
		}
		
		this->dim_x = (int)((vmax_x - vmin_x) / radius) + 2; // padding to avoid numerical out of bound
		this->dim_y = (int)((vmax_y - vmin_y) / radius) + 2;
		this->dim_z = (int)((vmax_z - vmin_z) / radius) + 2;

        int num_cells = dim_x * dim_y * dim_z;

		cell.resize(points.size);
		cell.zero();
		for (int i = 0; i < points.size; ++i) {
			T x = points[3 * i + 0];
			T y = points[3 * i + 1];
			T z = points[3 * i + 2];

			int cx = (int)((x - vmin_x) / radius);
			int cy = (int)((y - vmin_y) / radius);
			int cz = (int)((z - vmin_z) / radius);

			cell[i] = (cz * dim_y + cy) * dim_x + cx;
		}

		cell_count.resize(num_cells);
		cell_count.zero();
		for (int i = 0; i < points.size; ++i) {
			cell_count[cell[i]]++;
		}

		cell_start.resize(num_cells + 1);
		cell_start.zero();
		cell_start[0] = 0;
		for (int i = 1; i <= num_cells; ++i) {
			cell_start[i] = cell_start[i - 1] + cell_count[i - 1];
		}

        // Store point index in an order that allows query all points in a cell quickly
        //timer begin_timer_store_index;
        Array< Alloc, int> tmp_cell_count;
        tmp_cell_count.resize(num_cells);
		tmp_cell_count.zero();
		cell_sorted.resize(points.size);
        for (int i = 0; i < points.size; ++i) {
            int k = cell[i];
            int h = tmp_cell_count[k];
            int offset = cell_start[k];
            cell_sorted[offset + h] = i;
            tmp_cell_count[k]++;
        }
        tmp_cell_count.free();
        //time_store_index += begin_timer_store_index.getTimePassed();
	}

	/**
	 * Return the index of all points that fall within the filter centered at a query point
	 */
	void neighbor(T x, T y, T z, int filter_x, int filter_y, int filter_z, T voxel_size,
				  Array<Alloc, int> &point_index, Array<Alloc, int> &filter_cell, Array<Alloc, int> &filter_cell_count) {
		// Center the filter at the current point
		T xmin = x - filter_x * 0.5 * voxel_size;
		T xmax = x + filter_x * 0.5 * voxel_size;
		T ymin = y - filter_y * 0.5 * voxel_size;
		T ymax = y + filter_y * 0.5 * voxel_size;
		T zmin = z - filter_z * 0.5 * voxel_size;
		T zmax = z + filter_z * 0.5 * voxel_size;

		int nx = (int)((filter_x + 1) * 0.5);
		int ny = (int)((filter_y + 1) * 0.5);
		int nz = (int)((filter_z + 1) * 0.5);

		int center_x = (int)((x - vmin_x) / radius);
		int center_y = (int)((y - vmin_y) / radius);
		int center_z = (int)((z - vmin_z) / radius);

		point_index.clear();
		filter_cell.clear();
		filter_cell_count.zero();

		// We only need to check cells that intersect with this filter
		for (int oz = -nz; oz <= nz; ++oz) {
			for (int oy = -ny; oy <= ny; ++oy) {
				for (int ox = -nx; ox <= nx; ++ox) {
					int cx = center_x + ox;
					int cy = center_y + oy;
					int cz = center_z + oz;
                    if (cx < 0 || cx >= dim_x || cy < 0 || cy >= dim_y || cz < 0 || cz >= dim_z) continue;
					int cidx = (cz * dim_y + cy) * dim_x + cx;

					for (int k = cell_start[cidx]; k < cell_start[cidx + 1]; ++k) {

						int i = cell_sorted[k];

						T vx = points[3 * i + 0];
						T vy = points[3 * i + 1];
						T vz = points[3 * i + 2];

						if (vx < xmin || vx > xmax || vy < ymin || vy > ymax || vz < zmin || vz > zmax) continue;

						// Determine which cell
						int fx = std::min(filter_x - 1, (int)((vx - xmin) / voxel_size));
						int fy = std::min(filter_y - 1, (int)((vy - ymin) / voxel_size));
						int fz = std::min(filter_z - 1, (int)((vz - zmin) / voxel_size));

						int f = (fz * filter_y + fy) * filter_x + fx;

						// Good point
						point_index.append(i);
						filter_cell.append(f);
						filter_cell_count[f]++;

					}
				}
			}
		}
	}

    /**
     * For each filter cell, count total points within it.
     */
	void neighbor_count(T x, T y, T z, int point_index, int filter_x, int filter_y, int filter_z, T voxel_size,
						Array<Alloc, int> &count) {
		// Center the filter at the current point
		T xmin = x - filter_x * 0.5 * voxel_size;
		T xmax = x + filter_x * 0.5 * voxel_size;
		T ymin = y - filter_y * 0.5 * voxel_size;
		T ymax = y + filter_y * 0.5 * voxel_size;
		T zmin = z - filter_z * 0.5 * voxel_size;
		T zmax = z + filter_z * 0.5 * voxel_size;
		int filter_count = filter_z * filter_y * filter_x;

		int nx = (int)((filter_x + 1) * 0.5);
		int ny = (int)((filter_y + 1) * 0.5);
		int nz = (int)((filter_z + 1) * 0.5);

		int center_x = (int)((x - vmin_x) / radius);
		int center_y = (int)((y - vmin_y) / radius);
		int center_z = (int)((z - vmin_z) / radius);

		// We only need to check cells that intersect with this filter
		for (int oz = -nz; oz <= nz; ++oz) {
			for (int oy = -ny; oy <= ny; ++oy) {
				for (int ox = -nx; ox <= nx; ++ox) {
					int cx = center_x + ox;
					int cy = center_y + oy;
					int cz = center_z + oz;
					if (cx < 0 || cx >= dim_x || cy < 0 || cy >= dim_y || cz < 0 || cz >= dim_z) continue;
					int cidx = (cz * dim_y + cy) * dim_x + cx;

					for (int k = cell_start[cidx]; k < cell_start[cidx + 1]; ++k) {
						int i = cell_sorted[k];

						T vx = points[3 * i + 0];
						T vy = points[3 * i + 1];
						T vz = points[3 * i + 2];

						if (vx < xmin || vx > xmax || vy < ymin || vy > ymax || vz < zmin || vz > zmax) continue;

						// Determine which cell
						int fx = std::min(filter_x - 1, (int)((vx - xmin) / voxel_size));
						int fy = std::min(filter_y - 1, (int)((vy - ymin) / voxel_size));
						int fz = std::min(filter_z - 1, (int)((vz - zmin) / voxel_size));

						int f = (fz * filter_y + fy) * filter_x + fx;

						count[point_index * filter_count + f]++;
					}
				}
			}
		}
	}

	void neighbor_count(int filter_x, int filter_y, int filter_z, T voxel_size,
						Array<Alloc, int> &count) {
		count.zero();
		for (int i = 0; i < points.size; ++i) {
			T x = points[3 * i + 0];
			T y = points[3 * i + 1];
			T z = points[3 * i + 2];

			neighbor_count(x, y, z, i, filter_x, filter_y, filter_z, voxel_size, count);
		}
	}


	~Grid() {
		cell.free();
		cell_sorted.free();
		cell_count.free();
		cell_start.free();
	}
};

using namespace tensorflow;

typedef Eigen::ThreadPoolDevice CPUDevice;
typedef Eigen::GpuDevice GPUDevice;

template <typename Device, typename T>
class Conv3pOp : public OpKernel {
 public:
  explicit Conv3pOp(OpKernelConstruction* context) : OpKernel(context) {
  }

  void Compute(OpKernelContext* context) override {
  	//long build_grid_time = 0;
    //long whole_time = 0;
  	//long convolution_time = 0;

  	//timer begin_whole_time;
    // Point tensor is of the following dimensions:
    // [ batch, num_points, 3 ]
    const Tensor& points_tensor = context->input(0);
    OP_REQUIRES(context, points_tensor.dims()==3, errors::InvalidArgument("Conv3p expects (batch_size, num_points, 3) points shape"));
    int batch_size = points_tensor.shape().dim_size(0);
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
    int filter_count = filter_x * filter_y * filter_z;

    auto filter_flat = filter_tensor.flat<T>();
    const T *filter = &(filter_flat(0));

    const Tensor& voxel_tensor = context->input(3);
    OP_REQUIRES(context, voxel_tensor.shape().dim_size(0) == 1, errors::InvalidArgument("Conv3p expects voxel tensor to have dimension 1."));
    T voxel_size = voxel_tensor.flat<T>()(0);

    // Create output tensor
    Tensor* output_tensor = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(0, TensorShape{batch_size, num_points, filter_c_out},
                                                     &output_tensor));
    auto output_flat = output_tensor->flat<T>();
    memset(&(output_flat(0)), 0, sizeof(T) * batch_size * num_points * filter_c_out);

#ifdef CONV_OPENMP
#pragma omp parallel for
#endif
    for (int b = 0; b < batch_size; ++b) {
        const T *points = &(points_flat(0)) + b * num_points * 3;       // XYZ as input
        const T *input = &(input_flat(0)) + b * num_points * filter_c_in;
        T *output = &(output_flat(0)) + b * num_points * filter_c_out;

		// Build grid
		//timer begin_build_grid;
		Grid<CpuAlloc, T> grid(Array<CpuAlloc, T>((T*)points, num_points), voxel_size);
		Array<CpuAlloc, int> point_index;
		Array<CpuAlloc, int> filter_cell;
		Array<CpuAlloc, int> filter_cell_count;
		point_index.alloc(num_points);
		filter_cell.alloc(num_points);
		filter_cell_count.resize(filter_count);
		//build_grid_time += begin_build_grid.getTimePassed();

		//timer begin_convolution;
        for (int i = 0; i < num_points; ++i) {
            T x = points[3 * i + 0];
            T y = points[3 * i + 1];
            T z = points[3 * i + 2];

			grid.neighbor(x, y, z, filter_x, filter_y, filter_z, voxel_size, point_index, filter_cell, filter_cell_count);

			for (int j = 0; j < point_index.size; ++j) {
				int ii = point_index[j];
				int f = filter_cell[j];
				int fsize = filter_cell_count[f];

				// Convolution
				for (int c = 0; c < filter_c_out; ++c) {
					for (int k = 0; k < filter_c_in; ++k) {

						// Get filter weight
						T w = filter[(f * filter_c_in + k) * filter_c_out + c];

						output[i * filter_c_out + c] += w * input[ii * filter_c_in + k] / (T)fsize;
					}
				}

            }
        }

        //convolution_time += begin_convolution.getTimePassed();

        point_index.free();
        filter_cell.free();
        filter_cell_count.free();
    }
    //whole_time = begin_whole_time.getTimePassed();
    //printf("time_for_convolution_fowardpass: %f\n", (double)convolution_time/1e9);
    //printf("time_for_build_grid_fowardpass: %f\n", (double)build_grid_time/1e9);
    //printf("whole_time_forwardpass: %f\n", (double)whole_time/1e9);
  }
};
#define REGISTER_CPU_KERNEL(T)                                  \
  REGISTER_KERNEL_BUILDER(                                      \
      Name("Conv3p").Device(DEVICE_CPU).TypeConstraint<T>("T"), \
      Conv3pOp<CPUDevice, T>);
TF_CALL_float(REGISTER_CPU_KERNEL);
TF_CALL_double(REGISTER_CPU_KERNEL);
#undef REGISTER_CPU_KERNEL

////////////////////////////////////////////////////////////////////////////////
template <typename Device, typename T>
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
    const T *filter_arr = &(filter_flat(0));

    const Tensor& voxel_tensor = context->input(4);
    OP_REQUIRES(context, voxel_tensor.shape().dim_size(0) == 1, errors::InvalidArgument("Conv3p expects voxel tensor to have dimension 1."));
    T voxel_size = voxel_tensor.flat<T>()(0);

    // dimensional infos for the filters tensor
    int filter_z = filter_tensor.shape().dim_size(0);
    int filter_y = filter_tensor.shape().dim_size(1);
    int filter_x = filter_tensor.shape().dim_size(2);
    int filter_c_in = filter_tensor.shape().dim_size(3);
    int filter_c_out = filter_tensor.shape().dim_size(4);
    int n_weights = filter_z * filter_y * filter_x * filter_c_in * filter_c_out;
    int filter_count = filter_x * filter_y * filter_z;

    // Get shape of the grad tensors
    TensorShape grad_input_shape = input_tensor.shape();
    TensorShape grad_filter_shape = filter_tensor.shape();

    // Create the output tensor for the gradient of the inputs
    // How many points * number of input channel = how many gradients.
    Tensor* grad_input = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(0, grad_input_shape, &grad_input));
    auto grad_input_flat = grad_input->flat<T>();
    memset(&(grad_input_flat(0)), 0, sizeof(T) * input_tensor.shape().dim_size(0)*input_tensor.shape().dim_size(1)*input_tensor.shape().dim_size(2));

    // a) First we need to check if the size of the grad tensor and the number of points are compitable.
    OP_REQUIRES(context, grad_from_next_tensor.shape().dim_size(0) == batch_size, errors::InvalidArgument("backprop grad tensor has wrong size for dim 0"));
    OP_REQUIRES(context, grad_from_next_tensor.shape().dim_size(1) == num_points, errors::InvalidArgument("backprop grad tensor has wrong size for dim 1"));
    OP_REQUIRES(context, grad_from_next_tensor.shape().dim_size(2) == filter_c_out, errors::InvalidArgument("backprop grad tensor has wrong size for dim 2"));

    Tensor* grad_filter = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(1, grad_filter_shape, &grad_filter));
    auto grad_filter_flat = grad_filter->flat<T>();
    memset(&(grad_filter_flat(0)), 0, sizeof(T) * n_weights);
    T *grad_filter_arr = &(grad_filter_flat(0));

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

    // timer begin_backprop;
#ifndef CONV_OPENMP
    T *grad_filter_thread_arr = grad_filter_arr;
#else
    int num_threads = std::thread::hardware_concurrency(); // C++11

    Array<CpuAlloc, T> grad_filter_thread_arr;
    grad_filter_thread_arr.resize(n_weights * num_threads);
    grad_filter_thread_arr.zero();

    // force fixed number of threads
    omp_set_dynamic(0);
    omp_set_num_threads(num_threads);
#pragma omp parallel for
#endif
    for (int b = 0; b < batch_size; ++b) {
        const T *points_arr = &(points_flat(0)) + b * num_points * 3;       // XYZ as input
        const T *input_arr = &(input_flat(0)) + b * num_points * filter_c_in;
        const T *grad_from_next_tensor_arr = &(grad_from_next_tensor_flat(0)) + b * num_points * filter_c_out;
        T *grad_input_arr = &(grad_input_flat(0)) + b * num_points * filter_c_in;

		// Build grid
		Grid<CpuAlloc, T> grid(Array<CpuAlloc, T>((T*)points_arr, num_points), voxel_size);
		Array<CpuAlloc, int> point_index;
		Array<CpuAlloc, int> filter_cell;
		Array<CpuAlloc, int> filter_cell_count;
		point_index.alloc(num_points);
		filter_cell.alloc(num_points);
		filter_cell_count.resize(filter_count);

	    Array<CpuAlloc, int> neighbor_count;
	    neighbor_count.resize(num_points * filter_count);
		grid.neighbor_count(filter_x, filter_y, filter_z, voxel_size, neighbor_count);

#ifndef CONV_OPENMP
        const int tid = 0;
#else
        int tid = omp_get_thread_num();
#endif

        for (int j = 0; j < num_points; ++j) {
            T x = points_arr[3 * j + 0];
            T y = points_arr[3 * j + 1];
            T z = points_arr[3 * j + 2];

			grid.neighbor(x, y, z, filter_x, filter_y, filter_z, voxel_size, point_index, filter_cell, filter_cell_count);

			for (int i = 0; i < point_index.size; ++i) {
				int ii = point_index[i];

				// Take i as center
				T kx = points_arr[3 * ii + 0];
				T ky = points_arr[3 * ii + 1];
				T kz = points_arr[3 * ii + 2];

				T xmin = kx - filter_x * 0.5 * voxel_size;
				T ymin = ky - filter_y * 0.5 * voxel_size;
				T zmin = kz - filter_z * 0.5 * voxel_size;

				// Check which cell the point pj is in w.r.t the point pi
				int fx = std::min(filter_x - 1, (int)((x - xmin) / voxel_size));
				int fy = std::min(filter_y - 1, (int)((y - ymin) / voxel_size));
				int fz = std::min(filter_z - 1, (int)((z - zmin) / voxel_size));

				assert(fx >= 0 && fx < filter_x && fy >= 0 && fy < filter_y && fz >= 0 && fz < filter_z);

				int filter_index = (fz * filter_y + fy) * filter_x + fx;
				int count = neighbor_count[ii * filter_count + filter_index];
				if (count == 0) continue; // FIXME: non-symmetric neighbor issue

				// For all types of filters
				for (int c = 0; c < filter_c_out; ++c) {
					for (int k = 0; k < filter_c_in; ++k) {
						int weight_index = (filter_index * filter_c_in + k) * filter_c_out + c;
						int out_index = ii * filter_c_out + c;
						int in_index = j * filter_c_in + k;

						T dL_dxi = grad_from_next_tensor_arr[out_index];

                        // Update the gradient of an input xi
						T w_as = filter_arr[weight_index];
						grad_input_arr[in_index] += dL_dxi * w_as / (T)count;

                        // Update the gradient of a filter weight
                        T dxi_dw = input_arr[in_index];
                        grad_filter_thread_arr[tid * n_weights + weight_index] += dL_dxi * dxi_dw / (T)count;
					}
				}

            }
        }

         point_index.free();
         filter_cell.free();
         filter_cell_count.free();
         neighbor_count.free();
     }

#ifdef CONV_OPENMP
    for (int tid = 0; tid < num_threads; ++tid) {
        for (int weight_index = 0; weight_index < n_weights; ++weight_index) {
            grad_filter_arr[weight_index] += grad_filter_thread_arr[tid * n_weights + weight_index];
        }
    }
    grad_filter_thread_arr.free();
#endif

     //long time_backprop = begin_backprop.getTimePassed();
     //printf("time backprop : %f\n", (double)time_backprop / 1e9);
  }
};
#define REGISTER_CPU_KERNEL(T)                                        \
  REGISTER_KERNEL_BUILDER(                                            \
      Name("Conv3pGrad").Device(DEVICE_CPU).TypeConstraint<T>("T"),   \
      Conv3pGradOp<CPUDevice, T>);
TF_CALL_float(REGISTER_CPU_KERNEL);
TF_CALL_double(REGISTER_CPU_KERNEL);
#undef REGISTER_CPU_KERNEL
