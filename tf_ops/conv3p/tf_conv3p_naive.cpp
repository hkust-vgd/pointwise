#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "tensorflow/core/framework/common_shape_fns.h"

#include <cuda_runtime.h>

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

using namespace tensorflow;

typedef Eigen::ThreadPoolDevice CPUDevice;
typedef Eigen::GpuDevice GPUDevice;

REGISTER_OP("Conv3p")
    .Attr("T: {float, double}")
    .Input("points: T")
    .Input("input: T")
    .Input("filter: T")
    .Input("voxel_size: T")
    .Output("output: T")
    .Doc(R"doc(
Computes a 3-D convolution given a point cloud `input` and `filter` tensors.
The filter is applied at each point.
The filter size unit is voxel.
points: Shape `[batch, in_length, 3]`. Position of the point cloud.
input: Shape `[batch, in_length, in_channels]`. Other channels.
filter: Shape `[filter_depth, filter_height, filter_width, in_channels,
  out_channels]`. `in_channels` must match between `input` and `filter`.
voxel_size: size of a voxel to determine actual filter size w.r.t. the point cloud.
)doc");

REGISTER_OP("Conv3pGrad")
    .Attr("T: {float, double}")
    .Input("grad_from_next: T")
    .Input("points: T")
    .Input("input: T")
    .Input("filter: T")
    .Input("voxel_size: T")
    .Output("grad_input: T")
    .Output("grad_filter: T")
    .Doc(R"doc(
Computes the gradient of a 3-D point cloud convolution with respect to the input and weights
)doc");

template <typename Device, typename T>
struct Convolution {};

template <typename T>
struct Convolution<CPUDevice, T> {
    
    void getNeighbor(const T *points, int n, T x, T y, T z, int filter_x, int filter_y, int filter_z, float voxel_size,
                     std::vector< std::vector<int> > &neighbors) {
        // Center the filter at the current point
        T xmin = x - filter_x * 0.5 * voxel_size;
        T xmax = x + filter_x * 0.5 * voxel_size;
        T ymin = y - filter_y * 0.5 * voxel_size;
        T ymax = y + filter_y * 0.5 * voxel_size;
        T zmin = z - filter_z * 0.5 * voxel_size;
        T zmax = z + filter_z * 0.5 * voxel_size;

        // Brute force for now
        neighbors.resize(filter_x * filter_y * filter_z);
        for (int i = 0; i < n; ++i) {
            T vx = points[3 * i + 0];
            T vy = points[3 * i + 1];
            T vz = points[3 * i + 2];

            if (vx < xmin || vx >= xmax || vy < ymin || vy >= ymax || vz < zmin || vz >= zmax) continue;

            // determine which cell
            int fx = std::min(filter_x - 1, (int)((vx - xmin) / voxel_size));
            int fy = std::min(filter_y - 1, (int)((vy - ymin) / voxel_size));
            int fz = std::min(filter_z - 1, (int)((vz - zmin) / voxel_size));

            int idx = (fz * filter_y + fy) * filter_x + fx;

            neighbors[idx].push_back(i);
        }
        //std::cout << "Neighbor size: " << neighbors.size() << std::endl;
    }
    
    void getNeighborCount(const T *points, int n, int filter_x, int filter_y, int filter_z, float voxel_size,
                          std::vector< int > &count) {

        int filter_count = filter_x * filter_y * filter_z;
        count.resize(filter_count * n);
        for (int i = 0; i < count.size(); ++i) count[i] = 0;

        for (int j = 0; j < n; ++j) {
            T x = points[3 * j + 0];
            T y = points[3 * j + 1];
            T z = points[3 * j + 2];

            // Center the filter at the current point
            T xmin = x - filter_x * 0.5 * voxel_size;
            T xmax = x + filter_x * 0.5 * voxel_size;
            T ymin = y - filter_y * 0.5 * voxel_size;
            T ymax = y + filter_y * 0.5 * voxel_size;
            T zmin = z - filter_z * 0.5 * voxel_size;
            T zmax = z + filter_z * 0.5 * voxel_size;

            // Brute force for now
            for (int i = 0; i < n; ++i) {
                T vx = points[3 * i + 0];
                T vy = points[3 * i + 1];
                T vz = points[3 * i + 2];

                if (vx < xmin || vx >= xmax || vy < ymin || vy >= ymax || vz < zmin || vz >= zmax) continue;

                // determine which cell
                int fx = std::min(filter_x - 1, (int)((vx - xmin) / voxel_size));
                int fy = std::min(filter_y - 1, (int)((vy - ymin) / voxel_size));
                int fz = std::min(filter_z - 1, (int)((vz - zmin) / voxel_size));

                int idx = (fz * filter_y + fy) * filter_x + fx;

                count[j * filter_count + idx]++;
            }
        }
    }

    void forward(const T *points_flat,  const T *input_flat, const T *filter, T *output_flat, 
                 int batch_size, int num_points, int filter_x, int filter_y, int filter_z, int filter_c_in, int filter_c_out, float voxel_size) 
    {
        #ifdef CONV_OPENMP
        #pragma omp parallel for
        #endif
        for (int b = 0; b < batch_size; ++b) {
            const T *points = points_flat + b * num_points * 3;       // XYZ as input
            const T *input = input_flat + b * num_points * filter_c_in;
            T *output = output_flat + b * num_points * filter_c_out;
            
            for (int i = 0; i < num_points; ++i) {
                T x = points[3 * i + 0];
                T y = points[3 * i + 1];
                T z = points[3 * i + 2];
    
                std::vector< std::vector<int> > neighbors;
                getNeighbor(points, num_points, x, y, z, filter_x, filter_y, filter_z, voxel_size, neighbors);
    
                for (int f = 0; f < filter_x * filter_y * filter_z; ++f) {
    
                    for (int j = 0; j < neighbors[f].size(); ++j) {
                        int ii = neighbors[f][j];
    
                        // Convolution
                        for (int c = 0; c < filter_c_out; ++c) {
                            for (int k = 0; k < filter_c_in; ++k) {
    
                                // Get filter weight
                                T w = filter[(f * filter_c_in + k) * filter_c_out + c];
    
                                output[i * filter_c_out + c] += w * input[ii * filter_c_in + k] / (T)neighbors[f].size();
                            }
                        }
                    }
                }
            }        
        }
    }

    void gradient(const T *grad_from_next_tensor_flat, 
                  const T *points_flat, const T *input_flat, const T *filter, T *grad_input_flat, T *grad_filter, 
                  int batch_size, int num_points, int filter_x, int filter_y, int filter_z, int filter_c_in, int filter_c_out, float voxel_size) {

        /**2. Compute gradient of the input**/
        // dL_dxj = (sum over xi that have xj as a neighbor)dL/dxi * w_as
        // w_as is the weight associated xi and xj. w_as = 0 if xj not contribute to xi.
        // we take avantaged of if xj is a neighbor of xi, then xi is also a neighbor of xi
        // or that we have symmetric neighborhood, then
        // dL_dxj = (sum over all xi that are neightbor of xj)dL/dxi * w_as
        int filter_count = filter_x * filter_y * filter_z;

        #ifdef CONV_OPENMP
        #pragma omp parallel for
        #endif
        for (int b = 0; b < batch_size; ++b) {
            const T *points_arr = points_flat + b * num_points * 3;       // XYZ as input
            const T *input_arr = input_flat + b * num_points * filter_c_in;
            const T *grad_from_next_tensor_arr = grad_from_next_tensor_flat + b * num_points * filter_c_out;
            T *grad_input_arr = grad_input_flat + b * num_points * filter_c_in;
    
            std::vector<int> neighbor_count;
            getNeighborCount(points_arr, num_points, filter_x, filter_y, filter_z, voxel_size, neighbor_count);
    
            for (int j = 0; j < num_points; ++j) {
                T x = points_arr[3 * j + 0];
                T y = points_arr[3 * j + 1];
                T z = points_arr[3 * j + 2];
    
                std::vector< std::vector<int> > neighbors;
                getNeighbor(points_arr, num_points, x, y, z, filter_x, filter_y, filter_z, voxel_size, neighbors);
    
                for (int f = 0; f < filter_x * filter_y * filter_z; ++f) {
                    for (int i = 0; i < neighbors[f].size(); ++i) {
                        int ii = neighbors[f][i];
    
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
    
                        // for all types of filters
                        for (int c = 0; c < filter_c_out; ++c) {
                            for (int k = 0; k < filter_c_in; ++k) {
                                // Update the gradient of an input xi
                                int weight_index = (filter_index * filter_c_in + k) * filter_c_out + c;
                                int out_index = ii * filter_c_out + c;
                                int in_index = j * filter_c_in + k;
    
                                T dL_dxi = grad_from_next_tensor_arr[out_index];
    
                                T w_as = filter[weight_index];
                                grad_input_arr[in_index] += dL_dxi * w_as / (T)count;
                            }
                        }
                    }
                }
            }
        }
    
        // /**3. Compute gradient of the filter**/

        // Reminder: grad_from_next_tensor contain dL/dx, with xi is an output of the forward pass.
        // Reminder: we can calculate dL/dw = (sum over i)dL/dxi * dx/dw, with w is a weight that connect x to the next layer
        // =>
        // b) We do this by go through all the points and accumulate the gradients
        for (int b = 0; b < batch_size; ++b) {
            const T *points_arr = points_flat + b * num_points * 3;       // XYZ as input
            const T *input_arr = input_flat + b * num_points * filter_c_in;
            const T *grad_from_next_tensor_arr = grad_from_next_tensor_flat + b * num_points * filter_c_out;

            std::vector<int> neighbor_count;
            getNeighborCount(points_arr, num_points, filter_x, filter_y, filter_z, voxel_size, neighbor_count);

            for (int j = 0; j < num_points; ++j) {
                T x = points_arr[3 * j + 0];
                T y = points_arr[3 * j + 1];
                T z = points_arr[3 * j + 2];

                std::vector<std::vector<int> > neighbors;
                getNeighbor(points_arr, num_points, x, y, z, filter_x, filter_y, filter_z, voxel_size, neighbors);

                for (int f = 0; f < filter_x * filter_y * filter_z; ++f) {
                    for (int i = 0; i < neighbors[f].size(); ++i) {
                        int ii = neighbors[f][i];

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

                        // for all types of filters
                        for (int c = 0; c < filter_c_out; ++c) {
                            for (int k = 0; k < filter_c_in; ++k) {
                                // Update the gradient of an input xi
                                int weight_index = (filter_index * filter_c_in + k) * filter_c_out + c;
                                int out_index = ii * filter_c_out + c;
                                int in_index = j * filter_c_in + k;

                                T dL_dxi = grad_from_next_tensor_arr[out_index];
                                T dxi_dw = input_arr[in_index];
                                grad_filter[weight_index] += dL_dxi * dxi_dw / count;
                            }
                        }
                    }
                }
            }
        }
    }
};

struct ForwardQuery {
    ForwardQuery(const T *points_flat,  const T *input_flat, const T *filter, T *output_flat, 
                 int batch_size, int num_points, int filter_x, int filter_y, int filter_z, 
                 int filter_c_in, int filter_c_out, float voxel_size) 
    points_flat(points_flat), 
    input_flat(input_flat),
    filter(filter),
    output_flat(output_flat),
    batch_size(batch_size), num_points(num_points), filter_x(filter_x), filter_y(filter_y), filter_z(filter_z), 
    filter_c_in(filter_c_in), filter_c_out(filter_c_out), voxel_size(voxel_size)
    {
            
    }

    __device__ void operator()(int ii, int f, int cell_count) {
            
        // Convolution
        for (int c = 0; c < filter_c_out; ++c) {
            for (int k = 0; k < filter_c_in; ++k) {

                // Get filter weight
                T w = filter[(f * filter_c_in + k) * filter_c_out + c];

                output[i * filter_c_out + c] += w * input[ii * filter_c_in + k] / (T)cell_count;
            }
        }               
    }

    const T *points_flat,  const T *input_flat, const T *filter, T *output_flat, 
    int batch_size, int num_points, int filter_x, int filter_y, int filter_z, 
    int filter_c_in, int filter_c_out, float voxel_size;
};

template <typename T>
struct Convolution<GPUDevice, T> {
    
    __device__ void cuda_getNeighbor(const T *points, int n, T x, T y, T z, int filter_x, int filter_y, int filter_z, float voxel_size,
                                     ForwardQuery &query) {
        // Center the filter at the current point
        T xmin = x - filter_x * 0.5 * voxel_size;
        T xmax = x + filter_x * 0.5 * voxel_size;
        T ymin = y - filter_y * 0.5 * voxel_size;
        T ymax = y + filter_y * 0.5 * voxel_size;
        T zmin = z - filter_z * 0.5 * voxel_size;
        T zmax = z + filter_z * 0.5 * voxel_size;

        // Brute force for now
        int cell_count = 0;
        for (int i = 0; i < n; ++i) {
            T vx = points[3 * i + 0];
            T vy = points[3 * i + 1];
            T vz = points[3 * i + 2];

            if (vx < xmin || vx >= xmax || vy < ymin || vy >= ymax || vz < zmin || vz >= zmax) continue;
            cell_count++;
        }

        for (int i = 0; i < n; ++i) {
            T vx = points[3 * i + 0];
            T vy = points[3 * i + 1];
            T vz = points[3 * i + 2];

            if (vx < xmin || vx >= xmax || vy < ymin || vy >= ymax || vz < zmin || vz >= zmax) continue;

            // determine which cell
            int fx = min(filter_x - 1, (int)((vx - xmin) / voxel_size));
            int fy = min(filter_y - 1, (int)((vy - ymin) / voxel_size));
            int fz = min(filter_z - 1, (int)((vz - zmin) / voxel_size));

            int f = (fz * filter_y + fy) * filter_x + fx;

            query(i, f, cell_count);
        }
    }
        
    void forward(const T *points_flat,  const T *input_flat, const T *filter, T *output_flat, 
                 int batch_size, int num_points, int filter_x, int filter_y, int filter_z, int filter_c_in, int filter_c_out, float voxel_size) 
    {
        // launch kernel 
        dim3 size(32, 2048)
        dim3 block = dim3(1, 2048);
        cuda_forward<<<size, block>>>(points_flat, input_flat, filter, output_flat, batch_size, num_points, filter_x, filter_y, filter_z, filter_c_in, filter_c_out, voxel_size);
    }
    
    __global__ void cuda_forward(const T *points_flat,  const T *input_flat, const T *filter, T *output_flat, 
                                 int batch_size, int num_points, int filter_x, int filter_y, int filter_z, 
                                 int filter_c_in, int filter_c_out, float voxel_size) 
    {
        int b = blockIdx.x * blockDim.x + threadIdx.x;
        int i = blockIdx.y * blockDim.y + threadIdx.y;
        if (b >= batch_size) return;
        if (i >= num_points) return;
        
        const T *points = points_flat + b * num_points * 3;       // XYZ as input
        const T *input = input_flat + b * num_points * filter_c_in;
        T *output = output_flat + b * num_points * filter_c_out;

        T x = points[3 * i + 0];
        T y = points[3 * i + 1];
        T z = points[3 * i + 2];
        
        ForwardQuery query(points_flat, input_flat, filter, output_flat, batch_size, num_points, filter_x, filter_y, filter_z, 
                           filter_c_in, filter_c_out, voxel_size);
        cuda_getNeighbor(points, num_points, x, y, z, filter_x, filter_y, filter_z, voxel_size);
    }

    

    void getNeighbor(const T *points, int n, T x, T y, T z, int filter_x, int filter_y, int filter_z, float voxel_size,
        std::vector< std::vector<int> > &neighbors) {
        // Center the filter at the current point
        T xmin = x - filter_x * 0.5 * voxel_size;
        T xmax = x + filter_x * 0.5 * voxel_size;
        T ymin = y - filter_y * 0.5 * voxel_size;
        T ymax = y + filter_y * 0.5 * voxel_size;
        T zmin = z - filter_z * 0.5 * voxel_size;
        T zmax = z + filter_z * 0.5 * voxel_size;

        // Brute force for now
        neighbors.resize(filter_x * filter_y * filter_z);
        for (int i = 0; i < n; ++i) {
            T vx = points[3 * i + 0];
            T vy = points[3 * i + 1];
            T vz = points[3 * i + 2];

            if (vx < xmin || vx >= xmax || vy < ymin || vy >= ymax || vz < zmin || vz >= zmax) continue;

            // determine which cell
            int fx = std::min(filter_x - 1, (int)((vx - xmin) / voxel_size));
            int fy = std::min(filter_y - 1, (int)((vy - ymin) / voxel_size));
            int fz = std::min(filter_z - 1, (int)((vz - zmin) / voxel_size));

            int idx = (fz * filter_y + fy) * filter_x + fx;

            neighbors[idx].push_back(i);
        }
        //std::cout << "Neighbor size: " << neighbors.size() << std::endl;
    }
    
    void getNeighborCount(const T *points, int n, int filter_x, int filter_y, int filter_z, float voxel_size,
                          int *count) {

        int filter_count = filter_x * filter_y * filter_z;
        count.resize(filter_count * n);
        for (int i = 0; i < count.size(); ++i) count[i] = 0;

        for (int j = 0; j < n; ++j) {
            T x = points[3 * j + 0];
            T y = points[3 * j + 1];
            T z = points[3 * j + 2];

            // Center the filter at the current point
            T xmin = x - filter_x * 0.5 * voxel_size;
            T xmax = x + filter_x * 0.5 * voxel_size;
            T ymin = y - filter_y * 0.5 * voxel_size;
            T ymax = y + filter_y * 0.5 * voxel_size;
            T zmin = z - filter_z * 0.5 * voxel_size;
            T zmax = z + filter_z * 0.5 * voxel_size;

            // Brute force for now
            for (int i = 0; i < n; ++i) {
                T vx = points[3 * i + 0];
                T vy = points[3 * i + 1];
                T vz = points[3 * i + 2];

                if (vx < xmin || vx >= xmax || vy < ymin || vy >= ymax || vz < zmin || vz >= zmax) continue;

                // determine which cell
                int fx = std::min(filter_x - 1, (int)((vx - xmin) / voxel_size));
                int fy = std::min(filter_y - 1, (int)((vy - ymin) / voxel_size));
                int fz = std::min(filter_z - 1, (int)((vz - zmin) / voxel_size));

                int idx = (fz * filter_y + fy) * filter_x + fx;

                count[j * filter_count + idx]++;
            }
        }
    }

    void gradient(const T *grad_from_next_tensor_flat, 
                  const T *points_flat, const T *input_flat, const T *filter, T *grad_input_flat, T *grad_filter, 
                  int batch_size, int num_points, int filter_x, int filter_y, int filter_z, int filter_c_in, int filter_c_out, float voxel_size) {

        /**2. Compute gradient of the input**/
        // dL_dxj = (sum over xi that have xj as a neighbor)dL/dxi * w_as
        // w_as is the weight associated xi and xj. w_as = 0 if xj not contribute to xi.
        // we take avantaged of if xj is a neighbor of xi, then xi is also a neighbor of xi
        // or that we have symmetric neighborhood, then
        // dL_dxj = (sum over all xi that are neightbor of xj)dL/dxi * w_as
        int filter_count = filter_x * filter_y * filter_z;

        #ifdef CONV_OPENMP
        #pragma omp parallel for
        #endif
        for (int b = 0; b < batch_size; ++b) {
            const T *points_arr = points_flat + b * num_points * 3;       // XYZ as input
            const T *input_arr = input_flat + b * num_points * filter_c_in;
            const T *grad_from_next_tensor_arr = grad_from_next_tensor_flat + b * num_points * filter_c_out;
            T *grad_input_arr = grad_input_flat + b * num_points * filter_c_in;
    
            std::vector<int> neighbor_count;
            getNeighborCount(points_arr, num_points, filter_x, filter_y, filter_z, voxel_size, neighbor_count);
    
            for (int j = 0; j < num_points; ++j) {
                T x = points_arr[3 * j + 0];
                T y = points_arr[3 * j + 1];
                T z = points_arr[3 * j + 2];
    
                std::vector< std::vector<int> > neighbors;
                getNeighbor(points_arr, num_points, x, y, z, filter_x, filter_y, filter_z, voxel_size, neighbors);
    
                for (int f = 0; f < filter_x * filter_y * filter_z; ++f) {
                    for (int i = 0; i < neighbors[f].size(); ++i) {
                        int ii = neighbors[f][i];
    
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
    
                        // for all types of filters
                        for (int c = 0; c < filter_c_out; ++c) {
                            for (int k = 0; k < filter_c_in; ++k) {
                                // Update the gradient of an input xi
                                int weight_index = (filter_index * filter_c_in + k) * filter_c_out + c;
                                int out_index = ii * filter_c_out + c;
                                int in_index = j * filter_c_in + k;
    
                                T dL_dxi = grad_from_next_tensor_arr[out_index];
    
                                T w_as = filter[weight_index];
                                grad_input_arr[in_index] += dL_dxi * w_as / (T)count;
                            }
                        }
                    }
                }
            }
        }
    
        // /**3. Compute gradient of the filter**/

        // Reminder: grad_from_next_tensor contain dL/dx, with xi is an output of the forward pass.
        // Reminder: we can calculate dL/dw = (sum over i)dL/dxi * dx/dw, with w is a weight that connect x to the next layer
        // =>
        // b) We do this by go through all the points and accumulate the gradients
        for (int b = 0; b < batch_size; ++b) {
            const T *points_arr = points_flat + b * num_points * 3;       // XYZ as input
            const T *input_arr = input_flat + b * num_points * filter_c_in;
            const T *grad_from_next_tensor_arr = grad_from_next_tensor_flat + b * num_points * filter_c_out;

            std::vector<int> neighbor_count;
            getNeighborCount(points_arr, num_points, filter_x, filter_y, filter_z, voxel_size, neighbor_count);

            for (int j = 0; j < num_points; ++j) {
                T x = points_arr[3 * j + 0];
                T y = points_arr[3 * j + 1];
                T z = points_arr[3 * j + 2];

                std::vector<std::vector<int> > neighbors;
                getNeighbor(points_arr, num_points, x, y, z, filter_x, filter_y, filter_z, voxel_size, neighbors);

                for (int f = 0; f < filter_x * filter_y * filter_z; ++f) {
                    for (int i = 0; i < neighbors[f].size(); ++i) {
                        int ii = neighbors[f][i];

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

                        // for all types of filters
                        for (int c = 0; c < filter_c_out; ++c) {
                            for (int k = 0; k < filter_c_in; ++k) {
                                // Update the gradient of an input xi
                                int weight_index = (filter_index * filter_c_in + k) * filter_c_out + c;
                                int out_index = ii * filter_c_out + c;
                                int in_index = j * filter_c_in + k;

                                T dL_dxi = grad_from_next_tensor_arr[out_index];
                                T dxi_dw = input_arr[in_index];
                                grad_filter[weight_index] += dL_dxi * dxi_dw / count;
                            }
                        }
                    }
                }
            }
        }
    }
};

template <typename Device, typename T>
class Conv3pOp : public OpKernel {
 public:
  explicit Conv3pOp(OpKernelConstruction* context) : OpKernel(context) {
  }

  void Compute(OpKernelContext* context) override {
#ifdef CONV_BENCHMARK
    long whole_time = 0;    
    timer t;
#endif
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

    Convolution<Device, T> op; 
    op.forward(&(points_flat(0)), &(input_flat(0)), &(filter_flat(0)), &(output_flat(0)), 
               batch_size, num_points, filter_x, filter_y, filter_z, filter_c_in, filter_c_out, voxel_size);
#ifdef CONV_BENCHMARK
    whole_time = t.getTimePassed();    
    printf("whole_time_forwardpass: %f\n", (double)whole_time/1e9);
#endif
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
    
    // Get shape of the grad tensors
    TensorShape grad_input_shape = input_tensor.shape();
    TensorShape grad_filter_shape = filter_tensor.shape();

    // Create the output tensor for the gradient of the inputs
    // How many points * number of input channel = how many gradients.
    Tensor* grad_input = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(0, grad_input_shape, &grad_input));
    auto grad_input_flat = grad_input->flat<T>();
    memset(&(grad_input_flat(0)), 0, sizeof(T) * input_tensor.shape().dim_size(0)*input_tensor.shape().dim_size(1)*input_tensor.shape().dim_size(2));

    // Create the output tensor for the gradient of the filter
    Tensor* grad_filter = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(1, grad_filter_shape, &grad_filter));
    auto grad_filter_flat = grad_filter->flat<T>();
    memset(&(grad_filter_flat(0)), 0, sizeof(T) * n_weights);

    // a) First we need to check if the size of the grad tensor and the number of points are compitable.
    OP_REQUIRES(context, grad_from_next_tensor.shape().dim_size(0) == batch_size, errors::InvalidArgument("backprop grad tensor has wrong size for dim 0"));
    OP_REQUIRES(context, grad_from_next_tensor.shape().dim_size(1) == num_points, errors::InvalidArgument("backprop grad tensor has wrong size for dim 1"));
    OP_REQUIRES(context, grad_from_next_tensor.shape().dim_size(2) == filter_c_out, errors::InvalidArgument("backprop grad tensor has wrong size for dim 2"));

    Convolution<Device, T> op; 
    op.gradient(&(grad_from_next_tensor_flat(0)), 
                &(points_flat(0)), &(input_flat(0)), &(filter_flat(0)), &(grad_input_flat(0)), &(grad_filter_flat(0)),
                batch_size, num_points, filter_x, filter_y, filter_z, filter_c_in, filter_c_out, voxel_size);
  }
};
#define REGISTER_CPU_KERNEL(T)                                        \
  REGISTER_KERNEL_BUILDER(                                            \
      Name("Conv3pGrad").Device(DEVICE_CPU).TypeConstraint<T>("T"),   \
      Conv3pGradOp<CPUDevice, T>);
TF_CALL_float(REGISTER_CPU_KERNEL);
TF_CALL_double(REGISTER_CPU_KERNEL);
#undef REGISTER_CPU_KERNEL
