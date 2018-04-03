#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "tensorflow/core/framework/common_shape_fns.h"

#ifndef ATROUS 

REGISTER_OP("Conv3p")
.Attr("T: {float , double}")
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

#else 

// A-trous convolution

REGISTER_OP("Conv3p")
.Attr("T: {float, double}")
.Input("points: T")
.Input("input: T")
.Input("filter: T")
.Input("stride: int32")
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
.Input("stride: int32")
.Input("voxel_size: T")
.Output("grad_input: T")
.Output("grad_filter: T")
.Doc(R"doc(
Computes the gradient of a 3-D point cloud convolution with respect to the input and weights
)doc");

#endif
