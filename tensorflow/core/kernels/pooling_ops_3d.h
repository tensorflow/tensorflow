/* Copyright 2015 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#ifndef TENSORFLOW_CORE_KERNELS_POOLING_OPS_3D_H_
#define TENSORFLOW_CORE_KERNELS_POOLING_OPS_3D_H_

#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/util/padding.h"
#include "tensorflow/core/util/tensor_format.h"

namespace tensorflow {

enum PoolingType { MAX, AVG };

template <typename Device, typename T, PoolingType Type>
struct LaunchPoolingOp;

template <typename Device, typename T>
struct LaunchAvgPooling3dGradOp;

template <typename Device, typename T>
struct LaunchMaxPooling3dGradOp;

template <typename Device, typename T>
struct LaunchMaxPooling3dGradGradOp;

// A helper class to manage sizes and shapes for 3d pooling operations.
struct Pool3dParameters {
  // Updates context->status if there is an invalid input.
  Pool3dParameters(OpKernelContext* context, const std::vector<int32>& ksize,
                   const std::vector<int32>& stride, Padding padding,
                   TensorFormat data_format,
                   const TensorShape& tensor_in_shape);

  // Returns the shape of the output for "forward" pooling operations.
  TensorShape forward_output_shape();

  int depth;

  int tensor_in_planes;
  int tensor_in_cols;
  int tensor_in_rows;
  int tensor_in_batch;

  int window_planes;
  int window_cols;
  int window_rows;
  int depth_window;

  int plane_stride;
  int col_stride;
  int row_stride;
  int depth_stride;

  int64 out_plane;
  int64 out_height;
  int64 out_width;

  int64 pad_planes;
  int64 pad_cols;
  int64 pad_rows;

  TensorFormat data_format;
};

}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_KERNELS_POOLING_OPS_3D_H_
