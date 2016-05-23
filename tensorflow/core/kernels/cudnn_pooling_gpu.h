/* Copyright 2016 Google Inc. All Rights Reserved.

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

// Helper functions to run 3d pooling on GPU using CuDNN.

#ifndef TENSORFLOW_KERNELS_CUDNN_POOLING_GPU_H_
#define TENSORFLOW_KERNELS_CUDNN_POOLING_GPU_H_

#include <array>

#include "tensorflow/core/framework/op_kernel.h"

#if GOOGLE_CUDA
#include "tensorflow/core/platform/stream_executor.h"
#endif

#include "tensorflow/core/util/padding.h"

namespace tensorflow {

#if GOOGLE_CUDA

// Runs (avg/max)pooling on GPU.
// Dimension order for all array arguments is: x, y, z.
template <typename T>
class DnnPooling3dOp {
 public:
  static void Compute(OpKernelContext* context,
                      perftools::gputools::dnn::PoolingMode pooling_mode,
                      const std::array<int64, 3>& size,
                      const std::array<int64, 3>& stride,
                      const std::array<int64, 3>& padding,
                      const Tensor& tensor_in, Tensor* output);
};

// Computes the gradient of (avg/max)pooling on GPU.
// Dimension order for all array arguments is: x, y, z.
template <typename T>
class DnnPooling3dGradOp {
 public:
  static void Compute(OpKernelContext* context,
                      perftools::gputools::dnn::PoolingMode pooling_mode,
                      const std::array<int64, 3>& window,
                      const std::array<int64, 3>& stride,
                      const std::array<int64, 3>& padding,
                      const std::array<int64, 3>& output_size,
                      const Tensor& out_backprop,
                      const TensorShape& tensor_in_shape,
                      const Tensor* tensor_in, const Tensor* tensor_out,
                      Tensor* input_backprop);
};

#endif

}  // namespace tensorflow

#endif  // TENSORFLOW_KERNELS_CUDNN_POOLING_GPU_H_
