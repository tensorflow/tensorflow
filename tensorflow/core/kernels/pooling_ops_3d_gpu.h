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

#if !GOOGLE_CUDA
#error This file must only be included when building with Cuda support
#endif

#ifndef TENSORFLOW_CORE_KERNELS_POOLING_OPS_3D_GPU_H_
#define TENSORFLOW_CORE_KERNELS_POOLING_OPS_3D_GPU_H_

#define EIGEN_USE_GPU

#include "tensorflow/core/framework/tensor_types.h"
#include "tensorflow/core/platform/types.h"
#include "tensorflow/core/util/tensor_format.h"

namespace tensorflow {

namespace functor {
template <typename T>
struct MaxPool3dGradBackward {
  bool operator()(TensorFormat data_format, const T* bottom_data,
                  const T* output_data, const int batch, const int pooled_plane,
                  const int pooled_height, const int pooled_width,
                  const int channels, const int plane, const int height,
                  const int width, const int kernel_p, const int kernel_h,
                  const int kernel_w, const int stride_p, const int stride_h,
                  const int stride_w, const int pad_p, const int pad_t,
                  const int pad_l, const T* top_diff, T* bottom_diff,
                  const Eigen::GpuDevice& d);
};
}  // namespace functor

}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_KERNELS_POOLING_OPS_3D_GPU_H_
