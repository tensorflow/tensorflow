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

#if GOOGLE_CUDA

#define EIGEN_USE_GPU

#include "tensorflow/core/kernels/conv_2d.h"

#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/framework/tensor.h"

namespace tensorflow {

typedef Eigen::GpuDevice GPUDevice;
template struct functor::InflatePadAndShuffle<GPUDevice, float, 4, int>;
template struct functor::InflatePadAndShuffle<GPUDevice, float, 4,
                                              Eigen::DenseIndex>;
template struct functor::InflatePadAndShuffle<GPUDevice, Eigen::half, 4, int>;
template struct functor::InflatePadAndShuffle<GPUDevice, Eigen::half, 4,
                                              Eigen::DenseIndex>;
}  // namespace tensorflow

#endif  // GOOGLE_CUDA
