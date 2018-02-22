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
#error This file must only be included when building with CUDA suppot
#endif

#ifndef TENSORFLOW_KERNELS_UNPOOLING_OP_GPU_H
#define TENSORFLOW_KERNELS_UNPOOLING_OP_GPU_H

#include "tensorflow/core/framework/tensor_shape.h"

namespace tensorflow {

bool UnpoolForward(const float* input, TensorShape inputShape,
                   const int64* indices, float* unpooledData,
                   const Eigen::GpuDevice& device);
bool UnpoolBackward(const float* unpooled_gradient, const int64* indices,
                    float* pooled_gradient, const int64 num_pooled_points,
                    const Eigen::GpuDevice& device);
}

#endif
