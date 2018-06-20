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

#include "tensorflow/core/kernels/pad_op.h"

#include "tensorflow/core/framework/register_types.h"

namespace tensorflow {

typedef Eigen::GpuDevice GPUDevice;

// Definition of the GPU implementations declared in pad_op.cc.
#define DEFINE_GPU_PAD_SPECS(T, Tpadding)                  \
  template struct functor::Pad<GPUDevice, T, Tpadding, 0>; \
  template struct functor::Pad<GPUDevice, T, Tpadding, 1>; \
  template struct functor::Pad<GPUDevice, T, Tpadding, 2>; \
  template struct functor::Pad<GPUDevice, T, Tpadding, 3>; \
  template struct functor::Pad<GPUDevice, T, Tpadding, 4>; \
  template struct functor::Pad<GPUDevice, T, Tpadding, 5>; \
  template struct functor::Pad<GPUDevice, T, Tpadding, 6>;

#define DEFINE_GPU_SPECS(T)      \
  DEFINE_GPU_PAD_SPECS(T, int32) \
  DEFINE_GPU_PAD_SPECS(T, int64)

TF_CALL_GPU_NUMBER_TYPES(DEFINE_GPU_SPECS);
TF_CALL_int8(DEFINE_GPU_SPECS);

}  // namespace tensorflow

#endif  // GOOGLE_CUDA
