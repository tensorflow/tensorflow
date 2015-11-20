/* Copyright 2015 Google Inc. All Rights Reserved.

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

#include "tensorflow/core/kernels/reverse_op.h"

#include "tensorflow/core/framework/register_types.h"

namespace tensorflow {

typedef Eigen::GpuDevice GPUDevice;

#define DEFINE_REVERSE(DIM)                                \
  template struct functor::Reverse<GPUDevice, uint8, DIM>; \
  template struct functor::Reverse<GPUDevice, int8, DIM>;  \
  template struct functor::Reverse<GPUDevice, int32, DIM>; \
  template struct functor::Reverse<GPUDevice, bool, DIM>;  \
  template struct functor::Reverse<GPUDevice, float, DIM>; \
  template struct functor::Reverse<GPUDevice, double, DIM>;
DEFINE_REVERSE(0)
DEFINE_REVERSE(1)
DEFINE_REVERSE(2)
DEFINE_REVERSE(3)
DEFINE_REVERSE(4)
DEFINE_REVERSE(5)
DEFINE_REVERSE(6)
DEFINE_REVERSE(7)
DEFINE_REVERSE(8)
#undef DEFINE_REVERSE

}  // namespace tensorflow

#endif  // GOOGLE_CUDA
