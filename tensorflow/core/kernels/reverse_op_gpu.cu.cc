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

#if GOOGLE_CUDA || TENSORFLOW_USE_ROCM

#define EIGEN_USE_GPU

#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/kernels/reverse_op.h"

namespace tensorflow {

typedef Eigen::GpuDevice GPUDevice;

#define DEFINE_REVERSE(T, DIM) \
  template struct functor::Reverse<GPUDevice, T, DIM>;
#define DEFINE_REVERSE_ALL_DIMS(T) \
  DEFINE_REVERSE(T, 0)             \
  DEFINE_REVERSE(T, 1)             \
  DEFINE_REVERSE(T, 2)             \
  DEFINE_REVERSE(T, 3)             \
  DEFINE_REVERSE(T, 4)             \
  DEFINE_REVERSE(T, 5)             \
  DEFINE_REVERSE(T, 6)             \
  DEFINE_REVERSE(T, 7)             \
  DEFINE_REVERSE(T, 8)

TF_CALL_uint8(DEFINE_REVERSE_ALL_DIMS);
TF_CALL_int8(DEFINE_REVERSE_ALL_DIMS);
TF_CALL_GPU_ALL_TYPES(DEFINE_REVERSE_ALL_DIMS);
#undef DEFINE_REVERSE
#undef DEFINE_REVERSE_ALL_DIMS

}  // namespace tensorflow

#endif  // GOOGLE_CUDA || TENSORFLOW_USE_ROCM
