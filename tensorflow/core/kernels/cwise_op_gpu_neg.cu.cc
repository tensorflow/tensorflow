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

#include "tensorflow/core/kernels/cwise_ops_gpu_common.cu.h"

namespace tensorflow {
namespace functor {
#if !defined(MLIR_GENERATED_GPU_KERNELS_ENABLED)
DEFINE_UNARY4(neg, int8, int16, int32, int64);
DEFINE_UNARY3(neg, Eigen::half, float, double);
#endif
DEFINE_UNARY3(neg, bfloat16, complex64, complex128);
}  // namespace functor
}  // namespace tensorflow

#endif  // GOOGLE_CUDA || TENSORFLOW_USE_ROCM
