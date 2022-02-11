/* Copyright 2020 The TensorFlow Authors. All Rights Reserved.

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

#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"
#include "tensorflow/core/kernels/mlir_generated/base_gpu_op.h"

namespace tensorflow {

GENERATE_AND_REGISTER_UNARY_GPU_KERNEL(Sign, DT_HALF);
GENERATE_AND_REGISTER_UNARY_GPU_KERNEL(Sign, DT_FLOAT);
GENERATE_AND_REGISTER_UNARY_GPU_KERNEL(Sign, DT_DOUBLE);
// TODO(b/25387198): We cannot use a regular GPU kernel for int32.
GENERATE_AND_REGISTER_UNARY_GPU_KERNEL(Sign, DT_INT64);
GENERATE_AND_REGISTER_UNARY_GPU_KERNEL(Sign, DT_COMPLEX64);
GENERATE_AND_REGISTER_UNARY_GPU_KERNEL(Sign, DT_COMPLEX128);

// These kernels are JIT-compiled.
#if defined(MLIR_GENERATED_EXPERIMENTAL_KERNELS_ENABLED)
GENERATE_AND_REGISTER_UNARY_JIT_GPU_KERNEL(Sign, DT_INT8);
GENERATE_AND_REGISTER_UNARY_JIT_GPU_KERNEL(Sign, DT_INT16);
#endif

}  // namespace tensorflow
