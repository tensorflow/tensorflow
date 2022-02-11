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

// The aliased kernels for data types DT_UINT8 and DT_UINT16 are registered in
// gpu_op_div.cc, because they alias the Div kernel.

// These kernels are JIT-compiled.
#if defined(MLIR_GENERATED_EXPERIMENTAL_KERNELS_ENABLED)
GENERATE_AND_REGISTER_BINARY_JIT_GPU_KERNEL(FloorDiv, DT_INT8);
GENERATE_AND_REGISTER_BINARY_JIT_GPU_KERNEL(FloorDiv, DT_UINT32);
GENERATE_AND_REGISTER_BINARY_JIT_GPU_KERNEL(FloorDiv, DT_UINT64);
#endif

GENERATE_AND_REGISTER_BINARY_GPU_KERNEL(FloorDiv, DT_INT16);
GENERATE_AND_REGISTER_BINARY_GPU_KERNEL(FloorDiv, DT_INT64);

GENERATE_AND_REGISTER_BINARY_GPU_KERNEL(FloorDiv, DT_HALF);
GENERATE_AND_REGISTER_BINARY_GPU_KERNEL(FloorDiv, DT_FLOAT);
GENERATE_AND_REGISTER_BINARY_GPU_KERNEL(FloorDiv, DT_DOUBLE);

}  // namespace tensorflow
