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

#include "unsupported/Eigen/CXX11/Tensor"  // from @eigen_archive
#include "tensorflow/core/framework/types.pb.h"
#include "tensorflow/core/kernels/mlir_generated/base_gpu_op.h"

namespace tensorflow {

GENERATE_AND_REGISTER_BINARY_GPU_KERNEL(Div, DT_HALF);
GENERATE_AND_REGISTER_BINARY_GPU_KERNEL(Div, DT_FLOAT);
GENERATE_AND_REGISTER_BINARY_GPU_KERNEL(Div, DT_DOUBLE);
GENERATE_AND_REGISTER_BINARY_GPU_KERNEL(Div, DT_INT16);
GENERATE_AND_REGISTER_BINARY_GPU_KERNEL(Div, DT_INT64);
GENERATE_AND_REGISTER_BINARY_GPU_KERNEL(Div, DT_UINT8);
GENERATE_AND_REGISTER_BINARY_GPU_KERNEL(Div, DT_UINT16);
GENERATE_AND_REGISTER_BINARY_GPU_KERNEL(Div, DT_COMPLEX64);
GENERATE_AND_REGISTER_BINARY_GPU_KERNEL(Div, DT_COMPLEX128);

// These kernels are JIT-compiled.
GENERATE_AND_REGISTER_BINARY_JIT_GPU_KERNEL(Div, DT_INT8);
GENERATE_AND_REGISTER_BINARY_JIT_GPU_KERNEL(Div, DT_UINT32);
GENERATE_AND_REGISTER_BINARY_JIT_GPU_KERNEL(Div, DT_UINT64);

REGISTER_ALIASED_GPU_KERNEL(RealDiv, Div, DT_HALF, DT_HALF);
REGISTER_ALIASED_GPU_KERNEL(RealDiv, Div, DT_FLOAT, DT_FLOAT);
REGISTER_ALIASED_GPU_KERNEL(RealDiv, Div, DT_DOUBLE, DT_DOUBLE);
REGISTER_ALIASED_GPU_KERNEL(RealDiv, Div, DT_COMPLEX64, DT_COMPLEX64);
REGISTER_ALIASED_GPU_KERNEL(RealDiv, Div, DT_COMPLEX128, DT_COMPLEX128);

REGISTER_ALIASED_GPU_KERNEL(TruncateDiv, Div, DT_UINT8, DT_UINT8);
REGISTER_ALIASED_GPU_KERNEL(TruncateDiv, Div, DT_UINT16, DT_UINT16);
REGISTER_ALIASED_GPU_KERNEL(TruncateDiv, Div, DT_INT16, DT_INT16);
REGISTER_ALIASED_GPU_KERNEL(TruncateDiv, Div, DT_INT64, DT_INT64);

REGISTER_ALIASED_GPU_KERNEL(FloorDiv, Div, DT_UINT8, DT_UINT8);
REGISTER_ALIASED_GPU_KERNEL(FloorDiv, Div, DT_UINT16, DT_UINT16);

}  // namespace tensorflow
