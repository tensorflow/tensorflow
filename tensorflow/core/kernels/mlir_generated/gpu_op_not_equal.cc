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

GENERATE_AND_REGISTER_BINARY_GPU_KERNEL2(NotEqual, DT_HALF, DT_BOOL);
GENERATE_AND_REGISTER_BINARY_GPU_KERNEL2(NotEqual, DT_FLOAT, DT_BOOL);
GENERATE_AND_REGISTER_BINARY_GPU_KERNEL2(NotEqual, DT_DOUBLE, DT_BOOL);

GENERATE_AND_REGISTER_BINARY_GPU_KERNEL2(NotEqual, DT_BOOL, DT_BOOL);
GENERATE_AND_REGISTER_BINARY_GPU_KERNEL2(NotEqual, DT_INT8, DT_BOOL);
GENERATE_AND_REGISTER_BINARY_GPU_KERNEL2(NotEqual, DT_INT16, DT_BOOL);
// TODO(b/25387198): We cannot use a regular GPU kernel for int32.
GENERATE_AND_REGISTER_BINARY_GPU_KERNEL2(NotEqual, DT_INT64, DT_BOOL);
GENERATE_AND_REGISTER_BINARY_GPU_KERNEL3(NotEqual, DT_INT8, DT_BOOL, DT_UINT8,
                                         DT_BOOL);
GENERATE_AND_REGISTER_BINARY_GPU_KERNEL3(NotEqual, DT_INT16, DT_BOOL, DT_UINT16,
                                         DT_BOOL);
GENERATE_AND_REGISTER_BINARY_GPU_KERNEL3(NotEqual, DT_INT32, DT_BOOL, DT_UINT32,
                                         DT_BOOL);
GENERATE_AND_REGISTER_BINARY_GPU_KERNEL3(NotEqual, DT_INT64, DT_BOOL, DT_UINT64,
                                         DT_BOOL);

GENERATE_AND_REGISTER_BINARY_GPU_KERNEL2(NotEqual, DT_COMPLEX64, DT_BOOL);
GENERATE_AND_REGISTER_BINARY_GPU_KERNEL2(NotEqual, DT_COMPLEX128, DT_BOOL);

}  // namespace tensorflow
