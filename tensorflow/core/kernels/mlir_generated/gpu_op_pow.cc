/* Copyright 2021 The TensorFlow Authors. All Rights Reserved.

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
#include "tensorflow/core/framework/types.pb.h"
#include "tensorflow/core/kernels/integer_pow_gpu.h"
#include "tensorflow/core/kernels/mlir_generated/base_gpu_op.h"
#include "unsupported/Eigen/CXX11/Tensor"  // from @eigen_archive

namespace tensorflow {

GENERATE_AND_REGISTER_BINARY_GPU_KERNEL(Pow, DT_HALF);
GENERATE_AND_REGISTER_BINARY_GPU_KERNEL(Pow, DT_FLOAT);
GENERATE_AND_REGISTER_BINARY_GPU_KERNEL(Pow, DT_DOUBLE);
GENERATE_BINARY_GPU_KERNEL(Pow, DT_INT64);
using Int64GpuPowOp =
    IntegerPowGpuOp<int64_t, MLIR_OP(Pow, GPU, DT_INT64, DT_INT64)>;
REGISTER_KERNEL_BUILDER(
    Name("Pow").Device(DEVICE_GPU).TypeConstraint<int64_t>("T"), Int64GpuPowOp);

// These kernels are JIT-compiled.
GENERATE_BINARY_GPU_KERNEL(Pow, DT_INT8);
using Int8GpuPowOp =
    IntegerPowGpuOp<int8_t, MLIR_OP(Pow, GPU, DT_INT8, DT_INT8)>;
REGISTER_KERNEL_BUILDER(Name("Pow")
                            .Device(DEVICE_GPU)
                            .TypeConstraint<int8_t>("T")
                            .Label(kJitKernelLabel),
                        Int8GpuPowOp);

GENERATE_BINARY_GPU_KERNEL(Pow, DT_INT16);
using Int16GpuPowOp =
    IntegerPowGpuOp<int16_t, MLIR_OP(Pow, GPU, DT_INT16, DT_INT16)>;
REGISTER_KERNEL_BUILDER(Name("Pow")
                            .Device(DEVICE_GPU)
                            .TypeConstraint<int16_t>("T")
                            .Label(kJitKernelLabel),
                        Int16GpuPowOp);

}  // namespace tensorflow
