/* Copyright 2016 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/core/kernels/matmul_op_impl.h"

#if GOOGLE_CUDA
#include "third_party/gpus/cuda/include/cuda.h"
#endif  // GOOGLE_CUDA

namespace tensorflow {

TF_CALL_FLOAT_TYPES(REGISTER_BATCH_MATMUL_CPU);
TF_CALL_int16(REGISTER_BATCH_MATMUL_CPU);
TF_CALL_int32(REGISTER_BATCH_MATMUL_CPU);
TF_CALL_int64(REGISTER_BATCH_MATMUL_CPU);

REGISTER_BATCH_MATMUL_TOUT_CPU(bfloat16, bfloat16, bfloat16);
REGISTER_BATCH_MATMUL_TOUT_CPU(float, float, float);
REGISTER_BATCH_MATMUL_TOUT_CPU(double, double, double);
REGISTER_BATCH_MATMUL_TOUT_CPU(int16, int16, int16);
REGISTER_BATCH_MATMUL_TOUT_CPU(int32, int32, int32);
REGISTER_BATCH_MATMUL_TOUT_CPU(int64_t, int64_t, int64_t);
REGISTER_BATCH_MATMUL_TOUT_CPU(int8, int8, int32);
REGISTER_BATCH_MATMUL_TOUT_CPU(uint8, int8, int32);
REGISTER_BATCH_MATMUL_TOUT_CPU(int8, uint8, int32);
REGISTER_BATCH_MATMUL_TOUT_CPU(uint8, uint8, int32);

REGISTER_BATCH_MATMUL_TOUT_CPU(bfloat16, int8, bfloat16);
REGISTER_BATCH_MATMUL_TOUT_CPU(bfloat16, uint8, bfloat16);
REGISTER_BATCH_MATMUL_TOUT_CPU(int8, bfloat16, bfloat16);
REGISTER_BATCH_MATMUL_TOUT_CPU(uint8, bfloat16, bfloat16);

#if GOOGLE_CUDA || TENSORFLOW_USE_ROCM
TF_CALL_GPU_NUMBER_TYPES(REGISTER_BATCH_MATMUL_GPU);
REGISTER_BATCH_MATMUL_TOUT_GPU(Eigen::half, Eigen::half, Eigen::half);
REGISTER_BATCH_MATMUL_TOUT_GPU(float, float, float);
REGISTER_BATCH_MATMUL_TOUT_GPU(double, double, double);
REGISTER_BATCH_MATMUL_TOUT_GPU(Eigen::bfloat16, Eigen::bfloat16,
                               Eigen::bfloat16);
#endif  // GOOGLE_CUDA || TENSORFLOW_USE_ROCM

}  // namespace tensorflow
