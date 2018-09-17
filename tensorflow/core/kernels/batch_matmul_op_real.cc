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

#include "tensorflow/core/kernels/batch_matmul_op_impl.h"

#if GOOGLE_CUDA
#include "cuda/include/cuda.h"
#endif  // GOOGLE_CUDA

namespace tensorflow {

#if !defined(INTEL_MKL) || defined(INTEL_MKL_DNN_ONLY)
TF_CALL_float(REGISTER_BATCH_MATMUL_CPU);
TF_CALL_double(REGISTER_BATCH_MATMUL_CPU);
#endif
TF_CALL_half(REGISTER_BATCH_MATMUL_CPU);
TF_CALL_int32(REGISTER_BATCH_MATMUL_CPU);

#if GOOGLE_CUDA
TF_CALL_float(REGISTER_BATCH_MATMUL_GPU);
TF_CALL_double(REGISTER_BATCH_MATMUL_GPU);
TF_CALL_half(REGISTER_BATCH_MATMUL_GPU);
#endif  // GOOGLE_CUDA

#ifdef TENSORFLOW_USE_SYCL
TF_CALL_float(REGISTER_BATCH_MATMUL_SYCL);
TF_CALL_double(REGISTER_BATCH_MATMUL_SYCL);
#endif  // TENSORFLOW_USE_SYCL
}  // namespace tensorflow
