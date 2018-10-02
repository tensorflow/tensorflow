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

namespace tensorflow {

// MKL_ML registers its own complex64/128 kernels in mkl_batch_matmul_op.cc
// if defined(INTEL_MKL) && !defined(INTEL_MKL_DNN_ONLY) && defined(ENABLE_MKL).
// Anything else (the complement) should register the TF ones.
// (MKL-DNN doesn't implement these kernels either.)
#if !defined(INTEL_MKL) || defined(INTEL_MKL_DNN_ONLY) || !defined(ENABLE_MKL)
TF_CALL_complex64(REGISTER_BATCH_MATMUL_CPU);
TF_CALL_complex128(REGISTER_BATCH_MATMUL_CPU);
#endif  // !INTEL_MKL || INTEL_MKL_DNN_ONLY || !ENABLE_MKL

#if GOOGLE_CUDA
TF_CALL_complex64(REGISTER_BATCH_MATMUL_GPU);
TF_CALL_complex128(REGISTER_BATCH_MATMUL_GPU);
#endif  // GOOGLE_CUDA

}  // namespace tensorflow
