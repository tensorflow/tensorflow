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

namespace tensorflow {

TF_CALL_COMPLEX_TYPES(REGISTER_BATCH_MATMUL_CPU);
REGISTER_BATCH_MATMUL_TOUT_CPU(complex64, complex64, complex64);
REGISTER_BATCH_MATMUL_TOUT_CPU(complex128, complex128, complex128);

#if GOOGLE_CUDA || TENSORFLOW_USE_ROCM
TF_CALL_COMPLEX_TYPES(REGISTER_BATCH_MATMUL_GPU);
REGISTER_BATCH_MATMUL_TOUT_GPU(complex64, complex64, complex64);
REGISTER_BATCH_MATMUL_TOUT_GPU(complex128, complex128, complex128);
#endif  // GOOGLE_CUDA || TENSORFLOW_USE_ROCM

}  // namespace tensorflow
