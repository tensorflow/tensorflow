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

#include "tensorflow/core/kernels/matrix_triangular_solve_op_impl.h"

namespace tensorflow {

TF_CALL_complex64(REGISTER_BATCH_MATRIX_TRIANGULAR_SOLVE_CPU);
TF_CALL_complex128(REGISTER_BATCH_MATRIX_TRIANGULAR_SOLVE_CPU);

#if GOOGLE_CUDA || TENSORFLOW_USE_ROCM
TF_CALL_complex64(REGISTER_BATCH_MATRIX_TRIANGULAR_SOLVE_GPU);
TF_CALL_complex128(REGISTER_BATCH_MATRIX_TRIANGULAR_SOLVE_GPU);
#endif  // GOOGLE_CUDA || TENSORFLOW_USE_ROCM

}  // namespace tensorflow
