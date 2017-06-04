/* Copyright 2015 The TensorFlow Authors. All Rights Reserved.

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

#if GOOGLE_CUDA

#define EIGEN_USE_GPU

#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/kernels/matrix_diag_op.h"

namespace tensorflow {

typedef Eigen::GpuDevice GPUDevice;

#define DEFINE_GPU_SPEC(T)                              \
  template class generator::MatrixDiagGenerator<T>;     \
  template struct functor::MatrixDiag<GPUDevice, T>;    \
  template class generator::MatrixDiagPartGenerator<T>; \
  template struct functor::MatrixDiagPart<GPUDevice, T>;

TF_CALL_GPU_NUMBER_TYPES(DEFINE_GPU_SPEC);
TF_CALL_bool(DEFINE_GPU_SPEC);
TF_CALL_complex64(DEFINE_GPU_SPEC);
TF_CALL_complex128(DEFINE_GPU_SPEC);

}  // end namespace tensorflow

#endif  // GOOGLE_CUDA
