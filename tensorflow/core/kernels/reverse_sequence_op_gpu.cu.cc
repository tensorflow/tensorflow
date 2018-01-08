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
#include "tensorflow/core/kernels/reverse_sequence_op.h"

namespace tensorflow {

typedef Eigen::GpuDevice GPUDevice;

#define DEFINE_GPU_SPEC(T, Tlen, dims)                       \
  template class generator::ReverseGenerator<T, Tlen, dims>; \
  template struct functor::ReverseSequence<GPUDevice, T, Tlen, dims>;

#define DEFINE_GPU_SPEC_LEN(T, dims)  \
  DEFINE_GPU_SPEC(T, int32, dims);    \
  DEFINE_GPU_SPEC(T, int64, dims);

#define DEFINE_GPU_SPECS(T) \
  DEFINE_GPU_SPEC_LEN(T, 2);    \
  DEFINE_GPU_SPEC_LEN(T, 3);    \
  DEFINE_GPU_SPEC_LEN(T, 4);    \
  DEFINE_GPU_SPEC_LEN(T, 5);

TF_CALL_GPU_NUMBER_TYPES(DEFINE_GPU_SPECS);
TF_CALL_bool(DEFINE_GPU_SPECS);

}  // end namespace tensorflow

#endif  // GOOGLE_CUDA
