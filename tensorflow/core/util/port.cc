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

#include "tensorflow/core/util/port.h"

#if GOOGLE_CUDA
#include "cuda/include/cuda.h"
#endif

namespace tensorflow {

bool IsGoogleCudaEnabled() {
#if GOOGLE_CUDA
  return true;
#else
  return false;
#endif
}

bool CudaSupportsHalfMatMulAndConv() {
#if GOOGLE_CUDA
  // NOTE: We check compile-time and not runtime, since the check for
  // whether we include the fp16 kernels or not is compile-time.
  return CUDA_VERSION >= 7050;
#else
  return false;
#endif
}

bool IsMklEnabled() {
#ifdef INTEL_MKL
  return true;
#else
  return false;
#endif
}
}  // end namespace tensorflow
