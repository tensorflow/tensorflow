/* Copyright 2022 The TensorFlow Authors. All Rights Reserved.

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

#ifndef TENSORFLOW_COMPILER_XLA_PYTHON_PY_CLIENT_GPU_H_
#define TENSORFLOW_COMPILER_XLA_PYTHON_PY_CLIENT_GPU_H_

#if TENSORFLOW_USE_ROCM
#include "rocm/include/hip/hip_runtime.h"
#else
#include "third_party/gpus/cuda/include/cuda.h"
#endif
#include "tensorflow/compiler/xla/service/custom_call_status.h"

#if TENSORFLOW_USE_ROCM
#define gpuStreamHandle hipStream_t
#else
#define gpuStreamHandle CUstream
#endif

namespace xla {

void XlaPythonGpuCallback(gpuStreamHandle stream, void** buffers, const char* opaque,
                          size_t opaque_len, XlaCustomCallStatus* status);

}  // namespace xla

#endif  // TENSORFLOW_COMPILER_XLA_PYTHON_PY_CLIENT_GPU_H_
