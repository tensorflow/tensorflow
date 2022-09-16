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

// Defines the GpuStream type - the CUDA-specific implementation of the generic
// StreamExecutor Stream interface.

#ifndef TENSORFLOW_COMPILER_XLA_STREAM_EXECUTOR_CUDA_CUDA_STREAM_H_
#define TENSORFLOW_COMPILER_XLA_STREAM_EXECUTOR_CUDA_CUDA_STREAM_H_

#include "tensorflow/compiler/xla/stream_executor/gpu/gpu_stream.h"

namespace stream_executor {
namespace cuda {

using CUDAStream = gpu::GpuStream;

inline CUDAStream* AsCUDAStream(Stream* stream) {
  return gpu::AsGpuStream(stream);
}

}  // namespace cuda
}  // namespace stream_executor

#endif  // TENSORFLOW_COMPILER_XLA_STREAM_EXECUTOR_CUDA_CUDA_STREAM_H_
