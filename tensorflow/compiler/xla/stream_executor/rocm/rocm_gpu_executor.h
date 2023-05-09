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

// The ROCm implementation of the StreamExecutorInterface functionality.
// ROCm inclusions are ideally confined to this implementation file.
//
// The notions from the StreamExecutor basically correspond to the ROCm streams
// programming model provided by the librocm.so driver APIs, so we don't have
// to do much more than wrap the calls to the libraries appropriately.
#ifndef TENSORFLOW_COMPILER_XLA_STREAM_EXECUTOR_CUDA_ROCM_GPU_EXECUTOR_H_
#define TENSORFLOW_COMPILER_XLA_STREAM_EXECUTOR_CUDA_ROCM_GPU_EXECUTOR_H_

#include "tensorflow/compiler/xla/stream_executor/gpu/gpu_executor.h"

namespace stream_executor {
namespace rocm {

using ROCMExecutor = gpu::GpuExecutor;

}  // namespace rocm
}  // namespace stream_executor

#endif  // TENSORFLOW_COMPILER_XLA_STREAM_EXECUTOR_CUDA_ROCM_GPU_EXECUTOR_H_
