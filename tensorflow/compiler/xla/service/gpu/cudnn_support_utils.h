/* Copyright 2021 The TensorFlow Authors. All Rights Reserved.

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

#ifndef TENSORFLOW_COMPILER_XLA_SERVICE_GPU_CUDNN_SUPPORT_UTILS_H_
#define TENSORFLOW_COMPILER_XLA_SERVICE_GPU_CUDNN_SUPPORT_UTILS_H_

#include "tensorflow/compiler/xla/service/hlo_instructions.h"
#include "tensorflow/compiler/xla/stream_executor/device_description.h"
#include "tensorflow/compiler/xla/stream_executor/dnn.h"
#include "tensorflow/compiler/xla/util.h"
#include "tensorflow/compiler/xla/window_util.h"
#include "tensorflow/core/platform/status.h"

namespace xla {
namespace gpu {

// Indicates whether the `compute_capability` supports an optimized integer
// implementation of the given `conv` operation vectorized to `vector_size`.
//
// This function does not guarantee that a convolution will be padded and/or
// vectorized. It only checks that it is a valid candiate for such optimization.
StatusOr<bool> CudnnSupportsOptimizedIntegerConvolution(
    const se::CudaComputeCapability& compute_capability,
    HloCustomCallInstruction& conv, int vector_size);

}  // namespace gpu
}  // namespace xla

#endif  // TENSORFLOW_COMPILER_XLA_SERVICE_GPU_CUDNN_SUPPORT_UTILS_H_
