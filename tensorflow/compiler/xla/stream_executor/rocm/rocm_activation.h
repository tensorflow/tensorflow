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

// This file contains APIs that assume a StreamExecutor is backed by ROCM.
// It reaches into the ROCM implementation to activate an underlying ROCM
// context.
//
// Having this file separate from rocm/rocm_gpu_executor.h means that dependent
// code does not also have to depend on rocm.h.

#ifndef TENSORFLOW_COMPILER_XLA_STREAM_EXECUTOR_ROCM_ROCM_ACTIVATION_H_
#define TENSORFLOW_COMPILER_XLA_STREAM_EXECUTOR_ROCM_ROCM_ACTIVATION_H_

#include "tensorflow/compiler/xla/stream_executor/gpu/gpu_activation.h"

namespace stream_executor {

class StreamExecutor;

namespace rocm {

using ScopedActivateExecutorContext = gpu::ScopedActivateExecutorContext;

}  // namespace rocm
}  // namespace stream_executor

#endif  // TENSORFLOW_COMPILER_XLA_STREAM_EXECUTOR_ROCM_ROCM_ACTIVATION_H_
