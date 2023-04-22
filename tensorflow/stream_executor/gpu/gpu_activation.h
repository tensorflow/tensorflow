/* Copyright 2019 The TensorFlow Authors. All Rights Reserved.

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

// This file contains APIs that assume a StreamExecutor is backed by CUDA.
// It reaches into the CUDA implementation to activate an underlying CUDA
// context.
//
// Having this file separate from gpu/gpu_executor.h means that dependent
// code does not also have to depend on cuda.h.

#ifndef TENSORFLOW_STREAM_EXECUTOR_GPU_GPU_ACTIVATION_H_
#define TENSORFLOW_STREAM_EXECUTOR_GPU_GPU_ACTIVATION_H_

#include "tensorflow/stream_executor/platform/port.h"

namespace stream_executor {

class StreamExecutor;

namespace gpu {

class GpuExecutor;
class ScopedActivateContext;

// Activates a CUDA context within an enclosing scope.
class ScopedActivateExecutorContext {
 public:
  // Form that takes a CUDA executor implementation.
  explicit ScopedActivateExecutorContext(GpuExecutor* gpu_exec);

  // Form that takes a pImpl executor and extracts a CUDA implementation --
  // fatal failure if it is not CUDA inside.
  explicit ScopedActivateExecutorContext(StreamExecutor* stream_exec);

  ScopedActivateExecutorContext(ScopedActivateExecutorContext&& other);

  ~ScopedActivateExecutorContext();

 private:
  // The cuda.h-using datatype that we wrap.
  ScopedActivateContext* driver_scoped_activate_context_;

  SE_DISALLOW_COPY_AND_ASSIGN(ScopedActivateExecutorContext);
};

}  // namespace gpu
}  // namespace stream_executor

#endif  // TENSORFLOW_STREAM_EXECUTOR_GPU_GPU_ACTIVATION_H_
