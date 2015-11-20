/* Copyright 2015 Google Inc. All Rights Reserved.

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

#include "tensorflow/stream_executor/cuda/cuda_activation.h"

#include "tensorflow/stream_executor/cuda/cuda_driver.h"
#include "tensorflow/stream_executor/stream_executor.h"
#include "tensorflow/stream_executor/stream_executor_internal.h"

namespace perftools {
namespace gputools {
namespace cuda {

CUcontext ExtractCudaContext(CUDAExecutor *cuda_exec);
CUDAExecutor *ExtractCudaExecutor(StreamExecutor *stream_exec);

ScopedActivateExecutorContext::ScopedActivateExecutorContext(
    CUDAExecutor *cuda_exec, MultiOpActivation moa)
    : cuda_exec_(cuda_exec),
      driver_scoped_activate_context_(
          new ScopedActivateContext{ExtractCudaContext(cuda_exec), moa}) {}

ScopedActivateExecutorContext::ScopedActivateExecutorContext(
    StreamExecutor *stream_exec, MultiOpActivation moa)
    : ScopedActivateExecutorContext(ExtractCudaExecutor(stream_exec), moa) {}

ScopedActivateExecutorContext::~ScopedActivateExecutorContext() {
  delete static_cast<ScopedActivateContext *>(driver_scoped_activate_context_);
}

}  // namespace cuda
}  // namespace gputools
}  // namespace perftools
