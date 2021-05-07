/* Copyright 2017 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/compiler/xla/service/gpu/infeed_manager.h"

#include "absl/memory/memory.h"

#if GOOGLE_CUDA || TENSORFLOW_USE_ROCM
#include "tensorflow/compiler/xla/service/gpu/xla_executor_state.h"
#include "tensorflow/stream_executor/gpu/gpu_executor.h"
#endif  // GOOGLE_CUDA || TENSORFLOW_USE_ROCM

namespace xla {
namespace gpu {

InfeedManager::InfeedManager(se::StreamExecutor *executor)
    : stream_(absl::make_unique<se::Stream>(executor)) {
  stream_->Init();
}

InfeedManager *GetOrCreateInfeedManager(se::StreamExecutor *executor) {
#if GOOGLE_CUDA || TENSORFLOW_USE_ROCM
  stream_executor::gpu::GpuExecutor *gpu_executor =
      stream_executor::gpu::ExtractGpuExecutor(executor);
  auto *xla_state =
      gpu_executor->getOrCreateXLAState<GpuExecutorXLAState>(executor);
  return xla_state->getOrCreateInfeedManager(executor);
#else   // GOOGLE_CUDA || TENSORFLOW_USE_ROCM
  return nullptr;
#endif  // GOOGLE_CUDA || TENSORFLOW_USE_ROCM
}

}  // namespace gpu
}  // namespace xla
