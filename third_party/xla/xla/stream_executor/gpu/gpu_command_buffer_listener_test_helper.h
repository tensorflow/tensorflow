/* Copyright 2026 The OpenXLA Authors.

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

#ifndef XLA_STREAM_EXECUTOR_GPU_GPU_COMMAND_BUFFER_LISTENER_TEST_HELPER_H_
#define XLA_STREAM_EXECUTOR_GPU_GPU_COMMAND_BUFFER_LISTENER_TEST_HELPER_H_

#include "xla/stream_executor/gpu/gpu_command_buffer_listener.h"

namespace stream_executor::gpu {

class ScopedGpuCommandBufferListenerOverrideForTesting {
 public:
  explicit ScopedGpuCommandBufferListenerOverrideForTesting(
      GpuCommandBufferListener* listener)
      : old_listener_(GpuCommandBufferListener::ExchangeForTesting(listener)) {}
  ~ScopedGpuCommandBufferListenerOverrideForTesting() {
    GpuCommandBufferListener::ExchangeForTesting(old_listener_);
  }

 private:
  GpuCommandBufferListener* old_listener_;
};

}  // namespace stream_executor::gpu

#endif  // XLA_STREAM_EXECUTOR_GPU_GPU_COMMAND_BUFFER_LISTENER_TEST_HELPER_H_
