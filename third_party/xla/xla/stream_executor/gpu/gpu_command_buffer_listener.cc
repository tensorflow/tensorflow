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

#include "xla/stream_executor/gpu/gpu_command_buffer_listener.h"

#include <atomic>

namespace stream_executor::gpu {
namespace {

static std::atomic<GpuCommandBufferListener*> global_listener{nullptr};

}  // namespace

bool RegisterGpuCommandBufferListener(GpuCommandBufferListener* listener) {
  if (listener == nullptr) {
    return false;
  }
  GpuCommandBufferListener* expected = nullptr;
  return global_listener.compare_exchange_strong(expected, listener,
                                                 std::memory_order_acq_rel);
}

bool UnregisterGpuCommandBufferListener(GpuCommandBufferListener* listener) {
  if (listener == nullptr) {
    return false;
  }
  GpuCommandBufferListener* expected = listener;
  return global_listener.compare_exchange_strong(expected, nullptr,
                                                 std::memory_order_acq_rel);
}

GpuCommandBufferListener* GetGpuCommandBufferListener() {
  return global_listener.load(std::memory_order_acquire);
}

}  // namespace stream_executor::gpu
