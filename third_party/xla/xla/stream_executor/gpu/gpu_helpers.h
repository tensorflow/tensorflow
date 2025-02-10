/* Copyright 2019 The OpenXLA Authors.

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

// Common helper functions used for dealing with CUDA API datatypes.
//
// These are typically placed here for use by multiple source components (for
// example, BLAS and executor components).

#ifndef XLA_STREAM_EXECUTOR_GPU_GPU_HELPERS_H_
#define XLA_STREAM_EXECUTOR_GPU_GPU_HELPERS_H_

#include <stddef.h>

#include "xla/stream_executor/device_memory.h"

namespace stream_executor {

namespace gpu {

// Converts a const DeviceMemory reference to its underlying typed pointer in
// CUDA device memory.
template <typename T>
const T* GpuMemory(const DeviceMemory<T>& mem) {
  return static_cast<const T*>(mem.opaque());
}

// Converts a (non-const) DeviceMemory pointer reference to its underlying typed
// pointer in CUDA device memory.
template <typename T>
T* GpuMemoryMutable(DeviceMemory<T>* mem) {
  return static_cast<T*>(mem->opaque());
}

}  // namespace gpu
}  // namespace stream_executor

#endif  // XLA_STREAM_EXECUTOR_GPU_GPU_HELPERS_H_
