/* Copyright 2024 The OpenXLA Authors.

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

#ifndef XLA_STREAM_EXECUTOR_GPU_SCOPED_UPDATE_MODE_H_
#define XLA_STREAM_EXECUTOR_GPU_SCOPED_UPDATE_MODE_H_

namespace stream_executor::gpu {

// RAII wrapper for `GpuCommandBuffer::ActivateUpdateMode` that enables updates
// of nested command buffers.
class ScopedUpdateMode {
 public:
  virtual ~ScopedUpdateMode() = 0;
};

inline ScopedUpdateMode::~ScopedUpdateMode() = default;

}  // namespace stream_executor::gpu

#endif  // XLA_STREAM_EXECUTOR_GPU_SCOPED_UPDATE_MODE_H_
