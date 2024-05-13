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

#ifndef XLA_STREAM_EXECUTOR_DEVICE_MEMORY_HANDLE_H_
#define XLA_STREAM_EXECUTOR_DEVICE_MEMORY_HANDLE_H_

#include <utility>

#include "xla/stream_executor/device_memory.h"
#include "xla/stream_executor/stream_executor_interface.h"

namespace stream_executor {

// This class will deallocate the held DeviceMemoryBase upon destruction.
class DeviceMemoryHandle {
 public:
  DeviceMemoryHandle() : memory_(), executor_(nullptr) {}

  // A helper constructor to generate a scoped device memory given an already
  // allocated memory and a stream executor.
  DeviceMemoryHandle(StreamExecutorInterface *executor,
                     DeviceMemoryBase memory);
  ~DeviceMemoryHandle();

  // Moves ownership of the memory from other to the constructed
  // object.
  DeviceMemoryHandle(DeviceMemoryHandle &&other) noexcept;

  // Moves ownership of the memory from other to this object.
  DeviceMemoryHandle &operator=(DeviceMemoryHandle &&other) noexcept;

  // Accessors for the DeviceMemoryBase.
  const DeviceMemoryBase &memory() const { return memory_; }
  DeviceMemoryBase *memory_ptr() { return &memory_; }

 private:
  // Frees the associated memory.
  void Free();

  DeviceMemoryBase memory_;            // Value we wrap with scoped-release.
  StreamExecutorInterface *executor_;  // Null if this object is inactive.
};
}  // namespace stream_executor

#endif  // XLA_STREAM_EXECUTOR_DEVICE_MEMORY_HANDLE_H_
