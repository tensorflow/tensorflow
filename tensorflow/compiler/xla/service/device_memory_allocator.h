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

#ifndef TENSORFLOW_COMPILER_XLA_SERVICE_DEVICE_MEMORY_ALLOCATOR_H_
#define TENSORFLOW_COMPILER_XLA_SERVICE_DEVICE_MEMORY_ALLOCATOR_H_

#include <vector>

#include "tensorflow/compiler/xla/statusor.h"
#include "tensorflow/compiler/xla/types.h"
#include "tensorflow/core/lib/gtl/array_slice.h"
#include "tensorflow/core/platform/stream_executor_no_cuda.h"
#include "tensorflow/core/platform/types.h"

namespace xla {

// Interface for device memory allocators used within the XLA service. An
// allocator is responsible for allocating memory on all devices of a particular
// platform.
class DeviceMemoryAllocator {
 public:
  // Parameter platform indicates which platform the allocator allocates memory
  // on. Must be non-null.
  explicit DeviceMemoryAllocator(const perftools::gputools::Platform* platform)
      : platform_(platform) {}
  virtual ~DeviceMemoryAllocator() {}

  // 'retry_on_failure': If false, and the first attempt to allocate the memory
  // fails, the allocation should return immediately without retrying.
  // An example use case is optional scratch spaces where a failure
  // has only performance impact.
  // Allocate() should return a null pointer for a size-0 allocation.
  // Deallocate() must be a no-op for null pointers.
  virtual StatusOr<perftools::gputools::DeviceMemoryBase> Allocate(
      int device_ordinal, uint64 size, bool retry_on_failure = true) = 0;
  virtual tensorflow::Status Deallocate(
      int device_ordinal, perftools::gputools::DeviceMemoryBase* mem) = 0;

  // Return the platform that the allocator allocates memory on.
  const perftools::gputools::Platform* platform() const { return platform_; }

  // Can we call Deallocate() as soon as a computation has been scheduled on
  // a stream, or do we have to wait for the computation to complete first?
  virtual bool AllowsAsynchronousDeallocation() const = 0;

 protected:
  const perftools::gputools::Platform* platform_;
};

// Default memory allocator for a platform which uses
// StreamExecutor::Allocate/Deallocate.
class StreamExecutorMemoryAllocator : public DeviceMemoryAllocator {
 public:
  StreamExecutorMemoryAllocator(
      const perftools::gputools::Platform* platform,
      tensorflow::gtl::ArraySlice<perftools::gputools::StreamExecutor*>
          stream_executors);

  StatusOr<perftools::gputools::DeviceMemoryBase> Allocate(
      int device_ordinal, uint64 size, bool retry_on_failure = true) override;
  tensorflow::Status Deallocate(
      int device_ordinal, perftools::gputools::DeviceMemoryBase* mem) override;

  bool AllowsAsynchronousDeallocation() const override;

 private:
  StatusOr<perftools::gputools::StreamExecutor*> GetStreamExecutor(
      int device_ordinal);

  // A vector indexed by device ordinal of StreamExecutors for each device of
  // the allocator's platform type. If an element is nullptr, then the device
  // with the respective device ordinal is not supported by XLA.
  std::vector<perftools::gputools::StreamExecutor*> stream_executors_;
};

}  // namespace xla

#endif  // TENSORFLOW_COMPILER_XLA_SERVICE_DEVICE_MEMORY_ALLOCATOR_H_
