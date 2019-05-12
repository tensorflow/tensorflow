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

#include "tensorflow/stream_executor/device_memory_allocator.h"

#include <string>

#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"
#include "tensorflow/core/lib/strings/numbers.h"

namespace stream_executor {

StreamExecutorMemoryAllocator::StreamExecutorMemoryAllocator(
    const Platform* platform,
    absl::Span<StreamExecutor* const> stream_executors)
    : DeviceMemoryAllocator(platform),
      stream_executors_(stream_executors.begin(), stream_executors.end()) {}

port::StatusOr<OwningDeviceMemory> StreamExecutorMemoryAllocator::Allocate(
    int device_ordinal, uint64 size, bool retry_on_failure) {
  port::StatusOr<StreamExecutor*> stream_executor_or =
      GetStreamExecutor(device_ordinal);
  TF_RETURN_IF_ERROR(stream_executor_or.status());
  DeviceMemoryBase result =
      stream_executor_or.ValueOrDie()->AllocateArray<uint8>(size);
  if (size > 0 && result == nullptr) {
    return tensorflow::errors::ResourceExhausted(
        "Failed to allocate request for %s (%uB) on device ordinal %d",
        tensorflow::strings::HumanReadableNumBytes(size), size, device_ordinal);
  }
  VLOG(3) << absl::StreamFormat(
      "Allocated %s (%uB) on device ordinal %d: %p",
      tensorflow::strings::HumanReadableNumBytes(size), size, device_ordinal,
      result.opaque());
  return OwningDeviceMemory(result, device_ordinal, this);
}

port::Status StreamExecutorMemoryAllocator::Deallocate(int device_ordinal,
                                                       DeviceMemoryBase mem) {
  if (!mem.is_null()) {
    port::StatusOr<StreamExecutor*> stream_executor_or =
        GetStreamExecutor(device_ordinal);
    TF_RETURN_IF_ERROR(stream_executor_or.status());
    VLOG(3) << absl::StreamFormat("Freeing %p on device ordinal %d",
                                  mem.opaque(), device_ordinal);
    stream_executor_or.ValueOrDie()->Deallocate(&mem);
  }
  return port::Status::OK();
}

port::StatusOr<StreamExecutor*>
StreamExecutorMemoryAllocator::GetStreamExecutor(int device_ordinal) {
  if (device_ordinal < 0) {
    return tensorflow::errors::InvalidArgument(
        "device ordinal value (%d) must be non-negative", device_ordinal);
  }
  if (device_ordinal >= stream_executors_.size()) {
    return tensorflow::errors::InvalidArgument(
        "device ordinal value (%d) >= number of devices (%u)", device_ordinal,
        stream_executors_.size());
  }
  if (stream_executors_[device_ordinal] == nullptr) {
    return tensorflow::errors::NotFound(
        absl::StrFormat("Device %s:%d present but not supported",
                        platform()->Name(), device_ordinal));
  }
  return stream_executors_[device_ordinal];
}

bool StreamExecutorMemoryAllocator::AllowsAsynchronousDeallocation() const {
  return false;
}

}  // namespace stream_executor
