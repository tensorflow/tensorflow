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

#include "tensorflow/compiler/xla/service/device_memory_allocator.h"

#include <string>

#include "tensorflow/compiler/xla/status_macros.h"
#include "tensorflow/compiler/xla/types.h"
#include "tensorflow/compiler/xla/util.h"
#include "tensorflow/core/lib/strings/numbers.h"

namespace xla {

StreamExecutorMemoryAllocator::StreamExecutorMemoryAllocator(
    const se::Platform* platform,
    absl::Span<se::StreamExecutor* const> stream_executors)
    : DeviceMemoryAllocator(platform),
      stream_executors_(stream_executors.begin(), stream_executors.end()) {}

StatusOr<OwningDeviceMemory> StreamExecutorMemoryAllocator::Allocate(
    int device_ordinal, uint64 size, bool retry_on_failure) {
  TF_ASSIGN_OR_RETURN(se::StreamExecutor * stream_executor,
                      GetStreamExecutor(device_ordinal));
  se::DeviceMemoryBase result = stream_executor->AllocateArray<uint8>(size);
  if (size > 0 && result == nullptr) {
    return ResourceExhausted(
        "Failed to allocate request for %s (%uB) on device ordinal %d",
        tensorflow::strings::HumanReadableNumBytes(size), size, device_ordinal);
  }
  VLOG(3) << absl::StreamFormat(
      "Allocated %s (%uB) on device ordinal %d: %p",
      tensorflow::strings::HumanReadableNumBytes(size), size, device_ordinal,
      result.opaque());
  return OwningDeviceMemory(result, device_ordinal, this);
}

Status StreamExecutorMemoryAllocator::Deallocate(int device_ordinal,
                                                 se::DeviceMemoryBase mem) {
  if (!mem.is_null()) {
    TF_ASSIGN_OR_RETURN(se::StreamExecutor * stream_executor,
                        GetStreamExecutor(device_ordinal));
    VLOG(3) << absl::StreamFormat("Freeing %p on device ordinal %d",
                                  mem.opaque(), device_ordinal);
    stream_executor->Deallocate(&mem);
  }
  return Status::OK();
}

StatusOr<se::StreamExecutor*> StreamExecutorMemoryAllocator::GetStreamExecutor(
    int device_ordinal) {
  if (device_ordinal < 0) {
    return InvalidArgument("device ordinal value (%d) must be non-negative",
                           device_ordinal);
  }
  if (device_ordinal >= stream_executors_.size()) {
    return InvalidArgument(
        "device ordinal value (%d) >= number of devices (%u)", device_ordinal,
        stream_executors_.size());
  }
  if (stream_executors_[device_ordinal] == nullptr) {
    return NotFound("Device %s:%d present but not supported",
                    platform()->Name(), device_ordinal);
  }
  return stream_executors_[device_ordinal];
}

bool StreamExecutorMemoryAllocator::AllowsAsynchronousDeallocation() const {
  return false;
}

}  // namespace xla
