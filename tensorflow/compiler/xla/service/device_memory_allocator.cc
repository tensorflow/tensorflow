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

namespace xla {

StreamExecutorMemoryAllocator::StreamExecutorMemoryAllocator(
    perftools::gputools::Platform* platform,
    tensorflow::gtl::ArraySlice<perftools::gputools::StreamExecutor*>
        stream_executors)
    : DeviceMemoryAllocator(platform),
      stream_executors_(stream_executors.begin(), stream_executors.end()) {}

StatusOr<perftools::gputools::DeviceMemoryBase>
StreamExecutorMemoryAllocator::Allocate(int device_ordinal, uint64 size,
                                        bool retry_on_failure) {
  if (size == 0) {
    return perftools::gputools::DeviceMemoryBase(nullptr, 0);
  }
  TF_ASSIGN_OR_RETURN(perftools::gputools::StreamExecutor * stream_executor,
                      GetStreamExecutor(device_ordinal));
  return stream_executor->AllocateArray<uint8>(size);
}

tensorflow::Status StreamExecutorMemoryAllocator::Deallocate(
    int device_ordinal, perftools::gputools::DeviceMemoryBase* mem) {
  if (!mem->is_null()) {
    TF_ASSIGN_OR_RETURN(perftools::gputools::StreamExecutor * stream_executor,
                        GetStreamExecutor(device_ordinal));
    // We make a local copy of 'mem' so the original is not zeroed out by the
    // Deallocate() call below. This gives us a better chance of
    // catching double-free bugs, since Deallocate silently succeeds for null
    // values.
    perftools::gputools::DeviceMemoryBase mem_copy(*mem);
    stream_executor->Deallocate(&mem_copy);
  }
  return tensorflow::Status::OK();
}

StatusOr<perftools::gputools::StreamExecutor*>
StreamExecutorMemoryAllocator::GetStreamExecutor(int device_ordinal) {
  if (device_ordinal < 0) {
    return InvalidArgument("device ordinal value (%d) must be non-negative",
                           device_ordinal);
  }
  if (device_ordinal >= stream_executors_.size()) {
    return InvalidArgument(
        "device ordinal value (%d) >= number of devices (%zu)", device_ordinal,
        stream_executors_.size());
  }
  if (stream_executors_[device_ordinal] == nullptr) {
    return NotFound("Device %s:%d present but not supported",
                    platform()->Name().c_str(), device_ordinal);
  }
  return stream_executors_[device_ordinal];
}

}  // namespace xla
