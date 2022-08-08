/* Copyright 2015 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/compiler/xla/stream_executor/temporary_memory_manager.h"

#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"
#include "tensorflow/compiler/xla/stream_executor/platform/logging.h"
#include "tensorflow/compiler/xla/stream_executor/stream.h"
#include "tensorflow/compiler/xla/stream_executor/stream_executor_pimpl.h"

namespace stream_executor {
namespace internal {

void TemporaryMemoryManager::ForceDeallocateAll() {
  absl::MutexLock lock(&mutex_);
  VLOG(1) << "force-deallocating " << records_.size() << " remaining records";
  for (auto it = records_.begin(); it != records_.end(); ++it) {
    DeviceMemoryBase device_memory = it->first;
    stream_->parent()->Deallocate(&device_memory);
  }
}

void TemporaryMemoryManager::MarkFinalized(
    const DeviceMemoryBase& device_memory, uint64_t generation,
    bool must_exist) {
  absl::MutexLock lock(&mutex_);
  auto it = records_.find(device_memory);
  if (it == records_.end()) {
    if (must_exist) {
      LOG(FATAL) << "attempted to mark finalization for temporary "
                    "memory that does not exist";
    }
    return;
  }
  it->second.finalized = true;
}

void TemporaryMemoryManager::DeallocateFinalizedTemporaries() {
  absl::MutexLock lock(&mutex_);
  int deallocated_count = 0;
  for (auto it = records_.begin(); it != records_.end();) {
    if (it->second.finalized) {
      DeviceMemoryBase device_memory = it->first;
      stream_->parent()->Deallocate(&device_memory);
      ++deallocated_count;
      it = records_.erase(it);
    } else {
      ++it;
    }
  }
  VLOG(1) << "deallocated " << deallocated_count << " finalized temporaries";
}

bool TemporaryMemoryManager::IsFinalized(const DeviceMemoryBase& device_memory,
                                         uint64_t allocation_generation) const {
  absl::MutexLock lock(&mutex_);
  auto it = records_.find(device_memory);
  if (it == records_.end()) {
    return true;  // If there's no record present it's vacuously finalized.
  }

  if (it->second.allocation_generation == allocation_generation) {
    return it->second.finalized;
  }

  // If the allocation generation did not match, it's vacuously true.
  return true;
}

bool TemporaryMemoryManager::HasAllocated(const DeviceMemoryBase& device_memory,
                                          uint64_t generation) const {
  absl::MutexLock lock(&mutex_);
  auto it = records_.find(device_memory);
  if (it == records_.end()) {
    return false;
  }
  return it->second.allocation_generation == generation;
}

port::StatusOr<std::unique_ptr<TemporaryDeviceMemoryBase>>
TemporaryMemoryManager::AllocateArrayBase(uint64_t element_count,
                                          uint64_t element_size) {
  uint64_t byte_size = element_count * element_size;
  DeviceMemoryBase device_memory =
      stream_->parent()->AllocateArray<uint8>(byte_size);
  if (device_memory == nullptr) {
    return port::Status(port::error::RESOURCE_EXHAUSTED,
                        absl::StrCat("could not allocate temporary memory of ",
                                     byte_size, " bytes"));
  }

  uint64_t generation;

  // Add the record before instantiating the device memory instance so we can
  // check the allocation invariant at TemporaryDeviceMemory construction time.
  {
    absl::MutexLock lock(&mutex_);
    generation = ++generation_;
    DCHECK(records_.find(device_memory) == records_.end());
    records_[device_memory] = {generation,
                               /*finalized=*/false};
  }

  VLOG(1) << absl::StreamFormat(
      "stream %p allocated temporary device memory at %p (size %u) in "
      "generation %u",
      stream_, device_memory.opaque(), byte_size, generation);
  std::unique_ptr<TemporaryDeviceMemoryBase> result(
      new TemporaryDeviceMemoryBase(stream_, device_memory, generation));
  return std::move(result);
}

}  // namespace internal
}  // namespace stream_executor
