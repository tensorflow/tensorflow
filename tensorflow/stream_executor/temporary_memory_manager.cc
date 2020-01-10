#include "tensorflow/stream_executor/temporary_memory_manager.h"

#include "tensorflow/stream_executor/platform/logging.h"
#include "tensorflow/stream_executor/lib/stringprintf.h"
#include "tensorflow/stream_executor/lib/ptr_util.h"
#include "tensorflow/stream_executor/stream.h"
#include "tensorflow/stream_executor/stream_executor_pimpl.h"

namespace perftools {
namespace gputools {
namespace internal {

void TemporaryMemoryManager::ForceDeallocateAll() {
  mutex_lock lock(mutex_);
  VLOG(1) << "force-deallocating " << records_.size() << " remaining records";
  for (auto it = records_.begin(); it != records_.end(); ++it) {
    DeviceMemoryBase device_memory = it->first;
    stream_->parent()->Deallocate(&device_memory);
  }
}

void TemporaryMemoryManager::MarkFinalized(
    const DeviceMemoryBase& device_memory, uint64 generation, bool must_exist) {
  mutex_lock lock(mutex_);
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
  mutex_lock lock(mutex_);
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
                                         uint64 allocation_generation) const {
  mutex_lock lock(mutex_);
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
                                          uint64 generation) const {
  mutex_lock lock(mutex_);
  auto it = records_.find(device_memory);
  if (it == records_.end()) {
    return false;
  }
  return it->second.allocation_generation == generation;
}

port::StatusOr<std::unique_ptr<TemporaryDeviceMemoryBase>>
TemporaryMemoryManager::AllocateArrayBase(uint64 element_count,
                                          uint64 element_size) {
  uint64 byte_size = element_count * element_size;
  DeviceMemoryBase device_memory =
      stream_->parent()->AllocateArray<uint8>(byte_size);
  if (device_memory == nullptr) {
    return port::Status(port::error::RESOURCE_EXHAUSTED,
                        port::StrCat("could not allocate temporary memory of ",
                                     byte_size, " bytes"));
  }

  uint64 generation;

  // Add the record before instantiating the device memory instance so we can
  // check the allocation invariant at TemporaryDeviceMemory construction time.
  {
    mutex_lock lock(mutex_);
    generation = ++generation_;
    DCHECK(records_.find(device_memory) == records_.end());
    records_[device_memory] = {generation,
                               /*finalized=*/false};
  }

  VLOG(1) << port::Printf(
      "stream %p allocated temporary device memory at %p (size %llu) in "
      "generation %llu",
      stream_, device_memory.opaque(), byte_size, generation);
  std::unique_ptr<TemporaryDeviceMemoryBase> result(
      new TemporaryDeviceMemoryBase(stream_, device_memory, generation));
  return std::move(result);
}

}  // namespace internal
}  // namespace gputools
}  // namespace perftools
