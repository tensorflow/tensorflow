#include "tensorflow/stream_executor/temporary_device_memory.h"

#include "tensorflow/stream_executor/stream.h"

namespace perftools {
namespace gputools {

TemporaryDeviceMemoryBase::~TemporaryDeviceMemoryBase() {
  parent_->temporary_memory_manager()->MarkFinalized(device_memory_,
                                                     allocation_generation_,
                                                     /*must_exist=*/false);
}

DeviceMemoryBase* TemporaryDeviceMemoryBase::mutable_device_memory() {
  DCHECK(!IsFinalized())
      << "should not access device memory after finalization";
  return &device_memory_;
}

const DeviceMemoryBase& TemporaryDeviceMemoryBase::device_memory() const {
  DCHECK(!IsFinalized())
      << "should not access device memory after finalization";
  return device_memory_;
}

void TemporaryDeviceMemoryBase::Finalize() {
  DCHECK(!IsFinalized()) << "should not finalize more than once";
  parent_->temporary_memory_manager()->MarkFinalized(device_memory_,
                                                     allocation_generation_,
                                                     /*must_exist=*/true);
}

bool TemporaryDeviceMemoryBase::IsFinalized() const {
  return parent_->temporary_memory_manager()->IsFinalized(
      device_memory_, allocation_generation_);
}

bool TemporaryDeviceMemoryBase::IsAllocated() const {
  return parent_->temporary_memory_manager()->HasAllocated(
      device_memory_, allocation_generation_);
}

TemporaryDeviceMemoryBase::TemporaryDeviceMemoryBase(
    Stream* parent, DeviceMemoryBase device_memory,
    uint64 allocation_generation)
    : device_memory_(device_memory),
      allocation_generation_(allocation_generation),
      parent_(parent) {
  DCHECK(IsAllocated());
}

}  // namespace gputools
}  // namespace perftools
