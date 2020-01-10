#include "tensorflow/core/framework/tracking_allocator.h"

#include "tensorflow/core/platform/logging.h"

namespace tensorflow {

TrackingAllocator::TrackingAllocator(Allocator* allocator)
    : allocator_(allocator),
      ref_(1),
      allocated_(0),
      high_watermark_(0),
      total_bytes_(0) {}

void* TrackingAllocator::AllocateRaw(size_t alignment, size_t num_bytes) {
  void* ptr = allocator_->AllocateRaw(alignment, num_bytes);
  // If memory is exhausted AllocateRaw returns nullptr, and we should
  // pass this through to the caller
  if (nullptr == ptr) {
    return ptr;
  }
  if (allocator_->TracksAllocationSizes()) {
    size_t allocated_bytes = allocator_->AllocatedSize(ptr);
    {
      mutex_lock lock(mu_);
      allocated_ += allocated_bytes;
      high_watermark_ = std::max(high_watermark_, allocated_);
      total_bytes_ += allocated_bytes;
      ++ref_;
    }
  } else {
    mutex_lock lock(mu_);
    total_bytes_ += num_bytes;
    ++ref_;
  }
  return ptr;
}

void TrackingAllocator::DeallocateRaw(void* ptr) {
  // freeing a null ptr is a no-op
  if (nullptr == ptr) {
    return;
  }
  bool should_delete;
  // fetch the following outside the lock in case the call to
  // AllocatedSize is slow
  bool tracks_allocation_sizes = allocator_->TracksAllocationSizes();
  size_t allocated_bytes = 0;
  if (tracks_allocation_sizes) {
    allocated_bytes = allocator_->AllocatedSize(ptr);
  }
  Allocator* allocator = allocator_;
  {
    mutex_lock lock(mu_);
    if (tracks_allocation_sizes) {
      CHECK_GE(allocated_, allocated_bytes);
      allocated_ -= allocated_bytes;
    }
    should_delete = UnRef();
  }
  allocator->DeallocateRaw(ptr);
  if (should_delete) {
    delete this;
  }
}

bool TrackingAllocator::TracksAllocationSizes() {
  return allocator_->TracksAllocationSizes();
}

size_t TrackingAllocator::RequestedSize(void* ptr) {
  return allocator_->RequestedSize(ptr);
}

size_t TrackingAllocator::AllocatedSize(void* ptr) {
  return allocator_->AllocatedSize(ptr);
}

std::pair<size_t, size_t> TrackingAllocator::GetSizesAndUnRef() {
  size_t high_watermark;
  size_t total_bytes;
  bool should_delete;
  {
    mutex_lock lock(mu_);
    high_watermark = high_watermark_;
    total_bytes = total_bytes_;
    should_delete = UnRef();
  }
  if (should_delete) {
    delete this;
  }
  return std::make_pair(total_bytes, high_watermark);
}

bool TrackingAllocator::UnRef() {
  CHECK_GE(ref_, 1);
  --ref_;
  return (ref_ == 0);
}

}  // end namespace tensorflow
