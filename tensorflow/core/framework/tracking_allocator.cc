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

#include "tensorflow/core/framework/tracking_allocator.h"

#include "tensorflow/core/platform/env.h"
#include "tensorflow/core/platform/logging.h"

namespace tensorflow {

TrackingAllocator::TrackingAllocator(Allocator* allocator, bool track_sizes)
    : allocator_(allocator),
      ref_(1),
      allocated_(0),
      high_watermark_(0),
      total_bytes_(0),
      track_sizes_locally_(track_sizes && !allocator_->TracksAllocationSizes()),
      next_allocation_id_(0) {}

void* TrackingAllocator::AllocateRaw(
    size_t alignment, size_t num_bytes,
    const AllocationAttributes& allocation_attr) {
  void* ptr = allocator_->AllocateRaw(alignment, num_bytes, allocation_attr);
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
      allocations_.emplace_back(allocated_bytes, Env::Default()->NowMicros());
      ++ref_;
    }
  } else if (track_sizes_locally_) {
    // Call the underlying allocator to try to get the allocated size
    // whenever possible, even when it might be slow. If this fails,
    // use the requested size as an approximation.
    size_t allocated_bytes = allocator_->AllocatedSizeSlow(ptr);
    allocated_bytes = std::max(num_bytes, allocated_bytes);
    mutex_lock lock(mu_);
    next_allocation_id_ += 1;
    Chunk chunk = {num_bytes, allocated_bytes, next_allocation_id_};
    in_use_.emplace(std::make_pair(ptr, chunk));
    allocated_ += allocated_bytes;
    high_watermark_ = std::max(high_watermark_, allocated_);
    total_bytes_ += allocated_bytes;
    allocations_.emplace_back(allocated_bytes, Env::Default()->NowMicros());
    ++ref_;
  } else {
    mutex_lock lock(mu_);
    total_bytes_ += num_bytes;
    allocations_.emplace_back(num_bytes, Env::Default()->NowMicros());
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
  } else if (track_sizes_locally_) {
    mutex_lock lock(mu_);
    auto itr = in_use_.find(ptr);
    if (itr != in_use_.end()) {
      tracks_allocation_sizes = true;
      allocated_bytes = (*itr).second.allocated_size;
      in_use_.erase(itr);
    }
  }
  Allocator* allocator = allocator_;
  {
    mutex_lock lock(mu_);
    if (tracks_allocation_sizes) {
      CHECK_GE(allocated_, allocated_bytes);
      allocated_ -= allocated_bytes;
      allocations_.emplace_back(-allocated_bytes, Env::Default()->NowMicros());
    }
    should_delete = UnRef();
  }
  allocator->DeallocateRaw(ptr);
  if (should_delete) {
    delete this;
  }
}

bool TrackingAllocator::TracksAllocationSizes() {
  return track_sizes_locally_ || allocator_->TracksAllocationSizes();
}

size_t TrackingAllocator::RequestedSize(void* ptr) {
  if (track_sizes_locally_) {
    mutex_lock lock(mu_);
    auto it = in_use_.find(ptr);
    if (it != in_use_.end()) {
      return (*it).second.requested_size;
    }
    return 0;
  } else {
    return allocator_->RequestedSize(ptr);
  }
}

size_t TrackingAllocator::AllocatedSize(void* ptr) {
  if (track_sizes_locally_) {
    mutex_lock lock(mu_);
    auto it = in_use_.find(ptr);
    if (it != in_use_.end()) {
      return (*it).second.allocated_size;
    }
    return 0;
  } else {
    return allocator_->AllocatedSize(ptr);
  }
}

int64 TrackingAllocator::AllocationId(void* ptr) {
  if (track_sizes_locally_) {
    mutex_lock lock(mu_);
    auto it = in_use_.find(ptr);
    if (it != in_use_.end()) {
      return (*it).second.allocation_id;
    }
    return 0;
  } else {
    return allocator_->AllocationId(ptr);
  }
}

void TrackingAllocator::GetStats(AllocatorStats* stats) {
  allocator_->GetStats(stats);
}

std::tuple<size_t, size_t, size_t> TrackingAllocator::GetSizes() {
  size_t high_watermark;
  size_t total_bytes;
  size_t still_live_bytes;
  {
    mutex_lock lock(mu_);
    high_watermark = high_watermark_;
    total_bytes = total_bytes_;
    still_live_bytes = allocated_;
  }
  return std::make_tuple(total_bytes, high_watermark, still_live_bytes);
}

gtl::InlinedVector<AllocRecord, 4> TrackingAllocator::GetRecordsAndUnRef() {
  bool should_delete;
  gtl::InlinedVector<AllocRecord, 4> allocations;
  {
    mutex_lock lock(mu_);
    allocations.swap(allocations_);
    should_delete = UnRef();
  }
  if (should_delete) {
    delete this;
  }
  return allocations;
}

bool TrackingAllocator::UnRef() {
  CHECK_GE(ref_, 1);
  --ref_;
  return (ref_ == 0);
}

}  // end namespace tensorflow
