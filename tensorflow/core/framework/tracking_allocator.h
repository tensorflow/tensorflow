/* Copyright 2015 Google Inc. All Rights Reserved.

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

#ifndef TENSORFLOW_FRAMEWORK_TRACKING_ALLOCATOR_H_
#define TENSORFLOW_FRAMEWORK_TRACKING_ALLOCATOR_H_

#include <unordered_map>
#include "tensorflow/core/framework/allocator.h"
#include "tensorflow/core/lib/core/refcount.h"
#include "tensorflow/core/platform/mutex.h"
#include "tensorflow/core/platform/thread_annotations.h"
#include "tensorflow/core/platform/types.h"

namespace tensorflow {

// TrackingAllocator is a wrapper for an Allocator. It keeps a running
// count of the number of bytes allocated through the wrapper. It is
// used by the Executor to "charge" allocations to particular Op
// executions. Each Op gets a separate TrackingAllocator wrapper
// around the underlying allocator.
//
// The implementation assumes the invariant that all calls to
// AllocateRaw by an Op (or work items spawned by the Op) will occur
// before the Op's Compute method returns. Thus the high watermark is
// established once Compute returns.
//
// DeallocateRaw can be called long after the Op has finished,
// e.g. when an output tensor is deallocated, and the wrapper cannot
// be deleted until the last of these calls has occurred.  The
// TrackingAllocator keeps track of outstanding calls using a
// reference count, and deletes itself once the last call has been
// received and the high watermark has been retrieved.
class TrackingAllocator : public Allocator {
 public:
  explicit TrackingAllocator(Allocator* allocator, bool track_ids);
  string Name() override { return allocator_->Name(); }
  void* AllocateRaw(size_t alignment, size_t num_bytes) override {
    return AllocateRaw(alignment, num_bytes, AllocationAttributes());
  }
  void* AllocateRaw(size_t alignment, size_t num_bytes,
                    const AllocationAttributes& allocation_attr) override;
  void DeallocateRaw(void* ptr) override;
  bool TracksAllocationSizes() override;
  size_t RequestedSize(void* ptr) override;
  size_t AllocatedSize(void* ptr) override;
  int64 AllocationId(void* ptr) override;
  void GetStats(AllocatorStats* stats) override;

  // If the underlying allocator tracks allocation sizes, this returns
  // a pair where the first value is the total number of bytes
  // allocated through this wrapper, and the second value is the high
  // watermark of bytes allocated through this wrapper. If the
  // underlying allocator does not track allocation sizes the first
  // value is the total number of bytes requested through this wrapper
  // and the second is 0.
  //
  // After GetSizesAndUnref is called, the only further calls allowed
  // on this wrapper are calls to DeallocateRaw with pointers that
  // were allocated by this wrapper and have not yet been
  // deallocated. After this call completes and all allocated pointers
  // have been deallocated the wrapper will delete itself.
  std::pair<size_t, size_t> GetSizesAndUnRef();

 private:
  ~TrackingAllocator() override {}
  bool UnRef() EXCLUSIVE_LOCKS_REQUIRED(mu_);

  Allocator* allocator_;  // not owned.
  mutex mu_;
  // the number of calls to AllocateRaw that have not yet been matched
  // by a corresponding call to DeAllocateRaw, plus 1 if the Executor
  // has not yet read out the high watermark.
  int ref_ GUARDED_BY(mu_);
  // the current number of outstanding bytes that have been allocated
  // by this wrapper, or 0 if the underlying allocator does not track
  // allocation sizes.
  size_t allocated_ GUARDED_BY(mu_);
  // the maximum number of outstanding bytes that have been allocated
  // by this wrapper, or 0 if the underlying allocator does not track
  // allocation sizes.
  size_t high_watermark_ GUARDED_BY(mu_);
  // the total number of bytes that have been allocated by this
  // wrapper if the underlying allocator tracks allocation sizes,
  // otherwise the total number of bytes that have been requested by
  // this allocator.
  size_t total_bytes_ GUARDED_BY(mu_);

  // Track allocations locally if requested in the constructor and the
  // underlying allocator doesn't already do it for us.
  const bool track_sizes_locally_;
  struct Chunk {
    size_t requested_size;
    size_t allocated_size;
    int64 allocation_id;
  };
  std::unordered_map<void*, Chunk> in_use_ GUARDED_BY(mu_);
  int64 next_allocation_id_ GUARDED_BY(mu_);
};

}  // end namespace tensorflow

#endif  // TENSORFLOW_FRAMEWORK_TRACKING_ALLOCATOR_H_
