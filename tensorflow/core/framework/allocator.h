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

#ifndef TENSORFLOW_CORE_FRAMEWORK_ALLOCATOR_H_
#define TENSORFLOW_CORE_FRAMEWORK_ALLOCATOR_H_

#include <stdlib.h>

#include <functional>
#include <limits>

#include "absl/strings/string_view.h"
#include "absl/types/optional.h"
#include "tensorflow/core/framework/numeric_types.h"
#include "tensorflow/core/framework/type_traits.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/macros.h"
#include "tensorflow/core/platform/numa.h"
#include "tensorflow/core/platform/types.h"

namespace tensorflow {

class TensorShape;

// Attributes for a single allocation call. Different calls to the same
// allocator could potentially have different allocation attributes.
struct AllocationAttributes {
  AllocationAttributes() = default;

  AllocationAttributes(bool retry_on_failure, bool allocation_will_be_logged,
                       std::function<uint64()>* freed_by_func)
      : retry_on_failure(retry_on_failure),
        allocation_will_be_logged(allocation_will_be_logged),
        freed_by_func(freed_by_func) {}

  // If the first attempt to allocate the memory fails, the allocation should
  // wait and retry (with a timeout).
  //
  // This is usually set to true, but we may set it to false in cases where a
  // failure has only performance impact (e.g. optional scratch space
  // allocation).
  bool retry_on_failure = true;
  // If a Tensor is allocated without the following set to true, then
  // it is logged as an unknown allocation. During execution Tensors
  // should be allocated through the OpKernelContext which records
  // which Op is performing the allocation, and sets this flag to
  // true.
  bool allocation_will_be_logged = false;
  // EXPERIMENTAL: If provided, then evaluates to a timing count such that only
  // a memory chunk whose freed_at_count is at this value or earlier may be
  // returned.
  std::function<uint64()>* freed_by_func = nullptr;  // Not owned.

  TF_DISALLOW_COPY_AND_ASSIGN(AllocationAttributes);
};

// Annotations for memory profiling and debugging purpose. The runtime will
// cache the annotations in thread-local memory, and some allocators will try to
// tag allocations with the annotations.
struct MemoryDebugAnnotation {
  const char* pending_op_name = nullptr;
  int64 pending_step_id = 0;
  const char* pending_region_type = nullptr;
  int32 pending_data_type = 0;
  const TensorShape* pending_shape = nullptr;
};

// Wrapper class of MemoryDebugAnnotation for RAII.
class ScopedMemoryDebugAnnotation {
 public:
  static const MemoryDebugAnnotation& CurrentAnnotation() {
    return annotation_;
  }

  explicit ScopedMemoryDebugAnnotation(const char* op_name) {
    last_annotation_ = annotation_;
    CleanupAnnotation();
    annotation_.pending_op_name = op_name;
  }

  explicit ScopedMemoryDebugAnnotation(const char* op_name, int64 step_id) {
    last_annotation_ = annotation_;
    CleanupAnnotation();
    annotation_.pending_op_name = op_name;
    annotation_.pending_step_id = step_id;
  }

  // This constructor keeps the pending_op_name and pending_step_id from parent
  // (if any).  Otherwise it overwrites with op_name.
  explicit ScopedMemoryDebugAnnotation(const char* op_name,
                                       const char* region_type, int32 data_type,
                                       const TensorShape* shape) {
    last_annotation_ = annotation_;
    if (!annotation_.pending_op_name) {
      annotation_.pending_op_name = op_name;
    }
    annotation_.pending_region_type = region_type;
    annotation_.pending_data_type = data_type;
    annotation_.pending_shape = shape;
  }

  explicit ScopedMemoryDebugAnnotation(const char* op_name, int64 step_id,
                                       const char* region_type, int32 data_type,
                                       const TensorShape* shape) {
    last_annotation_ = annotation_;
    annotation_.pending_op_name = op_name;
    annotation_.pending_step_id = step_id;
    annotation_.pending_region_type = region_type;
    annotation_.pending_data_type = data_type;
    annotation_.pending_shape = shape;
  }

  ~ScopedMemoryDebugAnnotation() { annotation_ = last_annotation_; }

 private:
  void CleanupAnnotation() {
    annotation_.pending_op_name = nullptr;
    annotation_.pending_step_id = 0;
    annotation_.pending_region_type = nullptr;
    annotation_.pending_data_type = 0;
    annotation_.pending_shape = nullptr;
  }

  // Stores the current annotations.
  static thread_local MemoryDebugAnnotation annotation_;

  // Stores the previous values in case the annotations are nested.
  MemoryDebugAnnotation last_annotation_;

  TF_DISALLOW_COPY_AND_ASSIGN(ScopedMemoryDebugAnnotation);
};

// Runtime statistics collected by an allocator. Exactly the same as
// stream_executor::AllocatorStats, but independently defined to preserve the
// mutual independence of StreamExecutor and TensorFlow.
struct AllocatorStats {
  int64 num_allocs;          // Number of allocations.
  int64 bytes_in_use;        // Number of bytes in use.
  int64 peak_bytes_in_use;   // The peak bytes in use.
  int64 largest_alloc_size;  // The largest single allocation seen.

  // The upper limit of bytes of user allocatable device memory, if such a limit
  // is known.
  absl::optional<int64> bytes_limit;

  // Stats for reserved memory usage.
  int64 bytes_reserved;       // Number of bytes reserved.
  int64 peak_bytes_reserved;  // The peak number of bytes reserved.
  // The upper limit on the number bytes of reservable memory,
  // if such a limit is known.
  absl::optional<int64> bytes_reservable_limit;

  int64 largest_free_block_bytes;  // Largest free block's size in heap.

  AllocatorStats()
      : num_allocs(0),
        bytes_in_use(0),
        peak_bytes_in_use(0),
        largest_alloc_size(0),
        bytes_reserved(0),
        peak_bytes_reserved(0),
        largest_free_block_bytes(0) {}

  std::string DebugString() const;
};

// Allocator is an abstract interface for allocating and deallocating
// device memory.
class Allocator {
 public:
  // Align to 64 byte boundary.
  static constexpr size_t kAllocatorAlignment = 64;

  virtual ~Allocator();

  // Return a string identifying this allocator
  virtual std::string Name() = 0;

  // Return an uninitialized block of memory that is "num_bytes" bytes
  // in size.  The returned pointer is guaranteed to be aligned to a
  // multiple of "alignment" bytes.
  // REQUIRES: "alignment" is a power of 2.
  virtual void* AllocateRaw(size_t alignment, size_t num_bytes) = 0;

  // Return an uninitialized block of memory that is "num_bytes" bytes
  // in size with specified allocation attributes.  The returned pointer is
  // guaranteed to be aligned to a multiple of "alignment" bytes.
  // REQUIRES: "alignment" is a power of 2.
  virtual void* AllocateRaw(size_t alignment, size_t num_bytes,
                            const AllocationAttributes& allocation_attr) {
    // The default behavior is to use the implementation without any allocation
    // attributes.
    return AllocateRaw(alignment, num_bytes);
  }

  // Deallocate a block of memory pointer to by "ptr"
  // REQUIRES: "ptr" was previously returned by a call to AllocateRaw
  virtual void DeallocateRaw(void* ptr) = 0;

  // Returns true if this allocator tracks the sizes of allocations.
  // RequestedSize and AllocatedSize must be overridden if
  // TracksAllocationSizes is overridden to return true.
  virtual bool TracksAllocationSizes() const { return false; }

  // Returns true if this allocator allocates an opaque handle rather than the
  // requested number of bytes.
  //
  // This method returns false for most allocators, but may be used by
  // special-case allocators that track tensor usage. If this method returns
  // true, AllocateRaw() should be invoked for all values of `num_bytes`,
  // including 0.
  //
  // NOTE: It is the caller's responsibility to track whether an allocated
  // object is a buffer or an opaque handle. In particular, when this method
  // returns `true`, users of this allocator must not run any constructors or
  // destructors for complex objects, since there is no backing store for the
  // tensor in which to place their outputs.
  virtual bool AllocatesOpaqueHandle() const { return false; }

  // Returns the user-requested size of the data allocated at
  // 'ptr'.  Note that the actual buffer allocated might be larger
  // than requested, but this function returns the size requested by
  // the user.
  //
  // REQUIRES: TracksAllocationSizes() is true.
  //
  // REQUIRES: 'ptr!=nullptr' and points to a buffer previously
  // allocated by this allocator.
  virtual size_t RequestedSize(const void* ptr) const {
    CHECK(false) << "allocator doesn't track sizes";
    return size_t(0);
  }

  // Returns the allocated size of the buffer at 'ptr' if known,
  // otherwise returns RequestedSize(ptr). AllocatedSize(ptr) is
  // guaranteed to be >= RequestedSize(ptr).
  //
  // REQUIRES: TracksAllocationSizes() is true.
  //
  // REQUIRES: 'ptr!=nullptr' and points to a buffer previously
  // allocated by this allocator.
  virtual size_t AllocatedSize(const void* ptr) const {
    return RequestedSize(ptr);
  }

  // Returns either 0 or an identifier assigned to the buffer at 'ptr'
  // when the buffer was returned by AllocateRaw. If non-zero, the
  // identifier differs from every other ID assigned by this
  // allocator.
  //
  // REQUIRES: TracksAllocationSizes() is true.
  //
  // REQUIRES: 'ptr!=nullptr' and points to a buffer previously
  // allocated by this allocator.
  virtual int64 AllocationId(const void* ptr) const { return 0; }

  // Returns the allocated size of the buffer at 'ptr' if known,
  // otherwise returns 0. This method can be called when
  // TracksAllocationSizes() is false, but can be extremely slow.
  //
  // REQUIRES: 'ptr!=nullptr' and points to a buffer previously
  // allocated by this allocator.
  virtual size_t AllocatedSizeSlow(const void* ptr) const {
    if (TracksAllocationSizes()) {
      return AllocatedSize(ptr);
    }
    return 0;
  }

  // Fills in 'stats' with statistics collected by this allocator.
  virtual absl::optional<AllocatorStats> GetStats() { return absl::nullopt; }

  // If implemented, clears the internal stats except for the `in_use` fields
  // and sets the `peak_bytes_in_use` to be equal to the `bytes_in_use`. Returns
  //  true if implemented.
  //
  // REQUIRES: GetStats is overridden.
  virtual bool ClearStats() TF_MUST_USE_RESULT { return false; }

  virtual void SetSafeFrontier(uint64 count) {}

  // For allocator that are stream aware, allow to specify the compute
  // stream this allocator is used for.
  virtual void SetStream(void* stream) {}
};

// An implementation of Allocator that delegates all calls to another Allocator.
//
// Useful to clients who want to override part of the functionality of another
// allocator.
class AllocatorWrapper : public Allocator {
 public:
  explicit AllocatorWrapper(Allocator* wrapped) : wrapped_(wrapped) {}

  ~AllocatorWrapper() override {}

  // Returns the wrapped allocator to which all calls are delegated.
  Allocator* wrapped() const { return wrapped_; }

  std::string Name() override { return wrapped_->Name(); }

  void* AllocateRaw(size_t alignment, size_t num_bytes) override {
    return wrapped_->AllocateRaw(alignment, num_bytes);
  }

  void* AllocateRaw(size_t alignment, size_t num_bytes,
                    const AllocationAttributes& allocation_attr) override {
    return wrapped_->AllocateRaw(alignment, num_bytes, allocation_attr);
  }

  void DeallocateRaw(void* ptr) override { wrapped_->DeallocateRaw(ptr); }

  bool TracksAllocationSizes() const override {
    return wrapped_->TracksAllocationSizes();
  }

  bool AllocatesOpaqueHandle() const override {
    return wrapped_->AllocatesOpaqueHandle();
  }

  size_t RequestedSize(const void* ptr) const override {
    return wrapped_->RequestedSize(ptr);
  }

  size_t AllocatedSize(const void* ptr) const override {
    return wrapped_->AllocatedSize(ptr);
  }

  int64 AllocationId(const void* ptr) const override {
    return wrapped_->AllocationId(ptr);
  }

  size_t AllocatedSizeSlow(const void* ptr) const override {
    return wrapped_->AllocatedSizeSlow(ptr);
  }

 private:
  Allocator* const wrapped_;
};

// A tensorflow Op may need access to different kinds of memory that
// are not simply a function of the device to which the Op has been
// assigned.  For example, an Op executing on a GPU may still need
// to allocate CPU RAM for some purpose.  Internal to the tensorflow
// runtime we may choose to allocate CPU ram from special regions
// that have been prepared for higher performance in some use
// contexts, e.g. doing DMA with particular devices.  For these
// reasons, the Device interface does not expose just one memory
// Allocator, but instead provides an accessor that takes a
// specification of the desired memory attributes in order to select
// an Allocator.
//
// Example use:
//  // Allocator for ordinary device memory:
//  Allocator* a = allocator(AllocatorAttributes());
// ...
//  // Allocator for CPU RAM, regardless of where Op is executing:
//  AllocatorAttributes attr;
//  attr.set_on_host(true);
//  Allocator* a = allocator(attr);
struct AllocatorAttributes {
  void set_on_host(bool v) { value |= (static_cast<int>(v)); }
  bool on_host() const { return value & 0x1; }
  void set_nic_compatible(bool v) { value |= (static_cast<int>(v) << 1); }
  bool nic_compatible() const { return value & (0x1 << 1); }
  void set_gpu_compatible(bool v) { value |= (static_cast<int>(v) << 2); }
  bool gpu_compatible() const { return value & (0x1 << 2); }
  void Merge(AllocatorAttributes other) {
    value |= other.value;
    if (scope_id != other.scope_id) {
      CHECK(scope_id == 0 || other.scope_id == 0)
          << "At least one scope_id should be zero to merge "
             "AllocatorAttributes but found this.scope_id="
          << scope_id << " and other.scope_id=" << other.scope_id;
      scope_id = scope_id == 0 ? other.scope_id : scope_id;
    }
  }
  // Returns true if the fields set in *this is a subset of or equal to
  // those set in other.
  bool IsEqualOrLessRestrictiveThan(const AllocatorAttributes& other) const {
    return (value | other.value) == other.value;
  }

  // NOTE: The upper 8 bits of the value are reserved for
  // device-specific uses.  Implementors of a device can interpret these
  // upper 8 bits in device-specific ways, and ops implemented for those
  // devices are responsible for setting those 8 bits appropriately.
  uint32 value = 0;
  // EXPERIMENTAL: If this is greater than zero, then allocation is delegated to
  // a named special-purpose allocator on the same device.
  int32 scope_id = 0;

  // Returns a human readable representation of this.
  std::string DebugString() const;
};

// Returns a trivial implementation of Allocator, which is a process singleton.
// Access through this function is only intended for use by restricted parts
// of the infrastructure.
Allocator* cpu_allocator_base();

// If available, calls ProcessState::GetCPUAllocator(numa_node).
// If not, falls back to cpu_allocator_base().
// Intended for use in contexts where ProcessState is not visible at
// compile time. Where ProcessState is visible, it's preferable to
// call it directly.
Allocator* cpu_allocator(int numa_node = port::kNUMANoAffinity);

// Enables AllocatorStats in the default CPU allocator implementation.  By
// default, it's disabled.
void EnableCPUAllocatorStats();
// Disables AllocatorStats in the default CPU allocator implementation.  By
// default, it's disabled.
void DisableCPUAllocatorStats();
bool CPUAllocatorStatsEnabled();

// Enables full statistics collection in the default CPU allocator
// implementation.  By default, it's disabled.
void EnableCPUAllocatorFullStats();
bool CPUAllocatorFullStatsEnabled();

// An object that does the underlying suballoc/free of memory for a higher-level
// allocator.  The expectation is that the higher-level allocator is doing some
// kind of cache or pool management so that it will call SubAllocator::Alloc and
// Free relatively infrequently, compared to the number of times its own
// AllocateRaw and Free methods are called.
class SubAllocator {
 public:
  // Visitor gets called with a pointer to a memory area and its
  // size in bytes.  The index value will be numa_node for a CPU
  // allocator and GPU id for a GPU allocator.
  typedef std::function<void(void*, int index, size_t)> Visitor;

  SubAllocator(const std::vector<Visitor>& alloc_visitors,
               const std::vector<Visitor>& free_visitors);

  virtual ~SubAllocator() {}
  // Allocates at least num_bytes. Returns actual number of bytes allocated in
  // bytes_received. The caller can safely use the full bytes_received sized
  // buffer following the returend pointer.
  virtual void* Alloc(size_t alignment, size_t num_bytes,
                      size_t* bytes_received) = 0;
  virtual void Free(void* ptr, size_t num_bytes) = 0;

  // Returns true if the BFC allocator can safely coalesce adjacent regions
  // returned by this allocator.
  virtual bool SupportsCoalescing() const = 0;

 protected:
  // Implementation of Alloc() method must call this on newly allocated
  // value.
  void VisitAlloc(void* ptr, int index, size_t num_bytes);

  // Implementation of Free() method must call this on value to be
  // freed immediately before deallocation.
  void VisitFree(void* ptr, int index, size_t num_bytes);

  const std::vector<Visitor> alloc_visitors_;
  const std::vector<Visitor> free_visitors_;
};

}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_FRAMEWORK_ALLOCATOR_H_
