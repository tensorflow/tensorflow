/* Copyright 2018 The TensorFlow Authors. All Rights Reserved.

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
#ifndef TENSORFLOW_CORE_COMMON_RUNTIME_SCOPED_ALLOCATOR_H_
#define TENSORFLOW_CORE_COMMON_RUNTIME_SCOPED_ALLOCATOR_H_

#include "tensorflow/core/framework/allocator.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/lib/core/refcount.h"
#include "tensorflow/core/platform/mutex.h"

namespace tensorflow {
class ScopedAllocatorContainer;
class ScopedAllocatorInstance;

// Manages a single backing tensor and a collection of aliases.
class ScopedAllocator {
 public:
  static constexpr int32_t kInvalidId = 0;
  static constexpr size_t kMaxAlignment = 64;

  // A subrange of the TensorBuffer associated with this object that
  // will be the backing memory for one aliased tensor.
  struct Field {
    int32 scope_id;
    size_t offset;
    size_t bytes_requested;
    size_t bytes_allocated;
  };
  // Field index that refers to backing tensor, not any aliased field.
  static constexpr int32_t kBackingIndex = -1;

  // backing_tensor is expected to be newly allocated by a ScopedAllocatorOp
  // instance.  It must be large enough to back all of the specified
  // (offset, byte) ranges of the fields.
  ScopedAllocator(const Tensor& backing_tensor, int32_t scope_id,
                  const std::string& name, const absl::Span<const Field> fields,
                  int32_t expected_call_count,
                  ScopedAllocatorContainer* container);

  // Automatically deletes when last use expires, or when
  // ScopedAllocatorContainer decides to delete.
  ~ScopedAllocator() TF_LOCKS_EXCLUDED(mu_);

  // For debugging: returns true iff p is a pointer that could have
  // been returned by AllocateRaw.
  bool VerifyPointer(const void* p);
  bool VerifyTensor(const Tensor* t);

  const Tensor& tensor() const { return backing_tensor_; }

  const std::string& name() const { return name_; }

 private:
  friend class ScopedAllocatorInstance;
  // Only ScopedAllocatorInstances can call AllocateRaw and DeallocateRaw on a
  // ScopedAllocator
  void* AllocateRaw(int32_t field_index, size_t num_bytes)
      TF_LOCKS_EXCLUDED(mu_);
  void DeallocateRaw(void* p) TF_LOCKS_EXCLUDED(mu_);
  Tensor backing_tensor_;
  TensorBuffer* tbuf_;
  int32 id_;
  std::string name_;
  ScopedAllocatorContainer* container_;
  std::vector<Field> fields_;
  mutex mu_;
  int32 expected_call_count_ TF_GUARDED_BY(mu_);
  int32 live_alloc_count_ TF_GUARDED_BY(mu_);
};

// An Allocator that will return a pointer into the backing buffer of
// a previously allocated tensor, allowing creation of an alias
// tensor.  There is a one-to-one mapping between the fields of a
// ScopedAllocator and ScopedAllocatorInstances.  There is also a one-to-one
// mapping between scope_ids and ScopedAllocatorInstances.  It should be
// discarded immediately after a single use.
class ScopedAllocatorInstance : public Allocator {
 public:
  explicit ScopedAllocatorInstance(ScopedAllocator* sa, int32_t field_index);

 private:
  ~ScopedAllocatorInstance() override {
    VLOG(1) << "~ScopedAllocatorInstance " << this;
  }

 public:
  // When a ScopedAllocatorContainer "Drops" a scope_id, it calls DropFromTable
  // on the underlying ScopedAllocatorInstance.  If this instance has already
  // deallocated the tensor slice, we can safely delete this.
  void DropFromTable() TF_LOCKS_EXCLUDED(mu_);
  void* AllocateRaw(size_t alignment, size_t num_bytes)
      TF_LOCKS_EXCLUDED(mu_) override;
  void* AllocateRaw(size_t alignment, size_t num_bytes,
                    const AllocationAttributes& allocator_attr) override {
    return AllocateRaw(alignment, num_bytes);
  }
  void DeallocateRaw(void* p) TF_LOCKS_EXCLUDED(mu_) override;
  bool TracksAllocationSizes() const override { return false; }
  size_t RequestedSize(const void* ptr) const override { return 0; }
  size_t AllocatedSize(const void* ptr) const override { return 0; }
  int64_t AllocationId(const void* ptr) const override { return 0; }
  size_t AllocatedSizeSlow(const void* ptr) const override { return 0; }
  std::string Name() override;

 private:
  mutex mu_;
  ScopedAllocator* scoped_allocator_;
  int32 field_index_;
  bool allocated_ TF_GUARDED_BY(mu_);
  bool deallocated_ TF_GUARDED_BY(mu_);
  bool in_table_ TF_GUARDED_BY(mu_);
};

}  // namespace tensorflow
#endif  // TENSORFLOW_CORE_COMMON_RUNTIME_SCOPED_ALLOCATOR_H_
