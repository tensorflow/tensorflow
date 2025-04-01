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
#ifndef TENSORFLOW_CORE_COMMON_RUNTIME_SCOPED_ALLOCATOR_MGR_H_
#define TENSORFLOW_CORE_COMMON_RUNTIME_SCOPED_ALLOCATOR_MGR_H_

#include <string>
#include <unordered_map>

#include "tensorflow/core/common_runtime/scoped_allocator.h"
#include "tensorflow/core/lib/core/refcount.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/platform/mutex.h"

namespace tensorflow {
class ScopedAllocatorMgr;

// At most one of these exists per <device, step_id> pair.
// A Ref is held by every ScopedAllocator and also by the ScopedAllocatorMgr.
class ScopedAllocatorContainer : public core::RefCounted {
 public:
  // Establishes a reachable ScopedAllocator.
  absl::Status AddScopedAllocator(
      const Tensor& backing_tensor, int32_t scope_id,
      const std::string& scope_name,
      const absl::Span<const ScopedAllocator::Field>& fields,
      int32_t expected_call_count);

  ScopedAllocatorInstance* GetInstance(int32_t scope_id);
  ScopedAllocator* GetAllocator(int32_t scope_id);

  // Retire the scope_id.
  void Drop(int32_t scope_id, ScopedAllocator* sa);

 protected:
  friend class ScopedAllocatorMgr;
  ScopedAllocatorContainer(const ScopedAllocatorMgr* mgr, int64_t step_id)
      : mgr_(mgr), step_id_(step_id) {}
  ~ScopedAllocatorContainer();

 private:
  const ScopedAllocatorMgr* mgr_;
  int64_t step_id_;
  mutex mu_;
  struct SAField {
    int32 field_index;
    union {
      ScopedAllocator* scoped_allocator;
      ScopedAllocatorInstance* instance;
    };
    SAField(int32_t fi, ScopedAllocatorInstance* sai)
        : field_index(fi), instance(sai) {}
    SAField(int32_t fi, ScopedAllocator* sa)
        : field_index(fi), scoped_allocator(sa) {}
    SAField()
        : field_index(ScopedAllocator::kBackingIndex),
          scoped_allocator(nullptr) {}
  };
  std::unordered_map<int32, SAField> allocators_ TF_GUARDED_BY(mu_);
};

// At most one of these exists per device.
class ScopedAllocatorMgr {
 public:
  explicit ScopedAllocatorMgr(const std::string& device_name)
      : device_name_(device_name) {}
  ~ScopedAllocatorMgr();

  ScopedAllocatorContainer* GetContainer(int64_t step_id);

  // Establishes a reachable ScopedAllocator.
  absl::Status AddScopedAllocator(
      const Tensor& backing_tensor, int64_t step_id, int32_t scope_id,
      const std::string& scope_name,
      const absl::Span<const ScopedAllocator::Field>& fields,
      int32_t expected_call_count);

  void Cleanup(int64_t step_id);

  // Populate the bytes and offset members of Field.  Instance allocaters get
  // consecutive scope_id values following that of the base ScopedAllocator.
  // Returns the total number of bytes required to be allocated in the
  // backing tensor, for convenience.  (The same value can be obtained
  // by summing offset and bytes in the last field.)
  static size_t PopulateFields(int32_t scope_id,
                               const absl::Span<const TensorShape>& shapes,
                               const DataType dtype,
                               std::vector<ScopedAllocator::Field>* fields);

  const std::string& device_name() const { return device_name_; }

 private:
  std::string device_name_;
  mutex mu_;
  std::unordered_map<int64_t, ScopedAllocatorContainer*> per_step_map_
      TF_GUARDED_BY(mu_);
};

}  // namespace tensorflow
#endif  // TENSORFLOW_CORE_COMMON_RUNTIME_SCOPED_ALLOCATOR_MGR_H_
