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
#include "tensorflow/core/common_runtime/scoped_allocator_mgr.h"

#include "tensorflow/core/common_runtime/scoped_allocator.h"
#include "tensorflow/core/framework/allocator.h"

namespace tensorflow {

Status ScopedAllocatorContainer::AddScopedAllocator(
    const Tensor& backing_tensor, int32 scope_id, const string& scope_name,
    const gtl::ArraySlice<ScopedAllocator::Field>& fields,
    int32 expected_call_count) {
  VLOG(1) << "AddScopedAllocator " << mgr_->device_name()
          << " step_id_=" << step_id_ << " scope_id=" << scope_id;
  mutex_lock l(mu_);
  // Ensure none of the new scope_ids are in use.
  auto it = allocators_.find(scope_id);
  if (it != allocators_.end()) {
    return errors::Internal("Cannot create ScopedAllocator because scope_id ",
                            scope_id, " for name ", scope_name,
                            " already exists");
  }
  for (auto& f : fields) {
    if (allocators_.find(f.scope_id) != allocators_.end()) {
      return errors::Internal(
          "Cannot create ScopedAllocator because field scope_id ", f.scope_id,
          " for name ", scope_name, " already exists");
    }
  }
  VLOG(2) << " container " << this << " step_id " << step_id_;
  ScopedAllocator* sa = new ScopedAllocator(
      backing_tensor, scope_id, scope_name, fields, expected_call_count, this);
  allocators_[scope_id] =
      ScopedAllocatorContainer::SAField(ScopedAllocator::kBackingIndex, sa);
  VLOG(2) << "#fields " << fields.size();
  for (int i = 0; i < fields.size(); ++i) {
    const ScopedAllocator::Field& f = fields[i];
    VLOG(2) << "Adding instance with for " << mgr_->device_name()
            << " scope_id=" << f.scope_id;
    allocators_[f.scope_id] = ScopedAllocatorContainer::SAField(
        i, new ScopedAllocatorInstance(sa, i));
  }
  return Status::OK();
}

ScopedAllocator* ScopedAllocatorContainer::GetAllocator(int32 scope_id) {
  mutex_lock l(mu_);
  auto it = allocators_.find(scope_id);
  if (it != allocators_.end()) {
    CHECK_EQ(ScopedAllocator::kBackingIndex, it->second.field_index);
    return it->second.scoped_allocator;
  } else {
    LOG(ERROR) << "Failed to find ScopedAllocator for " << scope_id
               << " in container for step " << step_id_ << " on "
               << mgr_->device_name();
    return nullptr;
  }
}

ScopedAllocatorInstance* ScopedAllocatorContainer::GetInstance(int32 scope_id) {
  VLOG(2) << "GetInstance " << scope_id << " step " << step_id_ << " on "
          << mgr_->device_name();
  mutex_lock l(mu_);
  auto it = allocators_.find(scope_id);
  if (it != allocators_.end()) {
    return it->second.instance;
  }
  LOG(FATAL) << "Failed to find instance " << scope_id << " in container "
             << step_id_ << " on " << mgr_->device_name();
  return nullptr;
}

void ScopedAllocatorContainer::Drop(int32 scope_id, ScopedAllocator* sa) {
  VLOG(2) << "Drop " << scope_id << " from container " << this << " step "
          << step_id_ << " on " << mgr_->device_name();
  mutex_lock l(mu_);
  auto it = allocators_.find(scope_id);
  if (it != allocators_.end()) {
    if (it->second.field_index != ScopedAllocator::kBackingIndex) {
      it->second.instance->DropFromTable();
    }
    allocators_.erase(it);
  }
}

ScopedAllocatorContainer::~ScopedAllocatorContainer() {
  VLOG(2) << "~ScopedAllocatorContainer " << this << " step " << step_id_
          << " on " << mgr_->device_name();
  mutex_lock l(mu_);
  // In normal execution the table should be empty and all of its contents
  // deleted via Drop.  When a step ends early (e.g. through abnormal
  // termination) we need to clean up explicitly.  So long as graph execution
  // of the associated step has completely terminated this should be safe.
  for (auto& it : allocators_) {
    if (it.second.field_index == ScopedAllocator::kBackingIndex) {
      delete it.second.scoped_allocator;
    } else {
      it.second.instance->DropFromTable();
    }
  }
}

ScopedAllocatorMgr::~ScopedAllocatorMgr() {
  mutex_lock l(mu_);
  for (auto it : per_step_map_) {
    // In normal execution the associated ScopedAllocatorContainer is
    // empty and gone by the end of the step.  But in abnormal termination,
    // such as when an error has interrupted execution or in a unittest,
    // we need to remove all of its Refs here to avoid memory leaks.
    // This is safe so long as graph execution has ceased.
    while (!it.second->Unref()) {
    }
  }
}

void ScopedAllocatorMgr::Cleanup(int64 step_id) {
  mutex_lock l(mu_);
  auto it = per_step_map_.find(step_id);
  if (it != per_step_map_.end()) {
    it->second->Unref();
    per_step_map_.erase(it);
  }
}

ScopedAllocatorContainer* ScopedAllocatorMgr::GetContainer(int64 step_id) {
  VLOG(2) << "GetContainer " << step_id << " on " << device_name();
  ScopedAllocatorContainer* sac = nullptr;
  mutex_lock l(mu_);
  auto it = per_step_map_.find(step_id);
  if (it == per_step_map_.end()) {
    sac = new ScopedAllocatorContainer(this, step_id);
    per_step_map_[step_id] = sac;
  } else {
    sac = it->second;
  }
  return sac;
}

Status ScopedAllocatorMgr::AddScopedAllocator(
    const Tensor& backing_tensor, int64 step_id, int32 scope_id,
    const string& scope_name,
    const gtl::ArraySlice<ScopedAllocator::Field>& fields,
    int32 expected_call_count) {
  ScopedAllocatorContainer* sac = GetContainer(step_id);
  return sac->AddScopedAllocator(backing_tensor, scope_id, scope_name, fields,
                                 expected_call_count);
}

/*static*/
size_t ScopedAllocatorMgr::PopulateFields(
    int32 scope_id, const gtl::ArraySlice<TensorShape>& shapes,
    const DataType dtype, std::vector<ScopedAllocator::Field>* fields) {
  const int32 num_fields = static_cast<int32>(shapes.size());
  fields->resize(num_fields);
  // At the end of iteration `i`, `offset` points to the offset from the start
  // of the backing buffer until the end of `field[i].bytes_allocated`.  This
  // is aligned to `kAllocatorAlignment`.
  size_t offset = 0;
  for (int32 i = 0; i < num_fields; ++i) {
    size_t bytes_requested = shapes[i].num_elements() * DataTypeSize(dtype);
    auto* field = &((*fields)[i]);
    field->scope_id = scope_id + 1 + i;
    field->bytes_requested = bytes_requested;
    field->offset = offset;
    offset += bytes_requested;

    // Compute actual #bytes allocated, which may include padding due to
    // alignment.
    size_t bytes_allocated = bytes_requested;
    size_t overshoot = offset % Allocator::kAllocatorAlignment;
    if (overshoot > 0) {
      size_t alignment_bytes = Allocator::kAllocatorAlignment - overshoot;
      bytes_allocated += alignment_bytes;
      offset += alignment_bytes;
    }
    field->bytes_allocated = bytes_allocated;

    VLOG(1) << "field=" << i << " scope_id=" << field->scope_id
            << " bytes_requested=" << field->bytes_requested
            << " offset=" << field->offset
            << " bytes_allocated=" << field->bytes_allocated;
  }

  return offset;
}

}  // namespace tensorflow
