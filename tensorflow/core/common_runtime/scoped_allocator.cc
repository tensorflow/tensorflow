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
#include "tensorflow/core/common_runtime/scoped_allocator.h"

#include "tensorflow/core/common_runtime/scoped_allocator_mgr.h"
#include "tensorflow/core/platform/dynamic_annotations.h"

namespace tensorflow {

ScopedAllocator::ScopedAllocator(const Tensor& backing_tensor, int32 scope_id,
                                 const string& name,
                                 const gtl::ArraySlice<Field> fields,
                                 int32 expected_call_count,
                                 ScopedAllocatorContainer* container)
    : backing_tensor_(backing_tensor),
      tbuf_(backing_tensor_.buf_),
      id_(scope_id),
      name_(name),
      container_(container),
      fields_(fields.begin(), fields.end()),
      expected_call_count_(expected_call_count),
      live_alloc_count_(0) {
  // Hold this until all aliases have been deallocated.
  tbuf_->Ref();
  // Hold this until all expected_calls have been made.
  container->Ref();
  CHECK_GE(tbuf_->size(), fields.back().offset + fields.back().bytes_requested);
}

ScopedAllocator::~ScopedAllocator() {
  mutex_lock l(mu_);
  VLOG(1) << "~ScopedAllocator " << this << " tbuf_ " << tbuf_ << " data "
          << static_cast<void*>(tbuf_->data());
  // In the absence of incomplete graph execution situations
  // (interruption by error status or control flow branch crossing
  // ScopedAllocation region) we expect expected_call_count_ == 0 at
  // exit.
  if (VLOG_IS_ON(1)) {
    if (expected_call_count_ > 0)
      VLOG(1) << "expected_call_count_ = " << expected_call_count_
              << " at deallocation";
  }
  if (tbuf_) tbuf_->Unref();
}

void* ScopedAllocator::AllocateRaw(int32 field_index, size_t num_bytes) {
  VLOG(1) << "ScopedAllocator index " << id_ << " AllocateRaw "
          << "field " << field_index << " num_bytes " << num_bytes;
  void* ptr = nullptr;
  const Field* field = nullptr;
  {
    mutex_lock l(mu_);
    if (expected_call_count_ <= 0) {
      LOG(ERROR) << "Scoped allocator " << name_
                 << " could not satisfy request for " << num_bytes
                 << " bytes, expected uses exhausted. ";
      return nullptr;
    }

    int32_t num_fields = static_cast<int32>(fields_.size());
    if (field_index >= num_fields) {
      LOG(ERROR) << "ScopedAllocator " << name_
                 << " received unexpected field number " << field_index;
      return nullptr;
    }

    field = &fields_[field_index];
    if (num_bytes != field->bytes_requested) {
      LOG(ERROR) << "ScopedAllocator " << name_ << " got request for "
                 << num_bytes << " bytes from field " << field_index
                 << " which has precalculated size " << field->bytes_requested
                 << " and offset " << field->offset;
      return nullptr;
    }

    ptr = static_cast<void*>((tbuf_->template base<char>() + field->offset));

    ++live_alloc_count_;
    --expected_call_count_;
    if (0 == expected_call_count_) {
      for (auto& f : fields_) {
        container_->Drop(f.scope_id, this);
      }
      container_->Drop(id_, this);
      container_->Unref();
      container_ = nullptr;
    }
  }
  VLOG(2) << "AllocateRaw returning " << ptr << " bytes_requested "
          << field->bytes_requested << " bytes_allocated "
          << field->bytes_allocated;

  // If there is overshoot due to alignment, let MSAN believe that the padding
  // is initialized.  This is okay because we do not use this memory region for
  // anything meaningful.
  if (field->bytes_allocated > field->bytes_requested) {
    size_t extra_bytes = field->bytes_allocated - field->bytes_requested;
    void* extra_buf = static_cast<void*>(static_cast<char*>(ptr) +
                                         field->bytes_allocated - extra_bytes);
    VLOG(2) << "AllocateRaw requested " << num_bytes
            << " bytes which is not divisible by kAllocatorAlignment="
            << Allocator::kAllocatorAlignment << " and hence we allocated "
            << field->bytes_allocated << ". Annotating " << extra_bytes
            << " bytes starting at " << extra_buf
            << " with TF_ANNOTATE_MEMORY_IS_INITIALIZED";
    TF_ANNOTATE_MEMORY_IS_INITIALIZED(extra_buf, extra_bytes);
  }

  return ptr;
}

void ScopedAllocator::DeallocateRaw(void* p) {
  CHECK(VerifyPointer(p));

  bool dead = false;
  {
    mutex_lock l(mu_);
    CHECK_GT(live_alloc_count_, 0);
    if (0 == --live_alloc_count_) {
      if (0 == expected_call_count_) {
        dead = true;
      }
    }
  }
  if (dead) {
    delete this;
  }
}

bool ScopedAllocator::VerifyPointer(const void* p) {
  void* base = tbuf_->data();
  CHECK_GE(p, base);
  for (auto& f : fields_) {
    void* f_ptr = static_cast<void*>(static_cast<char*>(base) + f.offset);
    if (f_ptr == p) {
      return true;
      break;
    }
  }
  VLOG(1) << "ScopedAllocator index " << id_ << " VerifyPointer for p=" << p
          << " failed.";
  return false;
}

bool ScopedAllocator::VerifyTensor(const Tensor* t) {
  return VerifyPointer(t->buf_->data());
}

ScopedAllocatorInstance::ScopedAllocatorInstance(ScopedAllocator* sa,
                                                 int32 field_index)
    : scoped_allocator_(sa),
      field_index_(field_index),
      allocated_(false),
      deallocated_(false),
      in_table_(true) {
  VLOG(1) << "new ScopedAllocatorInstance " << this << " on SA " << sa
          << " field_index " << field_index;
}

void ScopedAllocatorInstance::DropFromTable() {
  bool del = false;
  {
    mutex_lock l(mu_);
    CHECK(in_table_);
    in_table_ = false;
    VLOG(2) << "ScopedAllocatorInstance::DropFromTable " << this
            << " allocated_ " << allocated_ << " deallocated_ " << deallocated_
            << " in_table_ " << in_table_;
    // Single use is complete when it is allocated and deallocated.
    // This check prevents a race between Allocating the tensor slice and
    // Dropping it from the parent container's table.
    if (allocated_ && deallocated_) {
      del = true;
    }
  }
  if (del) delete this;
}

void* ScopedAllocatorInstance::AllocateRaw(size_t alignment, size_t num_bytes) {
  void* ptr = scoped_allocator_->AllocateRaw(field_index_, num_bytes);
  {
    mutex_lock l(mu_);
    if (nullptr == ptr) {
      VLOG(2) << "ScopedAllocatorInstance::AllocateRaw " << this
              << " call to underlying ScopedAllocator unsuccessful,"
              << " allocated_ " << allocated_ << " deallocated_ "
              << deallocated_ << " in_table_ " << in_table_
              << " returning nullptr.";
    } else {
      allocated_ = true;
      VLOG(2) << "ScopedAllocatorInstance::AllocateRaw " << this
              << " allocated_ " << allocated_ << " deallocated_ "
              << deallocated_ << " in_table_ " << in_table_
              << " returning ptr = " << ptr;
    }
  }
  return ptr;
}

void ScopedAllocatorInstance::DeallocateRaw(void* p) {
  scoped_allocator_->DeallocateRaw(p);
  bool del = false;
  {
    mutex_lock l(mu_);
    CHECK(allocated_);
    deallocated_ = true;
    VLOG(2) << "ScopedAllocatorInstance::DeallocateRaw " << this
            << " allocated_ " << allocated_ << " deallocated_ " << deallocated_
            << " in_table_ " << in_table_;
    // Single use is now complete, but only delete this instance when it is
    // no longer in a ScopedAllocatorContainer's table.
    if (!in_table_) {
      del = true;
    }
  }
  if (del) delete this;
}

string ScopedAllocatorInstance::Name() {
  return strings::StrCat(scoped_allocator_->name(), "_field_", field_index_);
}

}  // namespace tensorflow
