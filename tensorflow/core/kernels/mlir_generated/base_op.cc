/* Copyright 2020 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/core/kernels/mlir_generated/base_op.h"

#include "tensorflow/core/framework/allocation_description.pb.h"
#include "tensorflow/core/framework/tensor.h"

namespace tensorflow {
namespace {

// A simple TensorBuffer implementation that allows us to create Tensors that
// take ownership of pre-allocated memory.
class MlirTensorBuffer : public TensorBuffer {
 public:
  MlirTensorBuffer(const void* ptr, size_t size, Allocator* allocator)
      : TensorBuffer(const_cast<void*>(ptr)),
        size_(size),
        allocator_(allocator) {}

  ~MlirTensorBuffer() override {
    if (data()) {
      allocator_->DeallocateRaw(data());
    }
  }

  size_t size() const override { return size_; }

  TensorBuffer* root_buffer() override { return this; }

  void FillAllocationDescription(AllocationDescription* proto) const override {
    proto->set_requested_bytes(static_cast<int64>(size_));
    proto->set_allocator_name(allocator_->Name());
    proto->set_ptr(reinterpret_cast<uintptr_t>(data()));
    if (allocator_->TracksAllocationSizes()) {
      auto ab = static_cast<int64>(allocator_->AllocatedSize(data()));
      proto->set_allocated_bytes(ab);
      int64 id = allocator_->AllocationId(data());
      if (id > 0) {
        proto->set_allocation_id(id);
      }
      if (RefCountIsOne()) {
        proto->set_has_single_reference(true);
      }
    }
  }

 private:
  size_t size_;
  Allocator* allocator_;
};

}  // namespace

TensorBuffer* GetMlirTensorBuffer(const void* ptr, size_t size,
                                  Allocator* allocator) {
  return new MlirTensorBuffer(ptr, size, allocator);
}

}  // namespace tensorflow
