/* Copyright 2016 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/core/kernels/immutable_constant_op.h"

#include <unordered_set>

namespace tensorflow {

namespace {
class MemmappedTensorAllocator : public Allocator {
 public:
  MemmappedTensorAllocator() {}

  Status InitializeFromRegion(const string& name, Env* env) {
    const auto status =
        env->NewReadOnlyMemoryRegionFromFile(name, &memory_region_);
    if (!status.ok()) {
      return status;
    }
    return Status::OK();
  }
  string Name() override { return "MemmappedTensorAllocator"; }

  void* AllocateRaw(size_t alignment, size_t num_bytes) override {
    if ((reinterpret_cast<intptr_t>(memory_region_->data())) % alignment != 0) {
      allocation_status_ =
          errors::Internal("Readonly memory region has wrong alignment");
      return nullptr;
    }
    if (num_bytes > memory_region_->length()) {
      allocation_status_ = errors::Internal(
          "Readonly memory region has wrong length (", memory_region_->length(),
          ") when allocating ", num_bytes);
      return nullptr;
    }
    return const_cast<void*>(memory_region_->data());
  }

  void DeallocateRaw(void* ptr) override {
    if (ptr != memory_region_->data()) {
      LOG(ERROR)
          << "Deallocating not allocated region for readonly memory region";
    }
    if (delete_on_deallocate_) {
      delete this;
    }
  }
  const Status& allocation_status() const { return allocation_status_; }

  void set_delete_on_deallocate() { delete_on_deallocate_ = true; }

 private:
  std::unique_ptr<ReadOnlyMemoryRegion> memory_region_;
  // If there is an error during allocation we keep it in this status.
  Status allocation_status_;

  // When the allocator is owned by TensorBuffer it will be deleted on
  // de-allocation.
  bool delete_on_deallocate_ = false;

  TF_DISALLOW_COPY_AND_ASSIGN(MemmappedTensorAllocator);
};
}  // namespace

ImmutableConstantOp::ImmutableConstantOp(OpKernelConstruction* context)
    : OpKernel(context) {
  OP_REQUIRES_OK(context,
                 context->GetAttr(kMemoryRegionNameAttr, &region_name_));
  OP_REQUIRES_OK(context, context->GetAttr(kDTypeAttr, &dtype_));
  OP_REQUIRES_OK(context, context->GetAttr(kShapeAttr, &shape_));
}

void ImmutableConstantOp::Compute(OpKernelContext* ctx) {
  std::unique_ptr<MemmappedTensorAllocator> allocator(
      new MemmappedTensorAllocator());

  OP_REQUIRES_OK(ctx,
                 allocator->InitializeFromRegion(region_name_, ctx->env()));
  ctx->set_output(0, Tensor(allocator.get(), dtype_, shape_));
  OP_REQUIRES_OK(ctx, allocator->allocation_status());
  // Allocator is owned by the tensor from this point.
  allocator.release()->set_delete_on_deallocate();
}

ImmutableConstantOp::~ImmutableConstantOp() {}
constexpr char ImmutableConstantOp::kDTypeAttr[];
constexpr char ImmutableConstantOp::kShapeAttr[];
constexpr char ImmutableConstantOp::kMemoryRegionNameAttr[];

REGISTER_KERNEL_BUILDER(Name("ImmutableConst").Device(DEVICE_CPU),
                        ImmutableConstantOp);
}  // namespace tensorflow
