/* Copyright 2016 Google Inc. All Rights Reserved.

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

namespace tensorflow {

ImmutableConstantOp::ImmutableConstantOp(OpKernelConstruction* context)
    : OpKernel(context) {
  ::tensorflow::DataType dtype;
  OP_REQUIRES_OK(context, context->GetAttr(kDTypeAttr, &dtype));
  ::tensorflow::TensorShape shape;
  OP_REQUIRES_OK(context, context->GetAttr(kShapeAttr, &shape));
  string region_name;
  OP_REQUIRES_OK(context,
                 context->GetAttr(kMemoryRegionNameAttr, &region_name));
  OP_REQUIRES_OK(context,
                 allocator_.InitWithMemoryRegion(region_name, context->env()));
  tensor_ = ::tensorflow::Tensor(&allocator_, dtype, shape);
  OP_REQUIRES_OK(context, allocator_.allocation_status());
}

void ImmutableConstantOp::Compute(OpKernelContext* ctx) {
  ctx->set_output(0, tensor_);
}

ImmutableConstantOp::~ImmutableConstantOp() {}

ImmutableConstantOp::ReadOnlyMemoryRegionAllocator::
    ReadOnlyMemoryRegionAllocator() {}

Status ImmutableConstantOp::ReadOnlyMemoryRegionAllocator::InitWithMemoryRegion(
    const string& name, Env* env) {
  ReadOnlyMemoryRegion* region_ptr;
  const auto status = env->NewReadOnlyMemoryRegionFromFile(name, &region_ptr);
  if (!status.ok()) {
    return status;
  }
  memory_region_.reset(region_ptr);
  return Status::OK();
}

ImmutableConstantOp::ReadOnlyMemoryRegionAllocator::
    ~ReadOnlyMemoryRegionAllocator() {}

string ImmutableConstantOp::ReadOnlyMemoryRegionAllocator::Name() {
  return "ReadOnlyMemoryRegionAllocator";
}

void* ImmutableConstantOp::ReadOnlyMemoryRegionAllocator::AllocateRaw(
    size_t alignment, size_t num_bytes) {
  if (!memory_region_) {
    allocation_status_.Update(errors::Internal(
        "Allocate memory from not initialized readonly memory region"));
    return nullptr;
  }
  // Checks that the memory region satisfies alignment and size.
  if (reinterpret_cast<uint64>(memory_region_->data()) % alignment != 0) {
    allocation_status_.Update(
        errors::Internal("Readonly memory region has wrong alignment"));
    return nullptr;
  }
  if (num_bytes > memory_region_->length()) {
    allocation_status_.Update(errors::Internal(
        "Readonly memory region has wrong length (", memory_region_->length(),
        ") when allocating ", num_bytes));
    return nullptr;
  }
  // TODO(mgubin): This is a hack, the memory is not writable. A special
  // readonly tensor with allocator interface need to be added.
  return const_cast<void*>(memory_region_->data());
}

void ImmutableConstantOp::ReadOnlyMemoryRegionAllocator::DeallocateRaw(
    void* ptr) {
  if (ptr != memory_region_->data()) {
    LOG(ERROR)
        << "Deallocating not allocated region for readonly memory region";
  }
}

constexpr char ImmutableConstantOp::kDTypeAttr[];
constexpr char ImmutableConstantOp::kShapeAttr[];
constexpr char ImmutableConstantOp::kMemoryRegionNameAttr[];

REGISTER_KERNEL_BUILDER(Name("ImmutableConst").Device(DEVICE_CPU),
                        ImmutableConstantOp);
}  // namespace tensorflow
