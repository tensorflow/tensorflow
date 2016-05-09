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
#ifndef TENSORFLOW_KERNELS_IMMUTABLE_CONSTANT_OP_H_
#define TENSORFLOW_KERNELS_IMMUTABLE_CONSTANT_OP_H_

#include <memory>

#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/tensor_types.h"
#include "tensorflow/core/platform/env.h"
#include "tensorflow/core/platform/macros.h"

namespace tensorflow {

class ImmutableConstantOp : public OpKernel {
 public:
  explicit ImmutableConstantOp(OpKernelConstruction* context);
  void Compute(OpKernelContext* ctx) override;
  bool IsExpensive() override { return false; }
  ~ImmutableConstantOp() override;

  // Names of attributes that are used by this op
  static constexpr char kDTypeAttr[] = "dtype";
  static constexpr char kShapeAttr[] = "shape";
  static constexpr char kMemoryRegionNameAttr[] = "memory_region_name";

 private:
  class ReadOnlyMemoryRegionAllocator : public ::tensorflow::Allocator {
   public:
    ReadOnlyMemoryRegionAllocator();
    Status InitWithMemoryRegion(const string& name, Env* env);
    ~ReadOnlyMemoryRegionAllocator() override;
    string Name() override;
    void* AllocateRaw(size_t alignment, size_t num_bytes) override;
    void DeallocateRaw(void* ptr) override;
    const Status& allocation_status() const { return allocation_status_; }

   private:
    std::unique_ptr<ReadOnlyMemoryRegion> memory_region_;
    // If there is an error during allocation we keep it in this status.
    Status allocation_status_;
  };
  ReadOnlyMemoryRegionAllocator allocator_;
  Tensor tensor_;
  TF_DISALLOW_COPY_AND_ASSIGN(ImmutableConstantOp);
};

}  // namespace tensorflow

#endif  // TENSORFLOW_KERNELS_IMMUTABLE_CONSTANT_OP_H_
