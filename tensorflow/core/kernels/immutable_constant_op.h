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
#ifndef TENSORFLOW_CORE_KERNELS_IMMUTABLE_CONSTANT_OP_H_
#define TENSORFLOW_CORE_KERNELS_IMMUTABLE_CONSTANT_OP_H_

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
  static constexpr char const* kDTypeAttr = "dtype";
  static constexpr char const* kShapeAttr = "shape";
  static constexpr char const* kMemoryRegionNameAttr = "memory_region_name";

 private:
  string region_name_;
  DataType dtype_;
  TensorShape shape_;
  TF_DISALLOW_COPY_AND_ASSIGN(ImmutableConstantOp);
};

}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_KERNELS_IMMUTABLE_CONSTANT_OP_H_
