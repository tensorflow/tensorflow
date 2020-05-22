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
#ifndef TENSORFLOW_CORE_KERNELS_RESOURCE_VARIABLE_OPS_H_
#define TENSORFLOW_CORE_KERNELS_RESOURCE_VARIABLE_OPS_H_

#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/resource_mgr.h"

namespace tensorflow {

class VarHandleOp : public OpKernel {
 public:
  explicit VarHandleOp(OpKernelConstruction* c);
  void Compute(OpKernelContext* ctx) override;
  const Tensor* const_tensor() const override {
    return name_ != ResourceHandle::ANONYMOUS_NAME ? &resource_ : nullptr;
  }

 private:
  // Same fields as in ResourceHandleOp.
  bool is_anonymous_;
  string container_;
  string name_;
  Tensor resource_;

  DtypeAndPartialTensorShape dtype_and_shape_;

  // A set of devices containing the resource variable. Set when the output
  // ResourceHandle represents a per-replica/partitioned resource variable.
  std::vector<string> allowed_devices_;
};

class ReadVariableOp : public OpKernel {
 public:
  explicit ReadVariableOp(OpKernelConstruction* c);
  void Compute(OpKernelContext* ctx) override;

 private:
  DataType dtype_;
};

class ReadVariablesOp : public OpKernel {
 public:
  explicit ReadVariablesOp(OpKernelConstruction* c);
  void Compute(OpKernelContext* ctx) override;
  bool IsExpensive() override { return false; }

 private:
  DataTypeVector dtypes_;
};

class DestroyResourceOp : public OpKernel {
 public:
  explicit DestroyResourceOp(OpKernelConstruction* ctx);
  void Compute(OpKernelContext* ctx) override;

 private:
  bool ignore_lookup_error_;
};

}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_KERNELS_RESOURCE_VARIABLE_OPS_H_
