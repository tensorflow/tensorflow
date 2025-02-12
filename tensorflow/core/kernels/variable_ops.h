/* Copyright 2015 The TensorFlow Authors. All Rights Reserved.

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

#ifndef TENSORFLOW_CORE_KERNELS_VARIABLE_OPS_H_
#define TENSORFLOW_CORE_KERNELS_VARIABLE_OPS_H_

#include "tensorflow/core/framework/allocator.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/framework/resource_mgr.h"
#include "tensorflow/core/framework/resource_var.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/platform/macros.h"
#include "tensorflow/core/platform/mutex.h"
#include "tensorflow/core/platform/types.h"

namespace tensorflow {

class VariableOp : public OpKernel {
 public:
  explicit VariableOp(OpKernelConstruction* context);
  void Compute(OpKernelContext* ctx) override;

 private:
  DataType dtype_;
  TensorShape shape_;
  ContainerInfo cinfo_;

  VariableOp(const VariableOp&) = delete;
  void operator=(const VariableOp&) = delete;
};

}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_KERNELS_VARIABLE_OPS_H_
