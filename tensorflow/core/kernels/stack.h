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

#ifndef TENSORFLOW_CORE_KERNELS_STACK_H_
#define TENSORFLOW_CORE_KERNELS_STACK_H_

// See docs in ../ops/data_flow_ops.cc.

#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/platform/types.h"

namespace tensorflow {

// A per-run local stack. The stack uses a "per-step" resource manager which
// ensures that correct garbage collection on error or successful completion.
class StackOp : public OpKernel {
 public:
  explicit StackOp(OpKernelConstruction* context);
  void Compute(OpKernelContext* ctx) override;

 private:
  DataType elem_type_;
  string stack_name_;

  TF_DISALLOW_COPY_AND_ASSIGN(StackOp);
};

class StackPushOp : public AsyncOpKernel {
 public:
  StackPushOp(OpKernelConstruction* context, bool allow_swapping);
  void ComputeAsync(OpKernelContext* ctx, DoneCallback done) override;
  bool IsExpensive() override;

 private:
  bool swap_memory_ = false;
};

// Templated helper to make it easier to register kernels with or without
// swapping.
template <bool allow_swapping>
class TemplatedStackPushOp : public StackPushOp {
 public:
  TemplatedStackPushOp(OpKernelConstruction* context)
      : StackPushOp(context, allow_swapping) {}
};

class StackPopOp : public AsyncOpKernel {
 public:
  explicit StackPopOp(OpKernelConstruction* context);
  void ComputeAsync(OpKernelContext* ctx, DoneCallback done) override;
  bool IsExpensive() override;
};

class StackCloseOp : public OpKernel {
 public:
  explicit StackCloseOp(OpKernelConstruction* context);
  void Compute(OpKernelContext* ctx) override;
  bool IsExpensive() override;
};

}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_KERNELS_STACK_H_
