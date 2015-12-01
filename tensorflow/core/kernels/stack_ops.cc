/* Copyright 2015 Google Inc. All Rights Reserved.

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

// See docs in ../ops/data_flow_ops.cc.

#include <limits.h>
#include <vector>

#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/framework/resource_mgr.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/gtl/map_util.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/port.h"
#include "tensorflow/core/platform/thread_annotations.h"
#include "tensorflow/core/public/tensor.h"
#include "tensorflow/core/public/tensor_shape.h"

namespace tensorflow {

class Stack : public ResourceBase {
 public:
  Stack(const DataType& elem_type, const Tensor& handle)
      : elem_type_(elem_type), handle_(handle) {}

  void Push(const PersistentTensor& value) {
    mutex_lock l(mu_);
    stack_.push_back(value);
  }

  bool Pop(PersistentTensor* value) {
    mutex_lock l(mu_);
    if (!stack_.empty()) {
      *value = stack_.back();
      stack_.pop_back();
      return true;
    }
    return false;
  }

  DataType ElemType() { return elem_type_; }

  string DebugString() override {
    mutex_lock l(mu_);
    return strings::StrCat("#elem:", stack_.size());
  }

 private:
  friend class StackOp;
  mutex* mu() { return &mu_; }
  Tensor* handle() { return &handle_; }

  mutex mu_;
  DataType elem_type_;
  Tensor handle_;
  std::vector<PersistentTensor> stack_ GUARDED_BY(mu_);
};

// A per-run local stack. The stack uses a "per-step" resource manager which
// ensures that correct garbage collection on error or successful completion.
class StackOp : public OpKernel {
 public:
  explicit StackOp(OpKernelConstruction* context) : OpKernel(context) {
    OP_REQUIRES_OK(context, context->GetAttr("elem_type", &elem_type_));
    OP_REQUIRES_OK(context, context->GetAttr("stack_name", &stack_name_));
    if (stack_name_ == "") stack_name_ = name();
  }

  void Compute(OpKernelContext* ctx) override {
    // Create the stack handle.
    Tensor stack_handle;
    AllocatorAttributes alloc_attr;
    alloc_attr.set_on_host(true);
    OP_REQUIRES_OK(ctx, ctx->allocate_temp(tensorflow::DT_STRING,
                                           tensorflow::TensorShape({2}),
                                           &stack_handle, alloc_attr));
    auto handle = stack_handle.flat<string>();
    handle(0) = "_stacks";
    handle(1) = stack_name_;
    // Store the handle in a container of the per-step RM.
    ResourceMgr* rm = ctx->step_resource_manager();
    OP_REQUIRES(ctx, rm != nullptr,
                errors::Internal("No per-step resource manager."));
    Stack* stack = new Stack(elem_type_, stack_handle);
    OP_REQUIRES_OK(ctx, rm->Create(handle(0), stack_name_, stack));
    ctx->set_output_ref(0, stack->mu(), stack->handle());
  }

 private:
  DataType elem_type_;
  string stack_name_;

  TF_DISALLOW_COPY_AND_ASSIGN(StackOp);
};

REGISTER_KERNEL_BUILDER(Name("Stack").Device(DEVICE_CPU), StackOp);
REGISTER_KERNEL_BUILDER(Name("Stack").Device(DEVICE_GPU).HostMemory("handle"),
                        StackOp);

class StackPushOp : public OpKernel {
 public:
  explicit StackPushOp(OpKernelConstruction* context) : OpKernel(context) {}

  void Compute(OpKernelContext* ctx) override {
    Tensor Tstack_handle = ctx->mutable_input(0, false);
    OP_REQUIRES(ctx, Tstack_handle.NumElements() == 2,
                errors::InvalidArgument(
                    "Stack handle must have two elements, but had shape: ",
                    Tstack_handle.shape().DebugString()));
    const string& container = Tstack_handle.flat<string>()(0);
    const string& stack_name = Tstack_handle.flat<string>()(1);
    ResourceMgr* rm = ctx->step_resource_manager();
    OP_REQUIRES(ctx, rm != nullptr,
                errors::Internal("No per-step resource manager."));
    Stack* stack = nullptr;
    OP_REQUIRES_OK(ctx, rm->Lookup(container, stack_name, &stack));
    OP_REQUIRES(ctx, ctx->input_dtype(1) == stack->ElemType(),
                errors::InvalidArgument("Must have type ", stack->ElemType(),
                                        " but got ", ctx->input_dtype(1)));
    stack->Push(PersistentTensor(ctx->input(1)));
    ctx->set_output(0, ctx->input(1));
  }
};

REGISTER_KERNEL_BUILDER(Name("StackPush").Device(DEVICE_CPU), StackPushOp);
REGISTER_KERNEL_BUILDER(
    Name("StackPush").Device(DEVICE_GPU).HostMemory("handle"), StackPushOp);

class StackPopOp : public OpKernel {
 public:
  explicit StackPopOp(OpKernelConstruction* context) : OpKernel(context) {}

  void Compute(OpKernelContext* ctx) override {
    Tensor Tstack_handle = ctx->mutable_input(0, false);
    OP_REQUIRES(ctx, Tstack_handle.NumElements() == 2,
                errors::InvalidArgument(
                    "Stack handle must have two elements, but had shape: ",
                    Tstack_handle.shape().DebugString()));
    const string& container = Tstack_handle.flat<string>()(0);
    const string& stack_name = Tstack_handle.flat<string>()(1);
    ResourceMgr* rm = ctx->step_resource_manager();
    OP_REQUIRES(ctx, rm != nullptr,
                errors::Internal("No per-step resource manager."));
    Stack* stack = nullptr;
    OP_REQUIRES_OK(ctx, rm->Lookup(container, stack_name, &stack));
    PersistentTensor value;
    bool has_value = stack->Pop(&value);
    if (!has_value) {
      errors::InvalidArgument("Calling Pop() when the stack is empty.");
    }
    ctx->set_output(0, *value.AccessTensor(ctx));
  }
};

REGISTER_KERNEL_BUILDER(Name("StackPop").Device(DEVICE_CPU), StackPopOp);
REGISTER_KERNEL_BUILDER(
    Name("StackPop").Device(DEVICE_GPU).HostMemory("handle"), StackPopOp);

class StackCloseOp : public OpKernel {
 public:
  explicit StackCloseOp(OpKernelConstruction* context) : OpKernel(context) {}

  void Compute(OpKernelContext* ctx) override {
    Tensor Tstack_handle = ctx->mutable_input(0, false);
    OP_REQUIRES(ctx, Tstack_handle.NumElements() == 2,
                errors::InvalidArgument(
                    "Stack handle must have two elements, but had shape: ",
                    Tstack_handle.shape().DebugString()));
    const string& container = Tstack_handle.flat<string>()(0);
    const string& stack_name = Tstack_handle.flat<string>()(1);
    ResourceMgr* rm = ctx->step_resource_manager();
    OP_REQUIRES(ctx, rm != nullptr,
                errors::Internal("No per-step resource manager."));
    OP_REQUIRES_OK(ctx, rm->Delete<Stack>(container, stack_name));
  }
};

REGISTER_KERNEL_BUILDER(Name("StackClose").Device(DEVICE_CPU), StackCloseOp);
REGISTER_KERNEL_BUILDER(
    Name("StackClose").Device(DEVICE_GPU).HostMemory("handle"), StackCloseOp);

}  // namespace tensorflow
