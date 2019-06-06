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

#include "tensorflow/core/kernels/stack.h"

#include <limits.h>

#include <atomic>
#include <map>
#include <queue>
#include <stack>
#include <vector>

#include "tensorflow/core/common_runtime/device.h"
#include "tensorflow/core/framework/device_base.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/framework/resource_mgr.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/core/refcount.h"
#include "tensorflow/core/lib/gtl/map_util.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/macros.h"
#include "tensorflow/core/platform/mutex.h"
#include "tensorflow/core/platform/thread_annotations.h"
#include "tensorflow/core/platform/types.h"

namespace tensorflow {

class Stack : public ResourceBase {
 public:
  static std::atomic<int64> stack_counter;

  struct TensorAndAllocation {
    Tensor tensor;
    AllocatorAttributes alloc_attrs;
    bool swapped_to_cpu;
  };

  Stack(const DataType& elem_type, const string& stack_name, int max_size)
      : elem_type_(elem_type),
        stack_name_(stack_name),
        max_size_(max_size),
        closed_(false) {}

  Status Push(const TensorAndAllocation& value) {
    mutex_lock l(mu_);
    TF_RETURN_IF_ERROR(CheckNotClosed());
    if (max_size_ >= 0 && stack_.size() >= max_size_) {
      return errors::InvalidArgument("Stack[", stack_name_, "] overflowed ",
                                     "its max_size (", max_size_, ")");
    }
    stack_.push_back(value);
    int index = stack_.size() - 1;
    unswapped_ids_.push(index);
    return Status::OK();
  }

  Status Pop(TensorAndAllocation* value) {
    mutex_lock l(mu_);
    TF_RETURN_IF_ERROR(CheckNotClosed());
    if (stack_.empty()) {
      return errors::InvalidArgument("Stack[", stack_name_,
                                     "] is empty when calling Pop().");
    }
    int index = stack_.size() - 1;
    while (is_swapping_ins_[index]) {
      cond_swapping_ins_[index].wait(l);
      index = stack_.size() - 1;
    }
    *value = stack_.back();
    stack_.pop_back();
    return Status::OK();
  }

  Status SwapOut(TensorAndAllocation* value, Tensor* cpu_tensor) {
    mutex_lock l(mu_);
    TF_RETURN_IF_ERROR(CheckNotClosed());

    value->tensor = *cpu_tensor;
    value->swapped_to_cpu = true;

    return Status::OK();
  }

  Status SwapIn(TensorAndAllocation* value, Tensor* device_tensor) {
    mutex_lock l(mu_);
    TF_RETURN_IF_ERROR(CheckNotClosed());

    value->tensor = *device_tensor;
    value->swapped_to_cpu = false;

    return Status::OK();
  }

  bool GetTensorToSwapOut(
      TensorAndAllocation** value,
      std::function<bool(const TensorAndAllocation&)> cond) {
    mutex_lock l(mu_);
    bool can_swap = false;
    while (!can_swap) {
      if (unswapped_ids_.empty()) {
        break;
      }
      int index = unswapped_ids_.front();
      unswapped_ids_.pop();
      if (!cond(stack_[index])) {
        continue;
      }
      swapped_ids_.push(index);
      *value = &stack_[index];
      can_swap = true;
    }
    return can_swap;
  }

  bool GetTensorToSwapIn(TensorAndAllocation** value, int& index) {
    mutex_lock l(mu_);
    bool can_unswap = false;
    while (!can_unswap) {
      if (swapped_ids_.empty()) {
        break;
      }
      index = swapped_ids_.top();
      swapped_ids_.pop();
      if (index >= stack_.size()) {
        continue;
      }
      *value = &stack_[index];
      is_swapping_ins_[index] = true;
      can_unswap = true;
    }
    return can_unswap;
  }

  void FinishSwappingIn(int index) {
    mutex_lock l(mu_);
    is_swapping_ins_[index] = false;
    cond_swapping_ins_[index].notify_one();
  }

  // We don't swap the first tensor on the stack and any subsequent tensors
  // that share the buffer with the first tensor.
  bool IsUsefulToSwap(const Tensor& tensor) const {
    if (stack_.empty()) {
      return false;
    }
    const Tensor& first = stack_.front().tensor;
    return !tensor.SharesBufferWith(first);
  }

  void Close() {
    mutex_lock l(mu_);
    stack_.clear();
    closed_ = true;
  }

  DataType ElemType() { return elem_type_; }

  string DebugString() const override {
    mutex_lock l(mu_);
    return strings::StrCat("Stack[", stack_name_, "]");
  }

  const string& stack_name() { return stack_name_; }

 private:
  friend class StackOp;
  mutex* mu() { return &mu_; }

  mutable mutex mu_;
  DataType elem_type_;
  const string stack_name_;
  Tensor handle_;
  int max_size_;
  bool closed_ GUARDED_BY(mu_);
  std::vector<TensorAndAllocation> stack_ GUARDED_BY(mu_);

  std::queue<int> unswapped_ids_ GUARDED_BY(mu_);
  std::stack<int> swapped_ids_ GUARDED_BY(mu_);

  std::map<int, bool> is_swapping_ins_ GUARDED_BY(mu_);
  std::map<int, condition_variable> cond_swapping_ins_;

  Status CheckNotClosed() const EXCLUSIVE_LOCKS_REQUIRED(mu_) {
    if (closed_) {
      return errors::InvalidArgument("Stack[", stack_name_,
                                     "] has already been closed.");
    }
    return Status::OK();
  }
};

Status GetStack(OpKernelContext* ctx, Stack** stack) {
  if (ctx->input_dtype(0) == DT_RESOURCE) {
    return LookupResource(ctx, HandleFromInput(ctx, 0), stack);
  } else {
    Tensor Tstack_handle = ctx->mutable_input(0, false);
    if (Tstack_handle.NumElements() != 2) {
      return errors::InvalidArgument(
          "Stack handle must have two elements, but had shape: ",
          Tstack_handle.shape().DebugString());
    }
    const string& container = Tstack_handle.flat<string>()(0);
    const string& stack_name = Tstack_handle.flat<string>()(1);
    string key = strings::StrCat(container, stack_name);
    ResourceMgr* rm = ctx->resource_manager();
    if (rm == nullptr) {
      return errors::Internal("No resource manager.");
    }
    auto* step_container = ctx->step_container();
    if (step_container == nullptr) {
      return errors::Internal("No step container.");
    }
    TF_RETURN_IF_ERROR(rm->Lookup(step_container->name(), key, stack));
    return Status::OK();
  }
}

std::atomic<int64> Stack::stack_counter{0};

// StackOp

StackOp::StackOp(OpKernelConstruction* context) : OpKernel(context) {
  OP_REQUIRES_OK(context, context->GetAttr("elem_type", &elem_type_));
  OP_REQUIRES_OK(context, context->GetAttr("stack_name", &stack_name_));
  if (stack_name_.empty()) stack_name_ = name();
}

void StackOp::Compute(OpKernelContext* ctx) {
  int32 size = std::numeric_limits<int32>::max();
  if (ctx->num_inputs() > 0) {
    const Tensor* tensor_size;
    OP_REQUIRES_OK(ctx, ctx->input("max_size", &tensor_size));

    OP_REQUIRES(
        ctx, TensorShapeUtils::IsScalar(tensor_size->shape()),
        errors::InvalidArgument("Stack size must be a scalar, but had shape: ",
                                tensor_size->shape().DebugString()));

    int32 size_value = tensor_size->scalar<int32>()();
    if (size_value >= 0) {
      size = size_value;
    }
  }

  static const char kContainer[] = "_stacks";
  auto stack_id = Stack::stack_counter.fetch_add(1);
  string stack_name = strings::StrCat(stack_name_, "_", stack_id);
  // Store the handle in a per-step container.
  ResourceMgr* rm = ctx->resource_manager();
  OP_REQUIRES(ctx, rm != nullptr, errors::Internal("No resource manager."));
  string key = strings::StrCat(kContainer, stack_name);
  Stack* stack = new Stack(elem_type_, stack_name, size);
  auto* step_container = ctx->step_container();
  OP_REQUIRES(ctx, step_container != nullptr,
              errors::Internal("No step container."));
  OP_REQUIRES_OK(ctx, rm->Create(step_container->name(), key, stack));
  if (IsRefType(ctx->expected_output_dtype(0))) {
    // Create the stack handle.
    AllocatorAttributes alloc_attr;
    alloc_attr.set_on_host(true);
    OP_REQUIRES_OK(ctx, ctx->allocate_temp(tensorflow::DT_STRING,
                                           tensorflow::TensorShape({2}),
                                           &stack->handle_, alloc_attr));
    auto handle = stack->handle_.flat<string>();
    handle(0) = kContainer;
    handle(1) = std::move(stack_name);
    ctx->set_output_ref(0, stack->mu(), &stack->handle_);
  } else {
    Tensor* handle;
    OP_REQUIRES_OK(ctx, ctx->allocate_output(0, TensorShape({}), &handle));
    handle->flat<ResourceHandle>()(0) =
        MakePerStepResourceHandle<Stack>(ctx, key);
  }
}

// StackPushOp

StackPushOp::StackPushOp(OpKernelConstruction* context, bool allow_swapping)
    : AsyncOpKernel(context) {
  if (allow_swapping) {
    OP_REQUIRES_OK(context, context->GetAttr("swap_memory", &swap_memory_));
  }
}

void StackPushOp::ComputeAsync(OpKernelContext* ctx, DoneCallback done) {
  // Get the stack from the handle.
  Stack* stack = nullptr;
  OP_REQUIRES_OK_ASYNC(ctx, GetStack(ctx, &stack), done);
  core::ScopedUnref unref(stack);

  if (ctx->input_dtype(1) != stack->ElemType()) {
    ctx->CtxFailure(errors::InvalidArgument("Must have type ",
                                            stack->ElemType(), " but got ",
                                            ctx->input_dtype(1)));
    done();
    return;
  }

  // Push the tensor onto the stack.
  // Swap the oldest tensor to CPU if instructed.
  const Tensor& tensor = ctx->input(1);
  AllocatorAttributes alloc_attrs = ctx->input_alloc_attr(1);

  OP_REQUIRES_OK_ASYNC(ctx, stack->Push({tensor, alloc_attrs, false}), done);
  ctx->set_output(0, tensor);

  // For now, we use a simple heuristic for swapping: A GPU tensor is moved
  // to CPU if the tensor has more than kCopyThreshold bytes and the GPU
  // allocator says more than kOccupancy of the memory is in use.
  static constexpr int kCopyThreshold = 2048;
  static constexpr double kOccupancy = 0.7;

  if (!swap_memory_ || alloc_attrs.on_host()) {
    done();
    return;
  }

  DeviceContext* device_ctxt = ctx->op_device_context();
  auto device = static_cast<tensorflow::Device*>(ctx->device());
  Allocator* allocator = device->GetAllocator(alloc_attrs);
  absl::optional<AllocatorStats> stats = allocator->GetStats();
  if (!stats || !*stats->bytes_limit ||
      stats->bytes_in_use <= (*stats->bytes_limit * kOccupancy)) {
    done();
    return;
  }

  // Obtain the oldest unswapped TensorAndAllocation.
  Stack::TensorAndAllocation* to_swap_out;
  if (!stack->GetTensorToSwapOut(
          &to_swap_out,
          [stack](const Stack::TensorAndAllocation& value) -> bool {
            return value.tensor.TotalBytes() > kCopyThreshold &&
                   stack->IsUsefulToSwap(value.tensor);
          })) {
    done();
    return;
  }

  // Asynchronously copy the tensor from GPU to CPU memory.
  // Swap the oldest tensor first.
  AllocatorAttributes host_alloc_attrs;
  host_alloc_attrs.set_gpu_compatible(true);
  host_alloc_attrs.set_on_host(true);
  Allocator* cpu_allocator = device->GetAllocator(host_alloc_attrs);
  Tensor* cpu_tensor = new Tensor(cpu_allocator, (to_swap_out->tensor).dtype(),
                                  (to_swap_out->tensor).shape());
  device_ctxt->CopyDeviceTensorToCPU(
      &(to_swap_out->tensor), "StackPush", device, cpu_tensor,
      [stack, to_swap_out, cpu_tensor, done](const Status& s) {
        if (s.ok()) {
          stack->SwapOut(to_swap_out, cpu_tensor);
        }
        done();
        delete cpu_tensor;
      });
}

bool StackPushOp::IsExpensive() { return false; }

// StackPopOp

StackPopOp::StackPopOp(OpKernelConstruction* context)
    : AsyncOpKernel(context) {}

void StackPopOp::ComputeAsync(OpKernelContext* ctx, DoneCallback done) {
  // Get the stack from the handle.
  Stack* stack = nullptr;
  OP_REQUIRES_OK_ASYNC(ctx, GetStack(ctx, &stack), done);
  core::ScopedUnref unref(stack);

  // Pop the tensor. Transfer the tensor back to device if it was
  // swapped out to CPU.
  Stack::TensorAndAllocation value;
  OP_REQUIRES_OK_ASYNC(ctx, stack->Pop(&value), done);
  if (value.swapped_to_cpu) {
    // Asynchronously copy the tensor back from CPU to GPU memory.
    DeviceContext* device_ctxt = ctx->op_device_context();
    Device* device = static_cast<Device*>(ctx->device());
    Tensor* cpu_tensor = &value.tensor;
    Allocator* gpu_allocator = device->GetAllocator(value.alloc_attrs);
    Tensor* device_tensor =
        new Tensor(gpu_allocator, cpu_tensor->dtype(), cpu_tensor->shape());
    device_ctxt->CopyCPUTensorToDevice(
        cpu_tensor, device, device_tensor,
        [device_tensor, ctx, done](const Status& s) {
          ctx->SetStatus(s);
          if (s.ok()) {
            ctx->set_output(0, *device_tensor);
          }
          done();
          delete device_tensor;
        });
  } else {
    // Execute synchronously if not swapped.
    ctx->set_output(0, value.tensor);
    done();
  }

  // This is the reverse of the heuristic for swapping out, asynchronously
  // swap in most recent swapped tensors if the GPU allocator says less than
  // kOccupancy of the memory is in use.
  static constexpr double kOccupancy = 0.7;

  DeviceContext* device_ctxt = ctx->op_device_context();
  Device* device = static_cast<Device*>(ctx->device());
  Allocator* gpu_allocator = device->GetAllocator(value.alloc_attrs);
  absl::optional<AllocatorStats> stats = gpu_allocator->GetStats();
  if (!stats || !*stats->bytes_limit ||
      stats->bytes_in_use > (*stats->bytes_limit * kOccupancy)) {
    return;
  }

  // Obtain the most recent swapped TensorAndAllocation.
  Stack::TensorAndAllocation* to_swap_in;
  int to_swap_in_index;
  if (!stack->GetTensorToSwapIn(&to_swap_in, to_swap_in_index)) {
    return;
  }

  Tensor* device_tensor =
      new Tensor(gpu_allocator, (to_swap_in->tensor).dtype(),
                 (to_swap_in->tensor).shape());
  device_ctxt->CopyCPUTensorToDevice(
      &(to_swap_in->tensor), device, device_tensor,
      [stack, to_swap_in, device_tensor, to_swap_in_index](const Status& s) {
        if (s.ok()) {
          stack->SwapIn(to_swap_in, device_tensor);
        }
        delete device_tensor;
        stack->FinishSwappingIn(to_swap_in_index);
      });
}

bool StackPopOp::IsExpensive() { return false; }

// StackCloseOp

StackCloseOp::StackCloseOp(OpKernelConstruction* context) : OpKernel(context) {}

void StackCloseOp::Compute(OpKernelContext* ctx) {
  Stack* stack = nullptr;
  OP_REQUIRES_OK(ctx, GetStack(ctx, &stack));
  core::ScopedUnref unref(stack);
  stack->Close();
}

bool StackCloseOp::IsExpensive() { return false; }

}  // namespace tensorflow
