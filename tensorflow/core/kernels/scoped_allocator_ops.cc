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

#include "tensorflow/core/common_runtime/dma_helper.h"
#include "tensorflow/core/common_runtime/scoped_allocator.h"
#include "tensorflow/core/common_runtime/scoped_allocator_mgr.h"
#include "tensorflow/core/framework/allocator.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/core/status.h"

namespace tensorflow {

class ScopedAllocatorOp : public OpKernel {
 public:
  explicit ScopedAllocatorOp(OpKernelConstruction* context)
      : OpKernel(context) {
    OP_REQUIRES_OK(context, context->GetAttr("T", &dtype_));
    OP_REQUIRES_OK(context, context->GetAttr("shapes", &shapes_));
    OP_REQUIRES_OK(context, context->GetAttr("sa_name", &name_));
    OP_REQUIRES_OK(context, context->GetAttr("id", &id_));
    OP_REQUIRES_OK(context, context->GetAttr("expected_call_count",
                                             &expected_call_count_));
    device_ = context->device();
    // Precalculate the size of the backing tensor and the offsets of
    // the subtensors to be allocated from it, taking into account
    // alignment considerations.
    ScopedAllocatorMgr::PopulateFields(id_, shapes_, dtype_, &fields_);
    size_t num_bytes = fields_.back().offset + fields_.back().bytes_allocated;
    num_elements_ = num_bytes / DataTypeSize(dtype_);
    OP_REQUIRES(context, num_bytes % DataTypeSize(dtype_) == 0,
                errors::InvalidArgument(
                    "Number of bytes ", num_bytes,
                    " must be divisible by size of datatype ", dtype_));
  }

  void Compute(OpKernelContext* context) override {
    ScopedAllocatorMgr* sam = device_->GetScopedAllocatorMgr();
    if (!sam) {
      context->SetStatus(errors::Internal(
          "ScopedAllocatorMgr not supported on device ", device_->name()));
      return;
    }
    Tensor* backing_tensor = nullptr;
    AllocatorAttributes attr = context->output_alloc_attr(0);
    Status s =
        context->allocate_output(0, {num_elements_}, &backing_tensor, attr);
    VLOG(1) << "_ScopedAllocatorOp " << context->op_kernel().name()
            << " new backing tensor size " << backing_tensor->TotalBytes()
            << " num_elements_ " << num_elements_ << " buffer "
            << DMAHelper::buffer(backing_tensor) << " base addr "
            << DMAHelper::base(backing_tensor);
    if (s.ok()) {
      s = sam->AddScopedAllocator(*backing_tensor, context->step_id(), id_,
                                  name_, fields_, expected_call_count_);
    }
    if (!s.ok()) {
      context->SetStatus(s);
    }
  }

 private:
  std::vector<TensorShape> shapes_;
  DataType dtype_;
  int64_t num_elements_;
  std::vector<ScopedAllocator::Field> fields_;
  string name_;
  int32 id_;
  int32 expected_call_count_;
  DeviceBase* device_;
};

REGISTER_KERNEL_BUILDER(Name("_ScopedAllocator").Device(DEVICE_CPU),
                        ScopedAllocatorOp);

REGISTER_KERNEL_BUILDER(Name("_ScopedAllocator").Device(DEVICE_GPU),
                        ScopedAllocatorOp);

REGISTER_KERNEL_BUILDER(Name("_ScopedAllocator").Device(DEVICE_DEFAULT),
                        ScopedAllocatorOp);

class ScopedAllocatorConcatOp : public OpKernel {
 public:
  explicit ScopedAllocatorConcatOp(OpKernelConstruction* context)
      : OpKernel(context) {
    OP_REQUIRES_OK(context, context->GetAttr("shape", &shape_));
    OP_REQUIRES_OK(context, context->GetAttr("T", &dtype_));
    OP_REQUIRES_OK(context, context->GetAttr("reshape", &reshape_));
    // These attributes are just for debugging.
    OP_REQUIRES_OK(context, context->GetAttr("sa_name", &name_));
    OP_REQUIRES_OK(context, context->GetAttr("id", &id_));
    device_ = context->device();
  }

  void Compute(OpKernelContext* context) override {
    const Tensor& backing_tensor = context->input(0);
    // Check that type matches.
    OP_REQUIRES(context, backing_tensor.dtype() == dtype_,
                errors::InvalidArgument("Backing tensor type ",
                                        DataTypeString(backing_tensor.dtype()),
                                        " does not match expected type ",
                                        DataTypeString(dtype_)));
    // Check that backing tensor is at least as large as the shape of the
    // output.
    OP_REQUIRES(context, backing_tensor.NumElements() >= shape_.num_elements(),
                errors::InvalidArgument("Backing tensor num elements ",
                                        backing_tensor.NumElements(),
                                        " is not >= to expected ",
                                        shape_.num_elements()));
    Tensor output(dtype_);
    if (reshape_) {
      CHECK(output.CopyFrom(backing_tensor, shape_));
    } else {
      CHECK(output.CopyFrom(backing_tensor, backing_tensor.shape()));
    }
    context->set_output(0, output);
    const TensorBuffer* backing_buf = DMAHelper::buffer(&output);
    const void* backing_tensor_lb = backing_buf->data();
    const void* backing_tensor_ub = static_cast<const void*>(
        static_cast<const char*>(backing_tensor_lb) + backing_buf->size());
    // Check that all inputs lie entirely within the backing tensor.
    for (int i = 1; i < context->num_inputs(); ++i) {
      const TensorBuffer* input_buf = DMAHelper::buffer(&context->input(i));
      const void* input_lb = input_buf->data();
      const void* input_ub = static_cast<const void*>(
          static_cast<const char*>(input_lb) + input_buf->size());
      OP_REQUIRES(
          context, input_lb >= backing_tensor_lb,
          errors::InvalidArgument(
              "Lower bound check fail for input ", i, " from node ",
              context->op_kernel().requested_input(i), " to node ",
              context->op_kernel().name(), " input bounds = [", input_lb, ", ",
              input_ub, "]", " backing_tensor bounds = [", backing_tensor_lb,
              ", ", backing_tensor_ub, "]"));
      OP_REQUIRES(
          context, input_ub <= backing_tensor_ub,
          errors::InvalidArgument(
              "Upper bound check fail for input ", i, " from node ",
              context->op_kernel().requested_input(i), " to node ",
              context->op_kernel().name(), " input bounds = [", input_lb, ", ",
              input_ub, "]", " backing_tensor bounds = [", backing_tensor_lb,
              ", ", backing_tensor_ub, "]"));
    }
    VLOG(1) << "_ScopedAllocatorConcatOp outputting backing tensor at "
            << backing_buf;
  }

 private:
  TensorShape shape_;
  DataType dtype_;
  string name_;
  int32 id_;
  bool reshape_;
  DeviceBase* device_;
};

REGISTER_KERNEL_BUILDER(Name("_ScopedAllocatorConcat").Device(DEVICE_CPU),
                        ScopedAllocatorConcatOp);

REGISTER_KERNEL_BUILDER(Name("_ScopedAllocatorConcat").Device(DEVICE_GPU),
                        ScopedAllocatorConcatOp);

REGISTER_KERNEL_BUILDER(Name("_ScopedAllocatorConcat").Device(DEVICE_DEFAULT),
                        ScopedAllocatorConcatOp);

class ScopedAllocatorSplitOp : public OpKernel {
 public:
  explicit ScopedAllocatorSplitOp(OpKernelConstruction* context)
      : OpKernel(context) {
    OP_REQUIRES_OK(context, context->GetAttr("T", &dtype_));
    // This stuff is just for debugging
    OP_REQUIRES_OK(context, context->GetAttr("sa_name", &name_));
    OP_REQUIRES_OK(context, context->GetAttr("id", &id_));
    device_ = context->device();
  }

  void Compute(OpKernelContext* context) override {
    Tensor backing_copy(context->input(0));
    // Check that type matches.
    OP_REQUIRES(context, backing_copy.dtype() == dtype_,
                errors::InvalidArgument("Backing tensor type ",
                                        DataTypeString(backing_copy.dtype()),
                                        " does not match expected type ",
                                        DataTypeString(dtype_)));
    const TensorBuffer* backing_buf = DMAHelper::buffer(&backing_copy);
    const void* backing_tensor_lb = backing_buf->data();
    const void* backing_tensor_ub = static_cast<const void*>(
        static_cast<const char*>(backing_tensor_lb) + backing_buf->size());
    for (int i = 1; i < context->num_inputs(); ++i) {
      VLOG(1) << "_ScopedAllocatorSplitOp assigning input " << i
              << " to output " << i - 1 << " buf addr "
              << DMAHelper::base(&context->input(i));
      Tensor copy(context->input(i));
      OP_REQUIRES(context, copy.dtype() == dtype_,
                  errors::InvalidArgument("Input ", i, " tensor type ",
                                          DataTypeString(copy.dtype()),
                                          " does not match expected type ",
                                          DataTypeString(dtype_)));
      context->set_output(i - 1, copy);
      const TensorBuffer* input_buf = DMAHelper::buffer(&copy);
      const void* input_lb = input_buf->data();
      OP_REQUIRES(
          context, input_lb >= backing_tensor_lb,
          errors::InvalidArgument("Lower bound check fail for input ", i,
                                  " to node ", context->op_kernel().name()));
      const void* input_ub = static_cast<const void*>(
          static_cast<const char*>(input_lb) + input_buf->size());
      OP_REQUIRES(
          context, input_ub <= backing_tensor_ub,
          errors::InvalidArgument("Upper bound check fail for input ", i,
                                  " to node ", context->op_kernel().name()));
    }
  }

 private:
  DataType dtype_;
  string name_;
  int32 id_;
  DeviceBase* device_;
};

REGISTER_KERNEL_BUILDER(Name("_ScopedAllocatorSplit").Device(DEVICE_CPU),
                        ScopedAllocatorSplitOp);

REGISTER_KERNEL_BUILDER(Name("_ScopedAllocatorSplit").Device(DEVICE_GPU),
                        ScopedAllocatorSplitOp);

REGISTER_KERNEL_BUILDER(Name("_ScopedAllocatorSplit").Device(DEVICE_DEFAULT),
                        ScopedAllocatorSplitOp);

}  // namespace tensorflow
