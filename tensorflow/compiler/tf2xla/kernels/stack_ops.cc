/* Copyright 2017 The TensorFlow Authors. All Rights Reserved.

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

// XLA Stack operators.

#include <limits>
#include <vector>

#include "absl/status/status.h"
#include "absl/strings/str_cat.h"
#include "tensorflow/compiler/tf2xla/shape_util.h"
#include "tensorflow/compiler/tf2xla/xla_op_kernel.h"
#include "tensorflow/compiler/tf2xla/xla_op_registry.h"
#include "tensorflow/compiler/tf2xla/xla_resource.h"
#include "xla/client/xla_builder.h"
#include "xla/shape.h"
#include "xla/shape_util.h"
#include "xla/status_macros.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/op_requires.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/framework/types.pb.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/platform/errors.h"
#include "tensorflow/core/platform/types.h"
#include "tsl/platform/errors.h"

namespace tensorflow {
namespace {

Status GetStackShape(xla::XlaBuilder* builder, XlaResource* resource,
                     TensorShape* stack_shape) {
  auto shape_or_status = builder->GetShape(resource->value());
  if (!shape_or_status.ok()) {
    return shape_or_status.status();
  }
  xla::Shape shape = shape_or_status.value();
  TF_RET_CHECK(shape.IsTuple());
  return XLAShapeToTensorShape(xla::ShapeUtil::GetTupleElementShape(shape, 0),
                               stack_shape);
}

// Since the element shape is not provided to the Stack operator,
// we lazily initialize the Stack at the time of the first write.
//
// If a Stack `resource` has not been initialized, constructs storage for the
// Stack with elements of `elem_shape`. For both initialized and
// uninitialized Stacks, checks that the tensor has a type compatible with
// 'dtype' and shape compatible with 'elem_shape'.
//
// TODO(phawkins): consider changing the API of the stack operators to
// allow an optional element shape at stack construction time.
Status MaybeInitializeStack(xla::XlaBuilder* builder, XlaResource* resource,
                            DataType dtype, const TensorShape& elem_shape) {
  if (resource->type() != dtype) {
    return errors::InvalidArgument(
        "Stack dtype is ", DataTypeString(resource->type()),
        " but op has dtype ", DataTypeString(dtype), ".");
  }

  TensorShape stack_shape;
  TF_RETURN_IF_ERROR(stack_shape.AddDimWithStatus(resource->max_array_size()));
  stack_shape.AppendShape(elem_shape);

  if (!resource->initialized()) {
    // Stack has not been initialized.
    TF_RETURN_IF_ERROR(resource->SetTypeAndShape(dtype, elem_shape));
    TF_RETURN_IF_ERROR(resource->SetZeroValue(builder));
  } else {
    // Checks the expected shape matches the actual shape.
    TensorShape actual_shape;
    TF_RETURN_IF_ERROR(GetStackShape(builder, resource, &actual_shape));
    if (stack_shape != actual_shape) {
      return errors::InvalidArgument(
          "Mismatched Stack shapes: ", stack_shape.DebugString(), " vs ",
          actual_shape.DebugString());
    }
  }
  return absl::OkStatus();
}

class StackOp : public XlaOpKernel {
 public:
  explicit StackOp(OpKernelConstruction* ctx) : XlaOpKernel(ctx) {
    OP_REQUIRES_OK(ctx, ctx->GetAttr("elem_type", &dtype_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("stack_name", &stack_name_));
  }

  void Compile(XlaOpKernelContext* ctx) override {
    int64_t max_size;
    OP_REQUIRES_OK(ctx, ctx->ConstantInputAsIntScalar(0, &max_size));
    OP_REQUIRES(
        ctx, max_size >= 0,
        errors::InvalidArgument(
            "XLA compilation requires a fixed stack size upper bound. If "
            "you are using tf.while_loop, set the maximum_iterations parameter "
            "to fix this issue."));

    // We defer initializing the Stack resource until we see the first push.
    // Otherwise we do not know the shape of the stack elements.
    XlaResource* resource =
        ctx->xla_context()->AddResource(XlaResource::CreateStack(
            /*name=*/absl::StrCat("Stack: ", stack_name_), dtype_, max_size));
    ctx->SetResourceOutput(0, resource);
  }

 private:
  DataType dtype_;
  string stack_name_;

  StackOp(const StackOp&) = delete;
  void operator=(const StackOp&) = delete;
};

REGISTER_XLA_OP(
    Name("StackV2").CompileTimeConstantInput("max_size").CompilationOnly(),
    StackOp);

class StackPushOp : public XlaOpKernel {
 public:
  explicit StackPushOp(OpKernelConstruction* ctx) : XlaOpKernel(ctx) {
    OP_REQUIRES_OK(ctx, ctx->GetAttr("T", &dtype_));
  }

  void Compile(XlaOpKernelContext* ctx) override {
    xla::XlaBuilder* b = ctx->builder();
    TensorShape elem_shape = ctx->InputShape(1);

    XlaResource* resource;
    OP_REQUIRES_OK(ctx, ctx->GetResourceInput(0, &resource));

    // Initializes the Stack, if the element shape was not already known.
    OP_REQUIRES_OK(ctx, MaybeInitializeStack(b, resource, dtype_, elem_shape));

    xla::XlaOp ta = xla::GetTupleElement(resource->value(), 0);
    xla::XlaOp index = xla::GetTupleElement(resource->value(), 1);
    xla::XlaOp value = ctx->Input(1);

    // start_indices of the DynamicUpdateSlice are [index, 0, 0, ..., 0].
    std::vector<xla::XlaOp> start_indices(elem_shape.dims() + 1,
                                          xla::ConstantR0<int32>(b, 0));
    start_indices[0] = index;

    TensorShape slice_shape = elem_shape;
    slice_shape.InsertDim(0, 1LL);
    auto update = xla::Reshape(value, slice_shape.dim_sizes());

    // TODO(phawkins): We don't check the index is in bounds --- there is no
    // error mechanism in XLA.
    OP_REQUIRES_OK(ctx,
                   resource->SetValue(xla::Tuple(
                       b, {xla::DynamicUpdateSlice(ta, update, start_indices),
                           xla::Add(index, xla::ConstantR0<int32>(b, 1))})));

    ctx->SetOutput(0, value);
  }

 private:
  DataType dtype_;

  StackPushOp(const StackPushOp&) = delete;
  void operator=(const StackPushOp&) = delete;
};

REGISTER_XLA_OP(Name("StackPushV2").CompilationOnly(), StackPushOp);

class StackPopOp : public XlaOpKernel {
 public:
  explicit StackPopOp(OpKernelConstruction* ctx) : XlaOpKernel(ctx) {
    OP_REQUIRES_OK(ctx, ctx->GetAttr("elem_type", &dtype_));
  }

  void Compile(XlaOpKernelContext* ctx) override {
    xla::XlaBuilder* b = ctx->builder();

    XlaResource* resource;
    OP_REQUIRES_OK(ctx, ctx->GetResourceInput(0, &resource));

    // There is a somewhat subtle issue here: here "uninitialized" means we have
    // not yet seen a pop in the order that we compile operators, not the order
    // that we run them. However, in practice the two orders should be the same
    // for the sole user of the stack operators (loop gradients).
    OP_REQUIRES(ctx, resource->initialized(),
                errors::InvalidArgument("Stack pop on uninitialized stack"));

    TensorShape stack_shape;
    OP_REQUIRES_OK(ctx, GetStackShape(b, resource, &stack_shape));

    xla::XlaOp state = resource->value();
    xla::XlaOp ta = xla::GetTupleElement(state, 0);
    xla::XlaOp index = xla::GetTupleElement(state, 1);

    index = Sub(index, xla::ConstantR0<int32>(b, 1));
    OP_REQUIRES_OK(ctx, resource->SetValue(xla::Tuple(b, {ta, index})));

    // start_indices of the DynamicSlice are [index, 0, 0, ..., 0].
    std::vector<xla::XlaOp> start_indices(stack_shape.dims(),
                                          xla::ConstantR0<int32>(b, 0));
    start_indices[0] = index;

    auto slice_shape = stack_shape.dim_sizes();
    slice_shape[0] = 1LL;

    // TODO(phawkins): We don't check the index is in bounds --- there is no
    // error mechanism in XLA.
    xla::XlaOp read = xla::DynamicSlice(ta, start_indices, slice_shape);

    // Remove the leading '1' dimension.
    std::vector<int64_t> value_shape(slice_shape.begin() + 1,
                                     slice_shape.end());
    ctx->SetOutput(0, xla::Reshape(read, value_shape));
  }

 private:
  DataType dtype_;

  StackPopOp(const StackPopOp&) = delete;
  void operator=(const StackPopOp&) = delete;
};

REGISTER_XLA_OP(Name("StackPopV2").CompilationOnly(), StackPopOp);

class StackCloseOp : public XlaOpKernel {
 public:
  explicit StackCloseOp(OpKernelConstruction* ctx) : XlaOpKernel(ctx) {}

  void Compile(XlaOpKernelContext* ctx) override {
    // Do nothing.
  }

 private:
  StackCloseOp(const StackCloseOp&) = delete;
  void operator=(const StackCloseOp&) = delete;
};

REGISTER_XLA_OP(Name("StackCloseV2").CompilationOnly(), StackCloseOp);

}  // anonymous namespace
}  // namespace tensorflow
