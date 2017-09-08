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

// XLA TensorArray operators.

#include <limits>
#include <vector>

#include "tensorflow/compiler/tf2xla/kernels/gather_op_helpers.h"
#include "tensorflow/compiler/tf2xla/shape_util.h"
#include "tensorflow/compiler/tf2xla/type_util.h"
#include "tensorflow/compiler/tf2xla/xla_helpers.h"
#include "tensorflow/compiler/tf2xla/xla_op_kernel.h"
#include "tensorflow/compiler/tf2xla/xla_op_registry.h"
#include "tensorflow/compiler/xla/literal_util.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/partial_tensor_shape.h"
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_types.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/kernels/bounds_check.h"
#include "tensorflow/core/kernels/concat_lib.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/platform/types.h"

namespace tensorflow {
namespace {

// Since the element shape is not always provided to the TensorArrayV3 operator,
// we must support lazily initialization of the TensorArray at the time of the
// first write.
// If a TensorArray `resource` has not been initialized, constructs storage for
// the TensorArray with elements of `elem_shape`. For both initialized and
// uninitialized TensorArrays, checks that the tensor has a type compatible with
// 'dtype' and shape compatible with 'elem_shape'.
Status MaybeInitializeTensorArray(xla::ComputationBuilder* builder,
                                  XlaResource* resource, DataType dtype,
                                  const TensorShape& elem_shape) {
  if (resource->kind != XlaResource::kTensorArray) {
    return errors::InvalidArgument("Unexpected non-TensorArray resource");
  }

  if (resource->type != dtype) {
    return errors::InvalidArgument(
        "TensorArray dtype is ", DataTypeString(resource->type),
        " but op has dtype ", DataTypeString(dtype), ".");
  }

  TF_RET_CHECK(resource->tensor_array_size >= 0)
      << resource->name << " size " << resource->tensor_array_size;
  TensorShape ta_shape;
  ta_shape.AddDim(resource->tensor_array_size);
  ta_shape.AppendShape(elem_shape);

  if (resource->value.handle() == 0) {
    // TensorArray has not been initialized.
    xla::ComputationDataHandle zero = XlaHelpers::Zero(builder, resource->type);
    resource->value = builder->Broadcast(zero, ta_shape.dim_sizes());
  } else {
    // Checks the elem_shape matches the TensorArray shape.
    auto shape_or_status = builder->GetShape(resource->value);
    if (!shape_or_status.ok()) {
      return shape_or_status.status();
    }
    TensorShape shape;
    TF_RETURN_IF_ERROR(
        XLAShapeToTensorShape(*shape_or_status.ValueOrDie(), &shape));
    if (ta_shape != shape) {
      return errors::InvalidArgument(
          "Mismatched TensorArray sizes: ", ta_shape.DebugString(), " vs ",
          shape.DebugString());
    }
  }
  return Status::OK();
}

// Checks that the TensorArray 'resource' has been initialized, and has type
// 'dtype'. Sets 'shape' to the shape
Status CheckTensorArrayIsInitialized(const string& op_name,
                                     const XlaResource* resource,
                                     DataType dtype) {
  if (resource->kind != XlaResource::kTensorArray) {
    return errors::InvalidArgument(
        "Unexpected non-TensorArray resource passed "
        "to ",
        op_name);
  }
  if (resource->value.handle() == 0) {
    return errors::InvalidArgument("Uninitialized TensorArray passed to ",
                                   op_name);
  }
  if (resource->type != dtype) {
    return errors::InvalidArgument(
        "TensorArray dtype is ", DataTypeString(resource->type),
        " but op has dtype ", DataTypeString(dtype), ".");
  }

  return Status::OK();
}

Status GetTensorArrayShape(const XlaResource* resource,
                           xla::ComputationBuilder* builder,
                           TensorShape* shape) {
  auto shape_or_status = builder->GetShape(resource->value);
  if (!shape_or_status.ok()) {
    return shape_or_status.status();
  }
  TF_RETURN_IF_ERROR(
      XLAShapeToTensorShape(*shape_or_status.ValueOrDie(), shape));
  if (shape->dims() < 1) {
    return errors::InvalidArgument("TensorArray rank must be >= 1");
  }
  return Status::OK();
}

// Like ComputationBuilder::DynamicUpdateSlice, but adds 'update' to the
// relevant slice of 'operand'.
xla::ComputationDataHandle DynamicAddSlice(
    xla::ComputationBuilder* builder, const xla::ComputationDataHandle& operand,
    const xla::ComputationDataHandle& update,
    const gtl::ArraySlice<int64>& update_dims,
    const xla::ComputationDataHandle& start_indices) {
  xla::ComputationDataHandle current =
      builder->DynamicSlice(operand, start_indices, update_dims);
  xla::ComputationDataHandle sum = builder->Add(current, update);
  return builder->DynamicUpdateSlice(operand, sum, start_indices);
}

class TensorArrayOp : public XlaOpKernel {
 public:
  explicit TensorArrayOp(OpKernelConstruction* ctx) : XlaOpKernel(ctx) {
    OP_REQUIRES_OK(ctx, ctx->GetAttr("element_shape", &element_shape_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("dtype", &dtype_));
    bool dynamic_size;
    OP_REQUIRES_OK(ctx, ctx->GetAttr("dynamic_size", &dynamic_size));
    OP_REQUIRES(
        ctx, !dynamic_size,
        errors::Unimplemented(
            "TensorArrays with dynamic size are not supported by XLA."));

    OP_REQUIRES_OK(ctx, ctx->GetAttr("tensor_array_name", &tensor_array_name_));
  }

  void Compile(XlaOpKernelContext* ctx) override {
    int64 size;
    OP_REQUIRES_OK(ctx, ctx->ConstantInputAsIntScalar(0, &size));
    OP_REQUIRES(ctx, size >= 0,
                errors::InvalidArgument("TensorArray size must be >= 0"));

    xla::ComputationBuilder* b = ctx->builder();

    // Initializes the TensorArray value if we know the element shape.
    // Otherwise, defer initialization to the first write.
    xla::ComputationDataHandle value;
    if (element_shape_.IsFullyDefined()) {
      TensorShape shape;
      CHECK(element_shape_.AsTensorShape(&shape));
      TensorShape ta_shape;
      ta_shape.AddDim(size);
      ta_shape.AppendShape(shape);
      xla::ComputationDataHandle zero = XlaHelpers::Zero(b, dtype_);
      value = b->Broadcast(zero, ta_shape.dim_sizes());
    }

    XlaContext& xc = XlaContext::Get(ctx);
    XlaResource* var;
    string name = strings::StrCat("TensorArray: ", tensor_array_name_);
    OP_REQUIRES_OK(
        ctx, xc.CreateResource(XlaResource::kTensorArray, -1, std::move(name),
                               dtype_, value, &var));
    var->tensor_array_size = size;
    ctx->SetResourceOutput(0, var);

    Tensor flow(DT_FLOAT, TensorShape({}));
    flow.scalar<float>()() = 0.0f;
    ctx->SetConstantOutput(1, flow);
  }

 private:
  PartialTensorShape element_shape_;
  DataType dtype_;
  string tensor_array_name_;

  TF_DISALLOW_COPY_AND_ASSIGN(TensorArrayOp);
};

REGISTER_XLA_OP(Name("TensorArrayV3"), TensorArrayOp);

class TensorArrayWriteOp : public XlaOpKernel {
 public:
  explicit TensorArrayWriteOp(OpKernelConstruction* ctx) : XlaOpKernel(ctx) {
    OP_REQUIRES_OK(ctx, ctx->GetAttr("T", &dtype_));
  }

  void Compile(XlaOpKernelContext* ctx) override {
    xla::ComputationBuilder* b = ctx->builder();

    TensorShape elem_shape = ctx->InputShape(2);

    // Initializes the TensorArray, if the element shape was not known at
    // construction time.
    XlaResource* resource;
    OP_REQUIRES_OK(ctx, ctx->GetResourceInput(0, &resource));
    OP_REQUIRES_OK(ctx,
                   MaybeInitializeTensorArray(b, resource, dtype_, elem_shape));

    xla::ComputationDataHandle ta = resource->value;
    xla::ComputationDataHandle index = ctx->Input(1);
    xla::ComputationDataHandle value = ctx->Input(2);
    xla::ComputationDataHandle flow = ctx->Input(3);

    // start_indices of the DynamicUpdateSlice are [index, 0, 0, ..., 0].
    auto start_indices = XlaHelpers::PadWithZeros(b, index, elem_shape.dims());

    TensorShape slice_shape = elem_shape;
    slice_shape.InsertDim(0, 1LL);
    auto update = b->Reshape(value, slice_shape.dim_sizes());

    xla::ComputationDataHandle written =
        DynamicAddSlice(b, ta, update, slice_shape.dim_sizes(), start_indices);

    resource->value = written;
    ctx->SetOutput(0, flow);
  }

 private:
  DataType dtype_;

  TF_DISALLOW_COPY_AND_ASSIGN(TensorArrayWriteOp);
};

REGISTER_XLA_OP(Name("TensorArrayWriteV3"), TensorArrayWriteOp);

class TensorArrayReadOp : public XlaOpKernel {
 public:
  explicit TensorArrayReadOp(OpKernelConstruction* ctx) : XlaOpKernel(ctx) {
    OP_REQUIRES_OK(ctx, ctx->GetAttr("dtype", &dtype_));
  }

  void Compile(XlaOpKernelContext* ctx) override {
    xla::ComputationBuilder* b = ctx->builder();

    XlaResource* resource;
    OP_REQUIRES_OK(ctx, ctx->GetResourceInput(0, &resource));

    OP_REQUIRES_OK(ctx,
                   CheckTensorArrayIsInitialized(name(), resource, dtype_));
    TensorShape ta_shape;
    OP_REQUIRES_OK(ctx, GetTensorArrayShape(resource, b, &ta_shape));

    xla::ComputationDataHandle ta = resource->value;
    xla::ComputationDataHandle index = ctx->Input(1);

    // start_indices of the DynamicSlice are [index, 0, 0, ..., 0].
    auto start_indices =
        XlaHelpers::PadWithZeros(b, index, ta_shape.dims() - 1);

    auto slice_shape = ta_shape.dim_sizes();
    slice_shape[0] = 1LL;

    xla::ComputationDataHandle read =
        b->DynamicSlice(ta, start_indices, slice_shape);

    // Remove the leading '1' dimension.
    std::vector<int64> value_shape(slice_shape.begin() + 1, slice_shape.end());
    ctx->SetOutput(0, b->Reshape(read, value_shape));
  }

 private:
  DataType dtype_;

  TF_DISALLOW_COPY_AND_ASSIGN(TensorArrayReadOp);
};

REGISTER_XLA_OP(Name("TensorArrayReadV3"), TensorArrayReadOp);

class TensorArrayGatherOp : public XlaOpKernel {
 public:
  explicit TensorArrayGatherOp(OpKernelConstruction* ctx) : XlaOpKernel(ctx) {
    OP_REQUIRES_OK(ctx, ctx->GetAttr("dtype", &dtype_));
  }

  void Compile(XlaOpKernelContext* ctx) override {
    xla::ComputationBuilder* b = ctx->builder();

    XlaResource* resource;
    OP_REQUIRES_OK(ctx, ctx->GetResourceInput(0, &resource));

    OP_REQUIRES_OK(ctx,
                   CheckTensorArrayIsInitialized(name(), resource, dtype_));
    TensorShape ta_shape;
    OP_REQUIRES_OK(ctx, GetTensorArrayShape(resource, b, &ta_shape));

    const TensorShape indices_shape = ctx->InputShape(1);
    OP_REQUIRES(ctx, indices_shape.dims() == 1,
                errors::InvalidArgument("indices must be rank 1"));
    auto indices = ctx->Input(1);

    xla::ComputationDataHandle ta = resource->value;

    xla::ComputationDataHandle gather = XlaComputeGatherDynamicSlice(
        ctx, ta, ta_shape, indices, indices_shape, dtype_, b);
    ctx->SetOutput(0, gather);
  }

 private:
  DataType dtype_;

  TF_DISALLOW_COPY_AND_ASSIGN(TensorArrayGatherOp);
};

REGISTER_XLA_OP(Name("TensorArrayGatherV3"), TensorArrayGatherOp);

class TensorArrayScatterOp : public XlaOpKernel {
 public:
  explicit TensorArrayScatterOp(OpKernelConstruction* ctx) : XlaOpKernel(ctx) {
    OP_REQUIRES_OK(ctx, ctx->GetAttr("T", &dtype_));
  }

  void Compile(XlaOpKernelContext* ctx) override {
    xla::ComputationBuilder* b = ctx->builder();

    const TensorShape value_shape = ctx->InputShape(2);

    XlaResource* resource;
    OP_REQUIRES_OK(ctx, ctx->GetResourceInput(0, &resource));
    TensorShape elem_shape = value_shape;
    elem_shape.RemoveDim(0);
    OP_REQUIRES_OK(ctx,
                   MaybeInitializeTensorArray(b, resource, dtype_, elem_shape));

    const TensorShape indices_shape = ctx->InputShape(1);
    OP_REQUIRES(ctx, indices_shape.dims() >= 1,
                errors::InvalidArgument("indices must be rank 1"));
    const int num_indices = indices_shape.dim_size(0);
    const xla::ComputationDataHandle indices = ctx->Input(1);

    xla::ComputationDataHandle ta = resource->value;
    const xla::ComputationDataHandle value = ctx->Input(2);
    const xla::ComputationDataHandle flow = ctx->Input(3);

    auto slice_dims = value_shape.dim_sizes();
    slice_dims[0] = 1LL;

    std::vector<int64> value_starts(value_shape.dims(), 0);
    auto value_ends = value_shape.dim_sizes();

    std::vector<int64> value_strides(value_shape.dims(), 1);

    // For every (index, value) pair, update the corresponding TensorArray
    // storage.
    for (int i = 0; i < num_indices; ++i) {
      // Slice out part of the value.
      value_starts[0] = i;
      value_ends[0] = i + 1;
      auto slice = b->Slice(value, value_starts, value_ends, value_strides);

      // start_indices of the DynamicUpdateSlice are [index, 0, 0, ..., 0].
      auto index = b->Slice(indices, {i}, {i + 1}, {1});
      auto start_indices =
          XlaHelpers::PadWithZeros(b, index, elem_shape.dims());
      ta = DynamicAddSlice(b, ta, slice, slice_dims, start_indices);
    }

    resource->value = ta;
    ctx->SetOutput(0, flow);
  }

 private:
  DataType dtype_;

  TF_DISALLOW_COPY_AND_ASSIGN(TensorArrayScatterOp);
};

REGISTER_XLA_OP(Name("TensorArrayScatterV3"), TensorArrayScatterOp);

class TensorArrayConcatOp : public XlaOpKernel {
 public:
  explicit TensorArrayConcatOp(OpKernelConstruction* ctx) : XlaOpKernel(ctx) {
    OP_REQUIRES_OK(ctx, ctx->GetAttr("dtype", &dtype_));
  }

  void Compile(XlaOpKernelContext* ctx) override {
    xla::ComputationBuilder* b = ctx->builder();

    XlaResource* resource;
    OP_REQUIRES_OK(ctx, ctx->GetResourceInput(0, &resource));

    OP_REQUIRES_OK(ctx,
                   CheckTensorArrayIsInitialized(name(), resource, dtype_));
    TensorShape ta_shape;
    OP_REQUIRES_OK(ctx, GetTensorArrayShape(resource, b, &ta_shape));

    xla::ComputationDataHandle ta = resource->value;

    auto ta_dims = ta_shape.dim_sizes();
    std::vector<int64> shape(ta_dims.begin() + 1, ta_dims.end());
    shape[0] *= ta_shape.dim_size(0);
    ctx->SetOutput(0, b->Reshape(ta, shape));

    Tensor lengths(DT_INT64, {ta_dims[0]});
    auto lengths_vec = lengths.vec<int64>();
    for (int i = 0; i < ta_dims[0]; ++i) {
      lengths_vec(i) = ta_dims[1];
    }
    ctx->SetConstantOutput(1, lengths);
  }

 private:
  DataType dtype_;

  TF_DISALLOW_COPY_AND_ASSIGN(TensorArrayConcatOp);
};

REGISTER_XLA_OP(Name("TensorArrayConcatV3"), TensorArrayConcatOp);

class TensorArraySplitOp : public XlaOpKernel {
 public:
  explicit TensorArraySplitOp(OpKernelConstruction* ctx) : XlaOpKernel(ctx) {
    OP_REQUIRES_OK(ctx, ctx->GetAttr("T", &dtype_));
  }

  void Compile(XlaOpKernelContext* ctx) override {
    std::vector<int64> lengths;
    OP_REQUIRES_OK(ctx, ctx->ConstantInputAsIntVector(2, &lengths));

    int64 length = 0;
    if (!lengths.empty()) {
      length = lengths[0];
      for (int i = 1; i < lengths.size(); ++i) {
        OP_REQUIRES(ctx, lengths[i] == length,
                    errors::InvalidArgument("lengths must be equal: ", length,
                                            " vs. ", lengths[i]));
      }
    }

    TensorShape value_shape = ctx->InputShape(1);
    OP_REQUIRES(ctx, value_shape.dims() >= 1,
                errors::InvalidArgument("value must have rank >= 1, got ",
                                        value_shape.DebugString()));
    TensorShape elem_shape = value_shape;
    elem_shape.set_dim(0, length);

    xla::ComputationBuilder* b = ctx->builder();
    XlaResource* resource;
    OP_REQUIRES_OK(ctx, ctx->GetResourceInput(0, &resource));
    OP_REQUIRES_OK(ctx,
                   MaybeInitializeTensorArray(b, resource, dtype_, elem_shape));
    xla::ComputationDataHandle ta = resource->value;

    TensorShape ta_shape;
    ta_shape.AddDim(resource->tensor_array_size);
    ta_shape.AppendShape(elem_shape);

    OP_REQUIRES(ctx, lengths.size() == resource->tensor_array_size,
                errors::InvalidArgument(
                    "TensorArray's size is not equal to the size of lengths (",
                    lengths.size(), " vs. ", resource->tensor_array_size, ")"));

    const xla::ComputationDataHandle value = ctx->Input(1);
    const xla::ComputationDataHandle flow = ctx->Input(3);

    OP_REQUIRES(ctx, value_shape.num_elements() == ta_shape.num_elements(),
                errors::InvalidArgument("mismatched element count ",
                                        value_shape.DebugString(), " vs. ",
                                        ta_shape.DebugString()));

    resource->value = b->Add(ta, b->Reshape(value, ta_shape.dim_sizes()));

    ctx->SetOutput(0, flow);
  }

 private:
  DataType dtype_;

  TF_DISALLOW_COPY_AND_ASSIGN(TensorArraySplitOp);
};

REGISTER_XLA_OP(Name("TensorArraySplitV3"), TensorArraySplitOp);

class TensorArraySizeOp : public XlaOpKernel {
 public:
  explicit TensorArraySizeOp(OpKernelConstruction* ctx) : XlaOpKernel(ctx) {}

  void Compile(XlaOpKernelContext* ctx) override {
    XlaResource* var;
    OP_REQUIRES_OK(ctx, ctx->GetResourceInput(0, &var));
    Tensor size_tensor(DT_INT32, {});
    size_tensor.scalar<int32>()() = static_cast<int32>(var->tensor_array_size);
    ctx->SetConstantOutput(0, size_tensor);
  }

 private:
  TF_DISALLOW_COPY_AND_ASSIGN(TensorArraySizeOp);
};

REGISTER_XLA_OP(Name("TensorArraySizeV3"), TensorArraySizeOp);

class TensorArrayGradOp : public XlaOpKernel {
 public:
  explicit TensorArrayGradOp(OpKernelConstruction* ctx) : XlaOpKernel(ctx) {
    OP_REQUIRES_OK(ctx, ctx->GetAttr("source", &source_));
  }

  void Compile(XlaOpKernelContext* ctx) override {
    xla::ComputationBuilder* b = ctx->builder();

    XlaResource* resource;
    OP_REQUIRES_OK(ctx, ctx->GetResourceInput(0, &resource));

    OP_REQUIRES_OK(
        ctx, CheckTensorArrayIsInitialized(name(), resource, resource->type));
    TensorShape ta_shape;
    OP_REQUIRES_OK(ctx, GetTensorArrayShape(resource, b, &ta_shape));

    // Finds or looks up the corresponding gradient TensorArray, which stores
    // gradients computed during backpropagation.
    XlaResource*& gradient = resource->tensor_array_gradient[source_];
    if (!gradient) {
      xla::ComputationDataHandle zero = XlaHelpers::Zero(b, resource->type);
      xla::ComputationDataHandle value =
          b->Broadcast(zero, ta_shape.dim_sizes());

      XlaContext& xc = XlaContext::Get(ctx);
      string name = strings::StrCat("TensorArrayGrad: ", resource->name);
      OP_REQUIRES_OK(
          ctx, xc.CreateResource(XlaResource::kTensorArray, -1, std::move(name),
                                 resource->type, value, &gradient));
      gradient->tensor_array_size = resource->tensor_array_size;
    }

    ctx->SetResourceOutput(0, gradient);
    ctx->SetConstantOutput(1, Tensor(DT_FLOAT));
  }

 private:
  string source_;

  TF_DISALLOW_COPY_AND_ASSIGN(TensorArrayGradOp);
};

REGISTER_XLA_OP(Name("TensorArrayGradV3"), TensorArrayGradOp);

class TensorArrayCloseOp : public XlaOpKernel {
 public:
  explicit TensorArrayCloseOp(OpKernelConstruction* ctx) : XlaOpKernel(ctx) {}

  void Compile(XlaOpKernelContext* ctx) override {
    // Do nothing; XLA handles resource management.
  }

 private:
  TF_DISALLOW_COPY_AND_ASSIGN(TensorArrayCloseOp);
};

REGISTER_XLA_OP(Name("TensorArrayCloseV3"), TensorArrayCloseOp);

}  // anonymous namespace
}  // namespace tensorflow
