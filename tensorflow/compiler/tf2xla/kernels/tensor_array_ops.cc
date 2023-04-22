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
#include "tensorflow/compiler/tf2xla/xla_resource.h"
#include "tensorflow/compiler/xla/client/xla_builder.h"
#include "tensorflow/compiler/xla/literal.h"
#include "tensorflow/core/framework/bounds_check.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/partial_tensor_shape.h"
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_types.h"
#include "tensorflow/core/framework/types.h"
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
Status MaybeInitializeTensorArray(xla::XlaBuilder* builder,
                                  XlaResource* resource, DataType dtype,
                                  const TensorShape& elem_shape) {
  if (resource->kind() != XlaResource::kTensorArray) {
    return errors::InvalidArgument("Unexpected non-TensorArray resource");
  }

  if (resource->type() != dtype) {
    return errors::InvalidArgument(
        "TensorArray dtype is ", DataTypeString(resource->type()),
        " but op has dtype ", DataTypeString(dtype), ".");
  }

  TF_RET_CHECK(resource->max_array_size() >= 0)
      << resource->name() << " size " << resource->max_array_size();

  if (!resource->initialized()) {
    TF_RETURN_IF_ERROR(resource->SetTypeAndShape(dtype, elem_shape));
    TF_RETURN_IF_ERROR(resource->SetZeroValue(builder));
  } else {
    // Checks the elem_shape matches the TensorArray shape.
    auto shape_or_status = builder->GetShape(resource->value());
    if (!shape_or_status.ok()) {
      return shape_or_status.status();
    }
    TensorShape shape;
    TF_RETURN_IF_ERROR(
        XLAShapeToTensorShape(shape_or_status.ValueOrDie(), &shape));

    TensorShape ta_shape;
    ta_shape.AddDim(resource->max_array_size());
    ta_shape.AppendShape(elem_shape);
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
  if (resource->kind() != XlaResource::kTensorArray) {
    return errors::InvalidArgument(
        "Unexpected non-TensorArray resource passed to ", op_name);
  }
  if (!resource->initialized()) {
    return errors::InvalidArgument("Uninitialized TensorArray passed to ",
                                   op_name);
  }
  if (resource->type() != dtype) {
    return errors::InvalidArgument(
        "TensorArray dtype is ", DataTypeString(resource->type()),
        " but op has dtype ", DataTypeString(dtype), ".");
  }

  return Status::OK();
}

Status GetTensorArrayShape(const XlaResource* resource,
                           xla::XlaBuilder* builder, TensorShape* shape) {
  *shape = resource->shape();
  shape->InsertDim(0, resource->max_array_size());
  return Status::OK();
}

// Like XlaBuilder::DynamicUpdateSlice, but adds 'update' to the
// relevant slice of 'operand'.
xla::XlaOp DynamicAddSlice(xla::XlaBuilder* builder, const xla::XlaOp& operand,
                           const xla::XlaOp& update,
                           absl::Span<const int64> update_dims,
                           absl::Span<const xla::XlaOp> start_indices,
                           DataType dtype) {
  xla::XlaOp current = xla::DynamicSlice(operand, start_indices, update_dims);
  xla::XlaOp sum =
      dtype == DT_BOOL ? xla::Or(current, update) : xla::Add(current, update);
  return xla::DynamicUpdateSlice(operand, sum, start_indices);
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

    xla::XlaBuilder* b = ctx->builder();

    // Initializes the TensorArray value if we know the element shape.
    // Otherwise, defer initialization to the first write.
    xla::XlaOp value;
    TensorShape shape;
    if (element_shape_.IsFullyDefined()) {
      CHECK(element_shape_.AsTensorShape(&shape));
      TensorShape ta_shape;
      ta_shape.AddDim(size);
      ta_shape.AppendShape(shape);
      xla::XlaOp zero = XlaHelpers::Zero(b, dtype_);
      value = xla::Broadcast(zero, ta_shape.dim_sizes());
    }

    XlaResource* var =
        ctx->xla_context()->AddResource(XlaResource::CreateTensorArray(
            /*name=*/absl::StrCat("TensorArray: ", tensor_array_name_), dtype_,
            shape, /*initial_value=*/value, /*max_array_size=*/size));
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

REGISTER_XLA_OP(Name("TensorArrayV3").CompileTimeConstantInput("size"),
                TensorArrayOp);

class TensorArrayWriteOp : public XlaOpKernel {
 public:
  explicit TensorArrayWriteOp(OpKernelConstruction* ctx) : XlaOpKernel(ctx) {
    OP_REQUIRES_OK(ctx, ctx->GetAttr("T", &dtype_));
  }

  void Compile(XlaOpKernelContext* ctx) override {
    xla::XlaBuilder* b = ctx->builder();

    TensorShape elem_shape = ctx->InputShape(2);

    // Initializes the TensorArray, if the element shape was not known at
    // construction time.
    XlaResource* resource;
    OP_REQUIRES_OK(ctx, ctx->GetResourceInput(0, &resource));
    OP_REQUIRES_OK(ctx,
                   MaybeInitializeTensorArray(b, resource, dtype_, elem_shape));

    xla::XlaOp ta = resource->value();
    xla::XlaOp index = ctx->Input(1);
    xla::XlaOp value = ctx->Input(2);
    xla::XlaOp flow = ctx->Input(3);

    // start_indices of the DynamicUpdateSlice are [index, 0, 0, ..., 0].
    std::vector<xla::XlaOp> start_indices(elem_shape.dims() + 1,
                                          xla::ConstantR0<int32>(b, 0));
    start_indices[0] = index;

    TensorShape slice_shape = elem_shape;
    slice_shape.InsertDim(0, 1LL);
    auto update = xla::Reshape(value, slice_shape.dim_sizes());

    xla::XlaOp written;
    if (resource->tensor_array_multiple_writes_aggregate()) {
      written = DynamicAddSlice(b, ta, update, slice_shape.dim_sizes(),
                                start_indices, dtype_);
    } else {
      // TODO(b/117569591): Ideally we would report an error in the case that we
      // see multiple writes to the same offset. Unfortunately there is no way
      // to report errors at the moment, so we silently overwrite.
      written = xla::DynamicUpdateSlice(ta, update, start_indices);
    }
    OP_REQUIRES_OK(ctx, resource->SetValue(written));
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
    xla::XlaBuilder* b = ctx->builder();

    XlaResource* resource;
    OP_REQUIRES_OK(ctx, ctx->GetResourceInput(0, &resource));

    OP_REQUIRES_OK(ctx,
                   CheckTensorArrayIsInitialized(name(), resource, dtype_));
    TensorShape ta_shape;
    OP_REQUIRES_OK(ctx, GetTensorArrayShape(resource, b, &ta_shape));

    xla::XlaOp ta = resource->value();
    xla::XlaOp index = ctx->Input(1);

    // start_indices of the DynamicSlice are [index, 0, 0, ..., 0].
    std::vector<xla::XlaOp> start_indices(ta_shape.dims(),
                                          xla::ConstantR0<int32>(b, 0));
    start_indices[0] = index;

    auto slice_shape = ta_shape.dim_sizes();
    slice_shape[0] = 1LL;

    xla::XlaOp read = xla::DynamicSlice(ta, start_indices, slice_shape);

    // Remove the leading '1' dimension.
    std::vector<int64> value_shape(slice_shape.begin() + 1, slice_shape.end());
    ctx->SetOutput(0, xla::Reshape(read, value_shape));
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
    xla::XlaBuilder* b = ctx->builder();

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
    DataType index_type = ctx->input_type(1);

    xla::XlaOp ta = resource->value();

    // Look for the case where the gather takes a simple slice from the
    // tensor array (0, 1, 2, 3, 4, ..., N)
    std::vector<int64> const_indices;
    Status status = ctx->ConstantInputAsIntVector(1, &const_indices);
    if (status.ok()) {
      bool gather_is_dense_slice = true;
      for (auto i = 0; i < const_indices.size(); i++) {
        if (const_indices[i] != i) {
          gather_is_dense_slice = false;
          break;
        }
      }

      if (gather_is_dense_slice) {
        std::vector<int64> begin(ta_shape.dims(), 0);
        std::vector<int64> strides(ta_shape.dims(), 1);
        std::vector<int64> end(ta_shape.dims(), 1);
        end[0] = const_indices.size();
        for (auto i = 1; i < ta_shape.dims(); i++) {
          end[i] = ta_shape.dim_size(i);
        }
        ctx->SetOutput(0, xla::Slice(ta, begin, end, strides));
        return;
      }
    }

    xla::XlaOp gather;
    OP_REQUIRES_OK(
        ctx,
        XlaGather(ta, ta_shape, indices, indices_shape, /*axis=*/0,
                  /*indices_are_nd=*/false, dtype_, index_type, b, &gather));
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
    xla::XlaBuilder* b = ctx->builder();

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
    const xla::XlaOp indices = ctx->Input(1);

    xla::XlaOp ta = resource->value();
    const xla::XlaOp value = ctx->Input(2);
    const xla::XlaOp flow = ctx->Input(3);

    // Look for the case where the scatter is for each sub-tensor in order. The
    // tensor array implementation allows for this to be a straight addition.
    bool scatter_all_elements_in_order = false;
    std::vector<int64> const_indices;
    Status status = ctx->ConstantInputAsIntVector(1, &const_indices);
    if (status.ok() && num_indices == value_shape.dim_size(0)) {
      scatter_all_elements_in_order = true;
      for (auto i = 0; i < num_indices; i++) {
        if (const_indices[i] != i) {
          scatter_all_elements_in_order = false;
          break;
        }
      }
    }

    if (scatter_all_elements_in_order) {
      if (dtype_ == DT_BOOL) {
        ta = xla::Or(ta, value);
      } else {
        ta = xla::Add(ta, value);
      }
    } else {
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
        auto slice = xla::Slice(value, value_starts, value_ends, value_strides);

        // start_indices of the DynamicUpdateSlice are [index, 0, 0, ..., 0].
        auto index = xla::Reshape(xla::Slice(indices, {i}, {i + 1}, {1}), {});
        std::vector<xla::XlaOp> start_indices(elem_shape.dims() + 1,
                                              xla::ConstantR0<int32>(b, 0));
        start_indices[0] = index;
        ta = DynamicAddSlice(b, ta, slice, slice_dims, start_indices, dtype_);
      }
    }

    OP_REQUIRES_OK(ctx, resource->SetValue(ta));
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
    xla::XlaBuilder* b = ctx->builder();

    XlaResource* resource;
    OP_REQUIRES_OK(ctx, ctx->GetResourceInput(0, &resource));

    OP_REQUIRES_OK(ctx,
                   CheckTensorArrayIsInitialized(name(), resource, dtype_));
    TensorShape ta_shape;
    OP_REQUIRES_OK(ctx, GetTensorArrayShape(resource, b, &ta_shape));

    xla::XlaOp ta = resource->value();

    auto ta_dims = ta_shape.dim_sizes();
    std::vector<int64> shape(ta_dims.begin() + 1, ta_dims.end());
    shape[0] *= ta_shape.dim_size(0);
    ctx->SetOutput(0, xla::Reshape(ta, shape));

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

    xla::XlaBuilder* b = ctx->builder();
    XlaResource* resource;
    OP_REQUIRES_OK(ctx, ctx->GetResourceInput(0, &resource));
    OP_REQUIRES_OK(ctx,
                   MaybeInitializeTensorArray(b, resource, dtype_, elem_shape));
    xla::XlaOp ta = resource->value();

    TensorShape ta_shape;
    ta_shape.AddDim(resource->max_array_size());
    ta_shape.AppendShape(elem_shape);

    OP_REQUIRES(ctx, lengths.size() == resource->max_array_size(),
                errors::InvalidArgument(
                    "TensorArray's size is not equal to the size of lengths (",
                    lengths.size(), " vs. ", resource->max_array_size(), ")"));

    const xla::XlaOp value = ctx->Input(1);
    const xla::XlaOp flow = ctx->Input(3);

    OP_REQUIRES(ctx, value_shape.num_elements() == ta_shape.num_elements(),
                errors::InvalidArgument("mismatched element count ",
                                        value_shape.DebugString(), " vs. ",
                                        ta_shape.DebugString()));

    const xla::XlaOp reshape = xla::Reshape(value, ta_shape.dim_sizes());
    if (dtype_ == DT_BOOL) {
      ta = xla::Or(ta, reshape);
    } else {
      ta = xla::Add(ta, reshape);
    }
    OP_REQUIRES_OK(ctx, resource->SetValue(ta));

    ctx->SetOutput(0, flow);
  }

 private:
  DataType dtype_;

  TF_DISALLOW_COPY_AND_ASSIGN(TensorArraySplitOp);
};

REGISTER_XLA_OP(Name("TensorArraySplitV3").CompileTimeConstantInput("lengths"),
                TensorArraySplitOp);

class TensorArraySizeOp : public XlaOpKernel {
 public:
  explicit TensorArraySizeOp(OpKernelConstruction* ctx) : XlaOpKernel(ctx) {}

  void Compile(XlaOpKernelContext* ctx) override {
    XlaResource* var;
    OP_REQUIRES_OK(ctx, ctx->GetResourceInput(0, &var));
    Tensor size_tensor(DT_INT32, {});
    size_tensor.scalar<int32>()() = static_cast<int32>(var->max_array_size());
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
    xla::XlaBuilder* b = ctx->builder();

    XlaResource* resource;
    OP_REQUIRES_OK(ctx, ctx->GetResourceInput(0, &resource));

    OP_REQUIRES_OK(
        ctx, CheckTensorArrayIsInitialized(name(), resource, resource->type()));
    TensorShape ta_shape;
    OP_REQUIRES_OK(ctx, GetTensorArrayShape(resource, b, &ta_shape));

    // Finds or looks up the corresponding gradient TensorArray, which stores
    // gradients computed during backpropagation.
    XlaResource* gradient;
    OP_REQUIRES_OK(
        ctx, resource->GetOrCreateTensorArrayGradient(source_, b, &gradient));

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
