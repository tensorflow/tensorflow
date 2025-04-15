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

// XLA-specific Shape Ops.

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <vector>

#include "absl/container/flat_hash_set.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "tensorflow/compiler/tf2xla/kernels/shape_util.h"
#include "tensorflow/compiler/tf2xla/kernels/tensor_list_utils.h"
#include "tensorflow/compiler/tf2xla/xla_helpers.h"
#include "tensorflow/compiler/tf2xla/xla_op_kernel.h"
#include "tensorflow/compiler/tf2xla/xla_op_registry.h"
#include "xla/hlo/builder/lib/constants.h"
#include "xla/hlo/builder/xla_builder.h"
#include "xla/literal.h"
#include "xla/literal_util.h"
#include "xla/shape.h"
#include "tensorflow/core/framework/bounds_check.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/op_requires.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/framework/types.pb.h"
#include "tensorflow/core/platform/errors.h"
#include "tensorflow/core/platform/types.h"

namespace tensorflow {
namespace {

class ShapeOp : public XlaOpKernel {
 public:
  explicit ShapeOp(OpKernelConstruction* ctx) : XlaOpKernel(ctx) {
    OP_REQUIRES_OK(ctx, ctx->GetAttr("out_type", &out_dtype_));
  }

  void Compile(XlaOpKernelContext* ctx) override {
    const TensorShape input_shape = ctx->InputShape(0);
    std::vector<xla::XlaOp> operands;
    const int rank = input_shape.dims();
    if (rank != 0) {
      for (int64_t i = 0; i < rank; ++i) {
        operands.push_back(xla::Broadcast(
            xla::ConvertElementType(xla::GetDimensionSize(ctx->Input(0), i),
                                    ctx->output_xla_type(0)),
            {1}));
      }

      ctx->SetOutput(0, xla::ConcatInDim(ctx->builder(), operands, 0));
    } else {
      // Rank 0 won't have dynamic size dimension, use constant output.
      Tensor shape_constant(out_dtype_, TensorShape({input_shape.dims()}));
      OP_REQUIRES_OK(ctx, TensorShapeToConstant(input_shape, &shape_constant));
      ctx->SetConstantOutput(0, shape_constant);
    }
  }

 private:
  DataType out_dtype_;
};

REGISTER_XLA_OP(Name("Shape").CompilationOnly().IsMetadataOp(), ShapeOp);

class XlaSetBoundOp : public XlaOpKernel {
 public:
  explicit XlaSetBoundOp(OpKernelConstruction* context)
      : XlaOpKernel(context) {}

  void Compile(XlaOpKernelContext* ctx) override {
    const TensorShape input_shape = ctx->InputShape("input");
    const TensorShape bound_shape = ctx->InputShape("bound");

    OP_REQUIRES(
        ctx,
        ctx->InputType("bound") == DT_INT32 &&
            ctx->InputType("input") == DT_INT32,
        errors::InvalidArgument(
            "XlaSetBound can only set bound for int32 scalar value: got",
            input_shape.DebugString()));

    OP_REQUIRES(
        ctx, input_shape.dims() == 0,
        errors::InvalidArgument("XlaSetBound should only be used to set a "
                                "bound to the an int32 scalar value: got",
                                input_shape.DebugString()));

    OP_REQUIRES(
        ctx, bound_shape.dims() == 0,
        errors::InvalidArgument("XlaSetBound should only be used to set a "
                                "bound to the an int32 scalar value: got",
                                bound_shape.DebugString()));
    int64_t bound;
    OP_REQUIRES_OK(ctx, ctx->ConstantInputAsIntScalar("bound", &bound));
    xla::Literal bound_literal = xla::LiteralUtil::CreateR0<int32>(bound);
    xla::XlaOp result = xla::CustomCall(
        ctx->builder(), "SetBound", {ctx->Input("input")},
        ctx->InputXlaShape("input").value(), "", false, {}, &bound_literal);
    ctx->SetOutput(0, result);
  }
};

REGISTER_XLA_OP(Name("XlaSetBound").CompileTimeConstantInput("bound"),
                XlaSetBoundOp);

class XlaSetDynamicDimensionSizeOp : public XlaOpKernel {
 public:
  explicit XlaSetDynamicDimensionSizeOp(OpKernelConstruction* context)
      : XlaOpKernel(context) {}

  void Compile(XlaOpKernelContext* ctx) override {
    const TensorShape dim_index_shape = ctx->InputShape("dim_index");
    const TensorShape size_shape = ctx->InputShape("size");

    OP_REQUIRES(ctx,
                ctx->InputType("dim_index") == DT_INT32 &&
                    ctx->InputType("size") == DT_INT32,
                errors::InvalidArgument("dim_index and size has to be int32 for"
                                        "XlaSetDynamicDimensionSizeOp"));

    OP_REQUIRES(
        ctx, dim_index_shape.dims() == 0 && size_shape.dims() == 0,
        errors::InvalidArgument("XlaSetDynamicDimensionSizeOp's dim_index and "
                                "size has to be int32 scalar value"));
    int64_t dim_index;
    OP_REQUIRES_OK(ctx, ctx->ConstantInputAsIntScalar("dim_index", &dim_index));

    xla::XlaOp result =
        xla::SetDimensionSize(ctx->Input(0), ctx->Input("size"), dim_index);
    ctx->SetOutput(0, result);
  }
};

REGISTER_XLA_OP(
    Name("XlaSetDynamicDimensionSize").CompileTimeConstantInput("dim_index"),
    XlaSetDynamicDimensionSizeOp);

class XlaRemoveDynamicDimensionSizeOp : public XlaOpKernel {
 public:
  explicit XlaRemoveDynamicDimensionSizeOp(OpKernelConstruction* context)
      : XlaOpKernel(context) {}

  void Compile(XlaOpKernelContext* ctx) override {
    const TensorShape dim_index_shape = ctx->InputShape("dim_index");

    OP_REQUIRES(ctx, ctx->InputType("dim_index") == DT_INT32,
                errors::InvalidArgument("dim_index has to be int32 for"
                                        "XlaRemoveDynamicDimensionSizeOp"));

    OP_REQUIRES(
        ctx, dim_index_shape.dims() == 0,
        errors::InvalidArgument("XlaRemoveDynamicDimensionSizeOp's dim_index "
                                "has to be int32 scalar value"));
    int64_t dim_index;
    OP_REQUIRES_OK(ctx, ctx->ConstantInputAsIntScalar("dim_index", &dim_index));

    xla::XlaOp result = xla::RemoveDynamicDimension(ctx->Input(0), dim_index);
    ctx->SetOutput(0, result);
  }
};

REGISTER_XLA_OP(
    Name("XlaRemoveDynamicDimensionSize").CompileTimeConstantInput("dim_index"),
    XlaRemoveDynamicDimensionSizeOp);

class ShapeNOp : public XlaOpKernel {
 public:
  explicit ShapeNOp(OpKernelConstruction* ctx) : XlaOpKernel(ctx) {
    OP_REQUIRES_OK(ctx, ctx->GetAttr("out_type", &out_dtype_));
  }

  void Compile(XlaOpKernelContext* ctx) override {
    for (int i = 0; i < ctx->num_inputs(); ++i) {
      const TensorShape input_shape = ctx->InputShape(i);
      std::vector<xla::XlaOp> operands;

      const int rank = input_shape.dims();
      if (rank != 0) {
        // Each dimension can be dynamic, so use GetDimensionSize to get the
        // runtime dimension.
        for (int64_t dim = 0; dim < rank; ++dim) {
          operands.push_back(xla::Broadcast(
              xla::ConvertElementType(xla::GetDimensionSize(ctx->Input(i), dim),
                                      ctx->output_xla_type(i)),
              {1}));
        }

        ctx->SetOutput(i, xla::ConcatInDim(ctx->builder(), operands, 0));
      } else {
        // Rank 0 won't have dynamic size dimension, use constant output.
        Tensor shape_constant(out_dtype_, TensorShape({input_shape.dims()}));
        OP_REQUIRES_OK(ctx,
                       TensorShapeToConstant(input_shape, &shape_constant));
        ctx->SetConstantOutput(i, shape_constant);
      }
    }
  }

  bool IsExpensive() override { return false; }

 private:
  DataType out_dtype_;
};
REGISTER_XLA_OP(Name("ShapeN").CompilationOnly().IsMetadataOp(), ShapeNOp);

class RankOp : public XlaOpKernel {
 public:
  explicit RankOp(OpKernelConstruction* ctx) : XlaOpKernel(ctx) {}

  void Compile(XlaOpKernelContext* ctx) override {
    const TensorShape input_shape = ctx->InputShape(0);
    const int rank = input_shape.dims();
    Tensor rank_constant(DT_INT32, TensorShape({}));
    rank_constant.scalar<int32_t>()() = rank;

    ctx->SetConstantOutput(0, rank_constant);
  }
};

REGISTER_XLA_OP(Name("Rank").CompilationOnly().IsMetadataOp(), RankOp);

class SizeOp : public XlaOpKernel {
 public:
  explicit SizeOp(OpKernelConstruction* ctx) : XlaOpKernel(ctx) {}

  void Compile(XlaOpKernelContext* ctx) override {
    const TensorShape input_shape = ctx->InputShape(0);
    xla::XlaBuilder* builder = ctx->builder();
    auto size = xla::One(builder, ctx->output_xla_type(0));

    const int rank = input_shape.dims();
    for (int64_t dim = 0; dim < rank; ++dim) {
      OP_REQUIRES(
          ctx,
          FastBoundsCheck(input_shape.dim_size(dim),
                          std::numeric_limits<int32_t>::max()),
          absl::InvalidArgumentError(absl::StrCat(
              "Size Op: XLA supported tensors must have <= int32max elements "
              "on all dimensions, found ",
              input_shape.dim_size(dim), " elements on dimension ", dim)));

      size = xla::Mul(size, xla::ConvertElementType(
                                xla::GetDimensionSize(ctx->Input(0), dim),
                                ctx->output_xla_type(0)));
    }

    ctx->SetOutput(0, size);
  }
};

REGISTER_XLA_OP(Name("Size").CompilationOnly().IsMetadataOp(), SizeOp);

class ExpandDimsOp : public XlaOpKernel {
 public:
  explicit ExpandDimsOp(OpKernelConstruction* ctx) : XlaOpKernel(ctx) {}

  void Compile(XlaOpKernelContext* ctx) override {
    const TensorShape input_shape = ctx->InputShape("input");
    const TensorShape dim_shape = ctx->InputShape("dim");

    std::vector<int64_t> dims;
    OP_REQUIRES_OK(ctx, ctx->ConstantInputReshapedToIntVector("dim", &dims));
    OP_REQUIRES(ctx, dims.size() == 1,
                errors::InvalidArgument(absl::StrCat(
                    "dim input to ExpandDims must be a scalar; got ",
                    dim_shape.DebugString())));
    int dim = dims[0];

    OP_REQUIRES(ctx,
                (dim >= -1 - input_shape.dims() && dim <= input_shape.dims()),
                errors::InvalidArgument("Tried to expand dim index ", dim,
                                        " for tensor with ", input_shape.dims(),
                                        " dimensions."));

    auto existing_dims = input_shape.dim_sizes();
    // Safe - # elements in tensor dims bounded.
    const int existing_dims_size = static_cast<int>(existing_dims.size());
    std::vector<int64_t> new_shape(existing_dims_size);
    for (size_t i = 0; i < new_shape.size(); ++i) {
      new_shape[i] = existing_dims[i];
    }

    // We emulate numpy's interpretation of the dim axis when
    // -input.dims() >= dim <= input.dims().
    if (dim < 0) {
      dim += existing_dims.size() + 1;
    }

    // Clamp to the end if needed.
    dim = std::min<int32_t>(dim, existing_dims_size);
    new_shape.emplace(new_shape.begin() + dim, 1);

    ctx->SetOutput(0, xla::Reshape(ctx->Input("input"), new_shape));
  }
};
REGISTER_XLA_OP(Name("ExpandDims").CompileTimeConstantInput("dim"),
                ExpandDimsOp);

class SqueezeOp : public XlaOpKernel {
 public:
  explicit SqueezeOp(OpKernelConstruction* ctx) : XlaOpKernel(ctx) {
    std::vector<int32_t> squeeze_dims;
    OP_REQUIRES_OK(ctx, ctx->GetAttr("squeeze_dims", &squeeze_dims));
    squeeze_dims_.insert(squeeze_dims.begin(), squeeze_dims.end());
  }

  void Compile(XlaOpKernelContext* ctx) override {
    absl::StatusOr<xla::Shape> input_shape =
        ctx->builder()->GetShape(ctx->Input(0));
    OP_REQUIRES_OK(ctx, input_shape.status());
    xla::Shape shape = input_shape.value();
    int64_t rank = shape.dimensions_size();

    absl::flat_hash_set<int32_t> wrapped_squeeze_dims;
    wrapped_squeeze_dims.reserve(squeeze_dims_.size());
    std::vector<int64_t> new_shape;
    // Validate squeeze dims against the input.
    for (int32_t dim : squeeze_dims_) {
      OP_REQUIRES(
          ctx, (dim >= -rank && dim < rank),
          errors::InvalidArgument("Tried to squeeze dim index ", dim,
                                  " for tensor with ", rank, " dimensions."));
      // If dim is < 0, we wrap around (-1 means the last element).
      if (dim < 0) {
        dim = rank + dim;
      }

      wrapped_squeeze_dims.insert(dim);
    }

    for (int i = 0; i < rank; ++i) {
      auto existing_dim = shape.dimensions(i);

      // If squeeze_set is non-empty, only squeeze those dimensions.
      if (!wrapped_squeeze_dims.empty()) {
        if (wrapped_squeeze_dims.count(i) > 0) {
          OP_REQUIRES(ctx, existing_dim == 1,
                      errors::InvalidArgument(
                          "Tried to explicitly squeeze dimension ", i,
                          " but dimension was not 1: ", existing_dim));
        } else {
          // This dimension is not being squeezed.
          new_shape.push_back(existing_dim);
        }
      } else {
        OP_REQUIRES(
            ctx, !shape.is_dynamic_dimension(i),
            errors::InvalidArgument("Squeeze op does not support bounded "
                                    "dynamic dimensions. Input shape: ",
                                    shape.DebugString()));
        // Copy over all non-1-length dimensions.
        if (existing_dim != 1) {
          new_shape.push_back(existing_dim);
        }
      }
    }

    ctx->SetOutput(0, xla::Reshape(ctx->Input(0), new_shape));
  }

 private:
  absl::flat_hash_set<int32_t> squeeze_dims_;
};

REGISTER_XLA_OP(Name("Squeeze"), SqueezeOp);

class ZerosLikeOp : public XlaOpKernel {
 public:
  explicit ZerosLikeOp(OpKernelConstruction* ctx) : XlaOpKernel(ctx) {}

  void Compile(XlaOpKernelContext* ctx) override {
    if (IsTensorListInput(ctx, 0)) {
      // Input is a TensorList.

      // Check the TensorList input is initialized.
      xla::XlaOp list = ctx->Input(0);
      bool is_initialized;
      OP_REQUIRES_OK(ctx, IsTensorListInitialized(list, &is_initialized));
      OP_REQUIRES(
          ctx, is_initialized,
          errors::InvalidArgument(
              "TensorList input for ZerosLike op is an uninitialized list"));

      auto list_shape_or = ctx->builder()->GetShape(list);
      OP_REQUIRES_OK(ctx, list_shape_or.status());
      const xla::Shape& list_shape = list_shape_or.value();
      std::vector<std::vector<xla::XlaOp>> list_dynamic_dims;
      list_dynamic_dims.reserve(list_shape.tuple_shapes_size() - 1);
      for (int i = 0; i < list_shape.tuple_shapes_size() - 1; ++i) {
        // Set dynamic dimension size to 0 for initialization value.
        std::vector<xla::XlaOp> dynamic_dims;
        const xla::Shape& shape = list_shape.tuple_shapes(i);
        auto sub_element = xla::GetTupleElement(list, i);
        dynamic_dims.reserve(shape.dimensions().size());
        for (int64_t dim = 0; dim < shape.dimensions().size(); ++dim) {
          dynamic_dims.push_back(xla::GetDimensionSize(sub_element, dim));
        }
        list_dynamic_dims.push_back(dynamic_dims);
      }
      xla::XlaOp new_list;
      OP_REQUIRES_OK(
          ctx, CreateZerosTensorListWithShape(ctx->builder(), list_shape,
                                              list_dynamic_dims, &new_list));

      xla::XlaOp push_index;
      OP_REQUIRES_OK(ctx, GetTensorListPushIndex(list, &push_index));

      xla::XlaOp result;
      OP_REQUIRES_OK(ctx,
                     SetTensorListPushIndex(new_list, push_index, &result));
      ctx->SetTensorListOutput(0, result);
    } else {
      auto zero = XlaHelpers::Zero(ctx->builder(), input_type(0));
      xla::XlaOp input = ctx->Input(0);
      auto input_shape = ctx->InputXlaShape(0).value();
      auto result = xla::Broadcast(zero, input_shape.dimensions());

      // Setting up dynamic dimensions of the broadcast.
      for (int64_t i = 0; i < input_shape.dimensions().size(); ++i) {
        if (input_shape.is_dynamic_dimension(i)) {
          xla::XlaOp input_dynamic_dim = xla::GetDimensionSize(input, i);
          result = xla::SetDimensionSize(result, input_dynamic_dim, i);
        }
      }

      ctx->SetOutput(0, result);
    }
  }
};

REGISTER_XLA_OP(Name("ZerosLike").AllowVariantTypes(), ZerosLikeOp);

class OnesLikeOp : public XlaOpKernel {
 public:
  explicit OnesLikeOp(OpKernelConstruction* ctx) : XlaOpKernel(ctx) {}

  void Compile(XlaOpKernelContext* ctx) override {
    const TensorShape input_shape = ctx->InputShape(0);

    auto one = XlaHelpers::One(ctx->builder(), input_type(0));
    ctx->SetOutput(0, xla::Broadcast(one, input_shape.dim_sizes()));
  }
};

REGISTER_XLA_OP(Name("OnesLike"), OnesLikeOp);

}  // namespace
}  // namespace tensorflow
