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
#include "tensorflow/compiler/tf2xla/lib/broadcast.h"
#include "tensorflow/compiler/tf2xla/lib/util.h"
#include "tensorflow/compiler/tf2xla/shape_util.h"
#include "tensorflow/compiler/tf2xla/type_util.h"
#include "tensorflow/compiler/tf2xla/xla_helpers.h"
#include "tensorflow/compiler/tf2xla/xla_op_kernel.h"
#include "tensorflow/compiler/tf2xla/xla_op_registry.h"
#include "tensorflow/compiler/xla/client/lib/arithmetic.h"
#include "tensorflow/compiler/xla/client/lib/constants.h"
#include "tensorflow/compiler/xla/client/xla_builder.h"
#include "tensorflow/compiler/xla/primitive_util.h"
#include "tensorflow/compiler/xla/xla_data.pb.h"
#include "tensorflow/core/framework/kernel_def_builder.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/lib/core/errors.h"

namespace tensorflow {
namespace {

class CastOp : public XlaOpKernel {
 public:
  explicit CastOp(OpKernelConstruction* ctx) : XlaOpKernel(ctx) {
    OP_REQUIRES_OK(ctx, ctx->GetAttr("SrcT", &src_dtype_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("DstT", &dst_dtype_));
    OP_REQUIRES_OK(ctx, DataTypeToPrimitiveType(src_dtype_, &src_type_));
    OP_REQUIRES_OK(ctx, DataTypeToPrimitiveType(dst_dtype_, &dst_type_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("Truncate", &use_truncation_));
  }

  void Compile(XlaOpKernelContext* ctx) override {
    xla::XlaBuilder* builder = ctx->builder();
    xla::XlaOp input = ctx->Input(0);
    xla::XlaOp output;

    if (src_dtype_ == dst_dtype_) {
      output = input;
    } else if (dst_dtype_ == DT_BOOL) {
      output = xla::Ne(input, XlaHelpers::Zero(builder, src_dtype_));
    } else if (xla::primitive_util::IsComplexType(src_type_) &&
               !xla::primitive_util::IsComplexType(dst_type_)) {
      // As in cast_op.h, we replicate the numpy behavior of truncating the
      // imaginary part.
      output = xla::ConvertElementType(xla::Real(input), dst_type_);
    } else {
      if (use_truncation_) {
        OP_REQUIRES(
            ctx,
            xla::primitive_util::IsFloatingPointType(src_type_) &&
                xla::primitive_util::IsFloatingPointType(dst_type_),
            errors::Unimplemented("Truncate attribute is only "
                                  "implemented for floating point datatypes."));
        int mantissa_difference =
            xla::primitive_util::SignificandWidth(src_type_) -
            xla::primitive_util::SignificandWidth(dst_type_);
        OP_REQUIRES(ctx, mantissa_difference > 0,
                    errors::Unimplemented(
                        "Truncate attribute is only implemented in cases where "
                        "dst datatype "
                        "has fewer mantissa bits than the src datatype"));
        int src_bitwidth = xla::primitive_util::BitWidth(src_type_);

        // Bitcast to same-width integer, mask off the LSBs, bitcast back to the
        // source datatype.
        int64 mask = ~((1L << mantissa_difference) - 1);
        xla::PrimitiveType same_width_int =
            xla::primitive_util::UnsignedIntegralTypeForBitWidth(src_bitwidth);
        OP_REQUIRES(ctx, same_width_int != xla::PRIMITIVE_TYPE_INVALID,
                    errors::Unimplemented("Unexpected type bitwidth"));
        input = xla::BitcastConvertType(
            xla::And(
                xla::BitcastConvertType(input, same_width_int),
                ::tensorflow::IntegerLiteral(builder, same_width_int, mask)),
            src_type_);
      }
      output = xla::ConvertElementType(input, dst_type_);
    }

    ctx->SetOutput(0, output);
  }

 protected:
  DataType src_dtype_, dst_dtype_;
  xla::PrimitiveType src_type_, dst_type_;
  bool use_truncation_;

  TF_DISALLOW_COPY_AND_ASSIGN(CastOp);
};

REGISTER_XLA_OP(Name("Cast"), CastOp);

class BitcastOp : public XlaOpKernel {
 public:
  explicit BitcastOp(OpKernelConstruction* ctx) : XlaOpKernel(ctx) {
    OP_REQUIRES_OK(ctx, ctx->GetAttr("T", &src_dtype_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("type", &dst_dtype_));
    OP_REQUIRES_OK(ctx, DataTypeToPrimitiveType(src_dtype_, &src_type_));
    OP_REQUIRES_OK(ctx, DataTypeToPrimitiveType(dst_dtype_, &dst_type_));
  }

  void Compile(XlaOpKernelContext* ctx) override {
    xla::XlaOp input = ctx->Input(0);
    xla::XlaOp output;

    if (src_dtype_ == dst_dtype_) {
      output = input;
      ctx->SetOutput(0, output);
      return;
    }
    // Error out if the bitcast has a complex source or destination type and
    // the bitcast is not trivial.
    OP_REQUIRES(ctx,
                !xla::primitive_util::IsComplexType(src_type_) &&
                    !xla::primitive_util::IsComplexType(dst_type_),
                errors::Unimplemented("Complex types not supported."));
    auto input_bit_width = xla::primitive_util::BitWidth(src_type_);
    auto output_bit_width = xla::primitive_util::BitWidth(dst_type_);

    auto input_logical_type =
        xla::primitive_util::UnsignedIntegralTypeForBitWidth(input_bit_width);
    auto output_logical_type =
        xla::primitive_util::UnsignedIntegralTypeForBitWidth(output_bit_width);

    OP_REQUIRES(ctx,

                output_bit_width % input_bit_width == 0 ||
                    input_bit_width % output_bit_width == 0,
                errors::InvalidArgument(
                    "Neither bit width is a multiple of the other."));

    // Modify the input as needed so we only need to bitcast to create the
    // output.
    if (input_bit_width > output_bit_width) {
      // Casting to a smaller bit width results in a new inner dimension.
      auto broadcasted_input_shape = ctx->InputShape(0);
      auto reshaped_input_shape = ctx->InputShape(0);
      broadcasted_input_shape.AddDim(input_bit_width / output_bit_width);
      reshaped_input_shape.AddDim(1);
      auto output_bit_width_mask = (int64(1) << output_bit_width) - 1;

      auto status_or_input =
          BroadcastTo(xla::Reshape(input, reshaped_input_shape.dim_sizes()),
                      broadcasted_input_shape.dim_sizes());
      OP_REQUIRES_OK(ctx, status_or_input.status());
      input = xla::BitcastConvertType(status_or_input.ConsumeValueOrDie(),
                                      input_logical_type);
      auto xla_input_shape_status = ctx->builder()->GetShape(input);
      OP_REQUIRES_OK(ctx, xla_input_shape_status.status());
      auto xla_input_shape = xla_input_shape_status.ConsumeValueOrDie();

      auto iota = xla::Iota(ctx->builder(), xla_input_shape,
                            xla_input_shape.dimensions_size() - 1);
      xla::XlaOp iota_m =
          xla::Mul(xla::ScalarLike(input, output_bit_width), iota);
      input = xla::And(xla::ShiftRightLogical(input, iota_m),
                       xla::ScalarLike(input, output_bit_width_mask));
      input = xla::ConvertElementType(input, output_logical_type);
    } else if (input_bit_width < output_bit_width) {
      // Casting to a larger bit width results in removing the innermost
      // dimension.
      auto input_shape = ctx->InputShape(0);
      xla::Shape input_xla_shape =
          TensorShapeToXLAShape(dst_type_, input_shape);
      OP_REQUIRES(
          ctx,
          input_shape.dim_size(input_shape.dims() - 1) ==
              output_bit_width / input_bit_width,
          errors::InvalidArgument(
              "Inner dimension of operand should be removed after cast."));

      auto zero = XlaHelpers::Zero(ctx->builder(), dst_dtype_);
      input = xla::ConvertElementType(input, dst_type_);

      // Shift bits and OR them together to reduce the inner dimension.
      xla::XlaOp iota_m =
          xla::Mul(xla::ScalarLike(input, input_bit_width),
                   xla::Iota(ctx->builder(), input_xla_shape,
                             input_xla_shape.dimensions_size() - 1));
      input = xla::ShiftLeft(input, iota_m);
      input = xla::Reduce(input, zero,
                          CreateScalarOrComputation(dst_type_, ctx->builder()),
                          {input_xla_shape.dimensions_size() - 1});
    }

    output = xla::BitcastConvertType(input, dst_type_);
    ctx->SetOutput(0, output);
  }

 protected:
  DataType src_dtype_, dst_dtype_;
  xla::PrimitiveType src_type_, dst_type_;

  TF_DISALLOW_COPY_AND_ASSIGN(BitcastOp);
};

REGISTER_XLA_OP(Name("Bitcast"), BitcastOp);

}  // anonymous namespace
}  // namespace tensorflow
