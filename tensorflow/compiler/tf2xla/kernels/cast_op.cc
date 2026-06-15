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
#include <cstdint>

#include "absl/log/check.h"
#include "absl/status/status.h"
#include "tensorflow/compiler/tf2xla/lib/broadcast.h"
#include "tensorflow/compiler/tf2xla/lib/util.h"
#include "tensorflow/compiler/tf2xla/shape_util.h"
#include "tensorflow/compiler/tf2xla/type_util.h"
#include "tensorflow/compiler/tf2xla/xla_helpers.h"
#include "tensorflow/compiler/tf2xla/xla_op_kernel.h"
#include "tensorflow/compiler/tf2xla/xla_op_registry.h"
#include "xla/hlo/builder/lib/arithmetic.h"
#include "xla/hlo/builder/lib/constants.h"
#include "xla/hlo/builder/xla_builder.h"
#include "xla/primitive_util.h"
#include "xla/xla_data.pb.h"
#include "tensorflow/core/framework/kernel_def_builder.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/types.pb.h"
#include "tensorflow/core/lib/core/errors.h"

namespace tensorflow {
namespace {

xla::XlaOp ConvertFloatingToUnsigned(xla::XlaOp input,
                                     xla::PrimitiveType src_type,
                                     xla::PrimitiveType dst_type) {
  if (!xla::primitive_util::IsFloatingPointType(src_type) ||
      !xla::primitive_util::IsUnsignedIntegralType(dst_type)) {
    return xla::ConvertElementType(input, dst_type);
  }

  // XLA convert saturates negative floats to unsigned integers at zero, while
  // TensorFlow Cast's CPU path wraps finite negative values in the destination
  // unsigned type. NaN and Inf casts to integral types are documented as
  // undefined, so keep them on the direct HLO convert path.
  int dst_width = xla::primitive_util::BitWidth(dst_type);
  int wider_width = dst_width == 64 ? 64 : dst_width * 2;
  xla::PrimitiveType signed_type =
      xla::primitive_util::SignedIntegralTypeForBitWidth(wider_width);
  xla::PrimitiveType unsigned_type =
      xla::primitive_util::UnsignedIntegralTypeForBitWidth(wider_width);
  CHECK_NE(signed_type, xla::PRIMITIVE_TYPE_INVALID);
  CHECK_NE(unsigned_type, xla::PRIMITIVE_TYPE_INVALID);

  xla::XlaOp direct = xla::ConvertElementType(input, dst_type);
  xla::XlaOp wrapped_negative = xla::ConvertElementType(
      xla::ConvertElementType(xla::ConvertElementType(input, signed_type),
                              unsigned_type),
      dst_type);
  xla::XlaOp finite_negative =
      xla::And(xla::Lt(input, xla::ZerosLike(input)), xla::IsFinite(input));
  return xla::Select(finite_negative, wrapped_negative, direct);
}

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
      output = ConvertFloatingToUnsigned(
          xla::Real(input), xla::primitive_util::ComplexComponentType(src_type_),
          dst_type_);
    } else {
      if (use_truncation_) {
        OP_REQUIRES(ctx,
                    xla::primitive_util::IsFloatingPointType(src_type_) &&
                        xla::primitive_util::IsFloatingPointType(dst_type_),
                    absl::UnimplementedError(
                        "Truncate attribute is only "
                        "implemented for floating point datatypes."));
        int mantissa_difference =
            xla::primitive_util::SignificandWidth(src_type_) -
            xla::primitive_util::SignificandWidth(dst_type_);
        OP_REQUIRES(ctx, mantissa_difference > 0,
                    absl::UnimplementedError(
                        "Truncate attribute is only implemented in cases where "
                        "dst datatype "
                        "has fewer mantissa bits than the src datatype"));
        int src_bitwidth = xla::primitive_util::BitWidth(src_type_);

        // Bitcast to same-width integer, mask off the LSBs, bitcast back to the
        // source datatype.
        int64_t mask = ~((1L << mantissa_difference) - 1);
        xla::PrimitiveType same_width_int =
            xla::primitive_util::UnsignedIntegralTypeForBitWidth(src_bitwidth);
        OP_REQUIRES(ctx, same_width_int != xla::PRIMITIVE_TYPE_INVALID,
                    absl::UnimplementedError("Unexpected type bitwidth"));
        input = xla::BitcastConvertType(
            xla::And(
                xla::BitcastConvertType(input, same_width_int),
                ::tensorflow::IntegerLiteral(builder, same_width_int, mask)),
            src_type_);
      }
      output = ConvertFloatingToUnsigned(input, src_type_, dst_type_);
    }

    ctx->SetOutput(0, output);
  }

 protected:
  DataType src_dtype_, dst_dtype_;
  xla::PrimitiveType src_type_, dst_type_;
  bool use_truncation_;

  CastOp(const CastOp&) = delete;
  void operator=(const CastOp&) = delete;
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
                absl::UnimplementedError("Complex types not supported."));
    auto input_bit_width = xla::primitive_util::BitWidth(src_type_);
    auto output_bit_width = xla::primitive_util::BitWidth(dst_type_);

    OP_REQUIRES(ctx,
                output_bit_width % input_bit_width == 0 ||
                    input_bit_width % output_bit_width == 0,
                absl::InvalidArgumentError(
                    "Neither bit width is a multiple of the other."));
    output = xla::BitcastConvertType(input, dst_type_);
    ctx->SetOutput(0, output);
  }

 protected:
  DataType src_dtype_, dst_dtype_;
  xla::PrimitiveType src_type_, dst_type_;

  BitcastOp(const BitcastOp&) = delete;
  void operator=(const BitcastOp&) = delete;
};

REGISTER_XLA_OP(Name("Bitcast"), BitcastOp);

}  // anonymous namespace
}  // namespace tensorflow
