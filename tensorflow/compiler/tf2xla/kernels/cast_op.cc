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
#include <vector>

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
      output = xla::ConvertElementType(input, dst_type_);
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

    const bool src_complex = xla::primitive_util::IsComplexType(src_type_);
    const bool dst_complex = xla::primitive_util::IsComplexType(dst_type_);

    if (src_complex && !dst_complex) {
      // A bitcast from a complex type reinterprets its contiguous storage,
      // which holds the real component followed by the imaginary component
      // (each component_bit_width bits). XLA's BitcastConvert cannot cross
      // the complex<->real boundary directly, so bitcast each component to
      // the destination type separately -- this keeps every BitcastConvert
      // call on a plain real-typed operand, matching the shape XLA's own
      // shape inference (and the Bitcast op's shape function) computes for
      // that type pair -- and concatenate the real result before the
      // imaginary result, since the real component occupies the lower
      // bytes. This is byte-for-byte equivalent to the eager Bitcast
      // kernel.
      const xla::PrimitiveType component =
          xla::primitive_util::ComplexComponentType(src_type_);
      const int64_t component_bit_width =
          xla::primitive_util::BitWidth(component);
      const int64_t dst_bit_width = xla::primitive_util::BitWidth(dst_type_);
      OP_REQUIRES(ctx,
                  dst_bit_width % component_bit_width == 0 ||
                      component_bit_width % dst_bit_width == 0,
                  absl::InvalidArgumentError(
                      "Neither bit width is a multiple of the other."));

      xla::XlaOp real_bits = xla::BitcastConvertType(xla::Real(input), dst_type_);
      xla::XlaOp imag_bits = xla::BitcastConvertType(xla::Imag(input), dst_type_);

      const TensorShape input_shape = ctx->InputShape(0);
      const int64_t rank = input_shape.dims();
      if (component_bit_width == dst_bit_width) {
        // BitcastConvertType does not introduce a new minor dimension when
        // the widths already match, so add one explicitly before
        // concatenating the real and imaginary halves.
        // dim_sizes() returns by value, so it must be materialized once: two
        // separate calls yield two distinct temporaries, and an iterator pair
        // taken across them does not denote a valid range.
        const auto dims = input_shape.dim_sizes();
        std::vector<int64_t> expanded_dims(dims.begin(), dims.end());
        expanded_dims.push_back(1);
        real_bits = xla::Reshape(real_bits, expanded_dims);
        imag_bits = xla::Reshape(imag_bits, expanded_dims);
      }
      // When component_bit_width > dst_bit_width, BitcastConvertType has
      // already appended a minor dimension of component_bit_width /
      // dst_bit_width to each of real_bits and imag_bits, so concatenating
      // along that existing dimension directly produces the same shape the
      // eager Bitcast kernel does, with no further reshape required.
      ctx->SetOutput(
          0, xla::ConcatInDim(ctx->builder(), {real_bits, imag_bits}, rank));
      return;
    }

    // Real->complex and complex->complex bitcasts are not yet supported;
    // XLA's BitcastConvert cannot cross the complex<->real boundary.
    OP_REQUIRES(ctx, !src_complex && !dst_complex,
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
