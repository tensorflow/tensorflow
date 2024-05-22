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

#include <cstddef>
#include <vector>

#include "absl/types/span.h"
#include "tensorflow/compiler/tf2xla/type_util.h"
#include "tensorflow/compiler/tf2xla/xla_helpers.h"
#include "tensorflow/compiler/tf2xla/xla_op_kernel.h"
#include "tensorflow/compiler/tf2xla/xla_op_registry.h"
#include "xla/client/lib/constants.h"
#include "xla/client/lib/math.h"
#include "xla/client/xla_builder.h"
#include "xla/client/xla_computation.h"
#include "xla/shape.h"
#include "xla/shape_util.h"
#include "xla/xla_data.pb.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/op_requires.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/framework/types.pb.h"
#include "tensorflow/core/platform/errors.h"
#include "tensorflow/core/platform/types.h"

namespace tensorflow {
namespace {

enum QuantizerRoundMode {
  // Round half up: if the fraction of y is exactly 0.5, then
  // round(y) = y + 0.5
  // E.g., -5.5 gets rounded to -5, -5.4 goes to -5,
  // 5.4 goes to 5, and 5.5 goes to 6.
  ROUND_HALF_UP,
  // Round half to even: if the fraction of y is exactly 0.5, then round(y) is
  // the nearest even integer to y.
  // E.g., 23.5 gets rounded to 24, 24.5 gets rounded to 24, while -23.5 becomes
  // -24, and -24.5 gets rounded to 24.
  ROUND_HALF_TO_EVEN,
};

class QuantizeAndDequantizeOp : public XlaOpKernel {
 public:
  explicit QuantizeAndDequantizeOp(OpKernelConstruction* ctx)
      : XlaOpKernel(ctx) {
    OP_REQUIRES_OK(ctx, ctx->GetAttr("signed_input", &signed_input_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("range_given", &range_given_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("narrow_range", &narrow_range_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("axis", &axis_));
    round_mode_ = ROUND_HALF_TO_EVEN;
  }

  void Compile(XlaOpKernelContext* ctx) override {
    xla::XlaOp input = ctx->Input(0);
    const DataType data_type = ctx->input_type(0);

    xla::PrimitiveType xla_type;
    OP_REQUIRES_OK(ctx, DataTypeToPrimitiveType(data_type, &xla_type));

    xla::XlaBuilder* b = ctx->builder();

    // The implementation follows
    // tensorflow/core/kernels/quantize_and_dequantize_op.h closely.
    xla::XlaOp min_range, max_range;
    if (range_given_) {
      min_range = ctx->Input(1);
      max_range = ctx->Input(2);
    } else {
      const xla::XlaComputation* fmax = ctx->GetOrCreateMax(data_type);
      const xla::XlaComputation* fmin = ctx->GetOrCreateMin(data_type);
      if (axis_ == -1) {
        min_range = ReduceAll(input, xla::MaxValue(b, xla_type), *fmin);
        max_range = ReduceAll(input, xla::MinValue(b, xla_type), *fmax);
      } else {
        std::vector<int64_t> dimensions_to_reduce;
        TensorShape input_shape = ctx->InputShape(0);
        int64_t input_rank = input_shape.dims();
        OP_REQUIRES(ctx, input_rank >= 1,
                    errors::Unimplemented("QuantizeAndDequantizeOp with axis "
                                          "!= -1 requires minimum rank 1"));
        OP_REQUIRES(
            ctx, axis_ >= 0 && axis_ < input_rank,
            errors::Unimplemented("QuantizeAndDequantizeOp with invalid axis"));
        dimensions_to_reduce.reserve(input_rank - 1);
        for (int64_t i = 0; i < input_rank; ++i) {
          if (i != axis_) {
            dimensions_to_reduce.push_back(i);
          }
        }
        min_range = Reduce(input, xla::MaxValue(b, xla_type), *fmin,
                           dimensions_to_reduce);
        max_range = Reduce(input, xla::MinValue(b, xla_type), *fmax,
                           dimensions_to_reduce);
      }
    }

    xla::XlaOp num_bits;
    if (num_bits_ < 0) {
      OP_REQUIRES(
          ctx, ctx->num_inputs() == 4,
          errors::Internal("Expected 4 inputs to QuantizeAndDequantize"));
      num_bits = ctx->Input(3);
    } else {
      num_bits = xla::ConstantR0<int32>(b, num_bits_);
    }

    const xla::XlaOp zero = XlaHelpers::Zero(b, data_type);
    const xla::XlaOp one = XlaHelpers::One(b, data_type);
    const xla::XlaOp two = XlaHelpers::FloatLiteral(b, data_type, 2.0);
    const xla::XlaOp half = XlaHelpers::FloatLiteral(b, data_type, 0.5);

    // Calculate the range for the simulated integer quantization:
    // e.g. [-128,127] for signed = true, num_bits = 8,
    // or [0, 255] for signed = false, num_bits = 8.
    // We do this in floating point for hardware that does not have 64-bit
    // integer support.
    xla::XlaOp min_quantized, max_quantized;
    if (signed_input_) {
      if (narrow_range_) {
        min_quantized =
            -Pow(two, ConvertElementType(
                          num_bits - xla::ConstantR0<int32>(b, 1), xla_type)) +
            one;
      } else {
        min_quantized =
            -Pow(two, ConvertElementType(
                          num_bits - xla::ConstantR0<int32>(b, 1), xla_type));
      }
      max_quantized =
          Pow(two, ConvertElementType(num_bits - xla::ConstantR0<int32>(b, 1),
                                      xla_type)) -
          one;
    } else {
      min_quantized = zero;
      max_quantized = Pow(two, ConvertElementType(num_bits, xla_type)) - one;
    }

    // Determine the maximum scaling factor that would scale
    // [min_range, max_range] to not exceed [min_quantized, max_quantized],
    // while keeping 0 unchanged.
    xla::XlaOp scale_from_min_side =
        Select(Gt(min_quantized * min_range, zero), min_quantized / min_range,
               xla::MaxFiniteValue(b, xla_type));
    xla::XlaOp scale_from_max_side =
        Select(Gt(max_quantized * max_range, zero), max_quantized / max_range,
               xla::MaxFiniteValue(b, xla_type));

    // Note: Avoids changing the side of the range that determines scale.
    xla::XlaOp cond = Lt(scale_from_min_side, scale_from_max_side);
    xla::XlaOp scale = Select(cond, scale_from_min_side, scale_from_max_side);
    xla::XlaOp inverse_scale =
        Select(cond, min_range / min_quantized, max_range / max_quantized);
    min_range = Select(cond, min_range, min_quantized * inverse_scale);
    max_range = Select(cond, max_quantized * inverse_scale, max_range);

    // The instruction min_range has the shape of the axis, which is also the
    // shape for max_range, scale and inverse_scale.
    xla::Shape axis_shape = b->GetShape(min_range).value();
    // The XLA client library can handle implicit broadcast from scalar. Add
    // explicit broadcast if the axis has a non-scalar shape.
    if (!xla::ShapeUtil::IsScalar(axis_shape)) {
      xla::Shape input_shape = b->GetShape(input).value();
      absl::Span<const int64_t> input_dimensions = input_shape.dimensions();
      auto convert_to_input_shape = [&](const xla::XlaOp op) {
        return xla::BroadcastInDim(op, input_dimensions, {axis_});
      };
      min_range = convert_to_input_shape(min_range);
      max_range = convert_to_input_shape(max_range);
      scale = convert_to_input_shape(scale);
      inverse_scale = convert_to_input_shape(inverse_scale);
    }

    if (range_given_) {
      // Note: The clamping here is to avoid overflow in the quantized type.
      // The semantics of the op does not guarantee to clamp to the specified
      // min_range and max_range - because we may have changed either min_range
      // or max_range.
      // No need to clamp to min_range and max_range if range_given_ == false as
      // in that case they were measured from the tensor.
      input = Clamp(min_range, input, max_range);
    }
    xla::XlaOp result;
    switch (round_mode_) {
      case ROUND_HALF_TO_EVEN: {
        result = xla::RoundToEven(input * scale) * inverse_scale;
        break;
      }
      case ROUND_HALF_UP: {
        result = Floor(input * scale + half) * inverse_scale;
        break;
      }
    }
    ctx->SetOutput(0, result);
  }

 protected:
  int64_t num_bits_ = -1;
  int axis_;
  bool signed_input_;
  bool range_given_;
  bool narrow_range_;
  QuantizerRoundMode round_mode_;
};

class QuantizeAndDequantizeV2Op : public QuantizeAndDequantizeOp {
 public:
  explicit QuantizeAndDequantizeV2Op(OpKernelConstruction* ctx)
      : QuantizeAndDequantizeOp(ctx) {
    OP_REQUIRES_OK(ctx, ctx->GetAttr("num_bits", &num_bits_));
    OP_REQUIRES(ctx, num_bits_ > 0 && num_bits_ < (signed_input_ ? 62 : 63),
                errors::InvalidArgument("num_bits is out of range: ", num_bits_,
                                        " with signed_input_ ", signed_input_));
    string round_mode_string;
    OP_REQUIRES_OK(ctx, ctx->GetAttr("round_mode", &round_mode_string));
    OP_REQUIRES(
        ctx,
        (round_mode_string == "HALF_UP" || round_mode_string == "HALF_TO_EVEN"),
        errors::InvalidArgument("Round mode string must be "
                                "'HALF_UP' or "
                                "'HALF_TO_EVEN', is '" +
                                round_mode_string + "'"));
    if (round_mode_string == "HALF_UP") {
      round_mode_ = ROUND_HALF_UP;
    } else if (round_mode_string == "HALF_TO_EVEN") {
      round_mode_ = ROUND_HALF_TO_EVEN;
    }
  }
};

REGISTER_XLA_OP(Name("QuantizeAndDequantizeV2"), QuantizeAndDequantizeV2Op);
REGISTER_XLA_OP(Name("QuantizeAndDequantizeV3"), QuantizeAndDequantizeOp);
REGISTER_XLA_OP(Name("QuantizeAndDequantizeV4"), QuantizeAndDequantizeV2Op);

}  // namespace
}  // namespace tensorflow
