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

#include "tensorflow/compiler/tf2xla/type_util.h"
#include "tensorflow/compiler/tf2xla/xla_helpers.h"
#include "tensorflow/compiler/tf2xla/xla_op_kernel.h"
#include "tensorflow/compiler/tf2xla/xla_op_registry.h"
#include "tensorflow/compiler/xla/client/lib/arithmetic.h"
#include "tensorflow/compiler/xla/client/lib/constants.h"
#include "tensorflow/compiler/xla/client/xla_builder.h"
#include "tensorflow/compiler/xla/client/xla_computation.h"
#include "tensorflow/core/platform/macros.h"

namespace tensorflow {
namespace {

class QuantizeAndDequantizeOp : public XlaOpKernel {
 public:
  explicit QuantizeAndDequantizeOp(OpKernelConstruction* ctx)
      : XlaOpKernel(ctx) {
    OP_REQUIRES_OK(ctx, ctx->GetAttr("signed_input", &signed_input_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("range_given", &range_given_));
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
      min_range = ReduceAll(input, xla::MaxValue(b, xla_type), *fmin);
      max_range = ReduceAll(input, xla::MinValue(b, xla_type), *fmax);
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
      min_quantized =
          -Pow(two, ConvertElementType(num_bits - xla::ConstantR0<int32>(b, 1),
                                       xla_type));
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

    if (range_given_) {
      // Note: The clamping here is to avoid overflow in the quantized type.
      // The semantics of the op does not guarantee to clamp to the specified
      // min_range and max_range - because we may have changed either min_range
      // or max_range.
      // No need to clamp to min_range and max_range if range_given_ == false as
      // in that case they were measured from the tensor.
      input = Clamp(min_range, input, max_range);
    }
    xla::XlaOp result =
        Floor((input - min_range) * scale + half) * inverse_scale + min_range;
    ctx->SetOutput(0, result);
  }

 protected:
  int64 num_bits_ = -1;
  bool signed_input_;
  bool range_given_;
};

class QuantizeAndDequantizeV2Op : public QuantizeAndDequantizeOp {
 public:
  explicit QuantizeAndDequantizeV2Op(OpKernelConstruction* ctx)
      : QuantizeAndDequantizeOp(ctx) {
    OP_REQUIRES_OK(ctx, ctx->GetAttr("num_bits", &num_bits_));
    OP_REQUIRES(ctx, num_bits_ > 0 && num_bits_ < (signed_input_ ? 62 : 63),
                errors::InvalidArgument("num_bits is out of range: ", num_bits_,
                                        " with signed_input_ ", signed_input_));
  }
};

REGISTER_XLA_OP(Name("QuantizeAndDequantizeV2"), QuantizeAndDequantizeV2Op);
REGISTER_XLA_OP(Name("QuantizeAndDequantizeV3"), QuantizeAndDequantizeOp);

}  // namespace
}  // namespace tensorflow
