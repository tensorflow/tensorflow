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

#include "tensorflow/compiler/tf2xla/xla_helpers.h"
#include "tensorflow/compiler/tf2xla/xla_op_kernel.h"
#include "tensorflow/compiler/tf2xla/xla_op_registry.h"
#include "tensorflow/compiler/xla/client/lib/arithmetic.h"
#include "tensorflow/core/platform/macros.h"

namespace tensorflow {
namespace {

class QuantizeAndDequantizeOp : public XlaOpKernel {
 public:
  explicit QuantizeAndDequantizeOp(OpKernelConstruction* ctx)
      : XlaOpKernel(ctx) {
    OP_REQUIRES_OK(ctx, ctx->GetAttr("signed_input", &signed_input_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("range_given", &range_given_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("num_bits", &num_bits_));
    OP_REQUIRES(ctx, num_bits_ > 0 && num_bits_ < (signed_input_ ? 62 : 63),
                errors::InvalidArgument("num_bits is out of range: ", num_bits_,
                                        " with signed_input_ ", signed_input_));
  }

  void Compile(XlaOpKernelContext* ctx) override {
    xla::ComputationDataHandle input = ctx->Input(0);
    const DataType data_type = ctx->input_type(0);

    // Comments taken from semantics description at
    // https://www.tensorflow.org/versions/r1.0/api_docs/cc/class/tensorflow/ops/quantize-and-dequantize
    //
    // ... we find m such that
    //
    // m = max(abs(input_min), abs(input_max)) if range_given is true,
    // m = max(abs(min_elem(input)),
    //         abs(max_elem(input))) otherwise.
    xla::ComputationBuilder* b = ctx->builder();
    xla::ComputationDataHandle input_min, input_max;
    if (range_given_) {
      double input_min_value, input_max_value;
      OP_REQUIRES_OK(ctx, ctx->ConstantInputAsFloatScalar(1, &input_min_value));
      OP_REQUIRES_OK(ctx, ctx->ConstantInputAsFloatScalar(2, &input_max_value));
      input_min = XlaHelpers::FloatLiteral(b, data_type, input_min_value);
      input_max = XlaHelpers::FloatLiteral(b, data_type, input_max_value);
    } else {
      const xla::Computation* fmax = ctx->GetOrCreateMax(data_type);
      const xla::Computation* fmin = ctx->GetOrCreateMin(data_type);
      input_min =
          b->ReduceAll(input, XlaHelpers::MaxValue(b, data_type), *fmin);
      input_max =
          b->ReduceAll(input, XlaHelpers::MinValue(b, data_type), *fmax);
    }
    xla::ComputationDataHandle m = b->Max(b->Abs(input_min), b->Abs(input_max));

    // Next, we choose our fixed-point quantization buckets, [min_fixed,
    // max_fixed]. If signed_input is true, this is
    //
    // [min_fixed, max_fixed ] = [-((1 << (num_bits - 1)) - 1),
    //                             (1 << (num_bits - 1)) - 1].
    //
    // Otherwise, if signed_input is false, the fixed-point range is
    //
    // [min_fixed, max_fixed] = [0, (1 << num_bits) - 1].
    int64 min_fixed, max_fixed;
    if (signed_input_) {
      min_fixed = -((1LL << (num_bits_ - 1)) - 1);
      max_fixed = (1LL << (num_bits_ - 1)) - 1;
    } else {
      min_fixed = 0;
      max_fixed = (1LL << num_bits_) - 1;
    }

    // From this we compute our scaling factor, s:
    //
    // s = (max_fixed - min_fixed) / (2 * m).
    xla::ComputationDataHandle s =
        b->Div(XlaHelpers::FloatLiteral(b, data_type, max_fixed - min_fixed),
               b->Mul(XlaHelpers::FloatLiteral(b, data_type, 2.0), m));

    // Now we can quantize and dequantize the elements of our tensor. An element
    // e is transformed into e':
    //
    // e' = (e * s).round_to_nearest() / s.
    xla::ComputationDataHandle result = b->Div(b->Round(b->Mul(input, s)), s);

    ctx->SetOutput(0, result);
  }

  int64 num_bits_;
  bool signed_input_;
  bool range_given_;
};

REGISTER_XLA_OP(Name("QuantizeAndDequantizeV2"), QuantizeAndDequantizeOp);

}  // namespace
}  // namespace tensorflow
