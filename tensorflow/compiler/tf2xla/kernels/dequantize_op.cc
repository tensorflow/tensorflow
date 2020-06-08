/* Copyright 2019 The TensorFlow Authors. All Rights Reserved.

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

#include <array>

#include "tensorflow/compiler/tf2xla/xla_helpers.h"
#include "tensorflow/compiler/tf2xla/xla_op_kernel.h"
#include "tensorflow/compiler/tf2xla/xla_op_registry.h"
#include "tensorflow/compiler/xla/client/lib/constants.h"
#include "tensorflow/compiler/xla/client/lib/matrix.h"
#include "tensorflow/compiler/xla/client/xla_builder.h"
#include "tensorflow/core/framework/numeric_types.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/types.pb.h"

namespace tensorflow {
namespace {

// TODO(mingyao|ylc): Support 16bits and 32 bits.
constexpr std::array<DataType, 2> kQuantizedType = {{DT_QINT8, DT_QUINT8}};

template <typename T>
float get_fullrange() {
  return static_cast<float>(std::numeric_limits<T>::max()) -
         std::numeric_limits<T>::min();
}

class DequantizeOp : public XlaOpKernel {
 public:
  explicit DequantizeOp(OpKernelConstruction* ctx) : XlaOpKernel(ctx) {
    string mode_string;
    int axis;
    bool narrow_range;

    OP_REQUIRES_OK(ctx, ctx->GetAttr("mode", &mode_string));
    OP_REQUIRES(
        ctx, (mode_string == "MIN_COMBINED"),
        errors::InvalidArgument("Mode string must be 'MIN_COMBINED' is " +
                                mode_string + "'"));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("narrow_range", &narrow_range));
    OP_REQUIRES(ctx, narrow_range == false,
                errors::InvalidArgument("narrow_range must be false"));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("axis", &axis));
    OP_REQUIRES(ctx, axis == -1,
                errors::InvalidArgument("axis must be -1' is ", axis));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("dtype", &dtype_));
  }

  ~DequantizeOp() override = default;

  void Compile(XlaOpKernelContext* ctx) override {
    DataType input_type = ctx->input_type(0);

    double minrange, maxrange;

    OP_REQUIRES_OK(ctx, ctx->ConstantInputAsFloatScalar(1, &minrange));
    OP_REQUIRES_OK(ctx, ctx->ConstantInputAsFloatScalar(2, &maxrange));

    float min_range = static_cast<float>(minrange);
    float max_range = static_cast<float>(maxrange);
    float full_range, half_range;
    if (input_type == DT_QINT8) {
      full_range = get_fullrange<qint8>();
      half_range = (full_range + 1.0f) / 2.0f;
    } else {
      OP_REQUIRES(ctx, input_type == DT_QUINT8,
                  errors::InvalidArgument(
                      "Only support DT_QINT8 or DT_QUINT8, got ", input_type));
      full_range = get_fullrange<quint8>();
      half_range = 0.0f;
    }

    float scale_factor = (max_range - min_range) / full_range;

    xla::XlaOp input = ctx->Input(0);
    xla::XlaOp output;

    output = xla::ConvertElementType(input, xla::F32);

    auto scale = ScalarLike(output, scale_factor);
    auto halfrange = ScalarLike(output, half_range);
    output = xla::Add(xla::Mul(xla::Add(output, halfrange), scale),
                      ScalarLike(output, min_range));

    if (dtype_ == DT_BFLOAT16) {
      output = xla::ConvertElementType(output, xla::BF16);
    }
    ctx->SetOutput(0, output);
  }

 private:
  DataType dtype_;
};

REGISTER_XLA_OP(Name("Dequantize").TypeConstraint("T", kQuantizedType),
                DequantizeOp);

}  // namespace
}  // namespace tensorflow
