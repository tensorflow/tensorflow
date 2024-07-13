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

// XLA-specific sequence and range Ops.

#include "absl/status/statusor.h"
#include "tensorflow/compiler/tf2xla/xla_op_kernel.h"
#include "tensorflow/compiler/tf2xla/xla_op_registry.h"
#include "xla/client/lib/constants.h"
#include "xla/client/value_inference.h"
#include "xla/client/xla_builder.h"
#include "xla/literal.h"
#include "xla/primitive_util.h"
#include "xla/xla_data.pb.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/op_requires.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/framework/types.pb.h"
#include "tensorflow/core/platform/errors.h"
#include "tensorflow/core/platform/types.h"

namespace tensorflow {
namespace {

// The type-specific part of the implementation of Range.
template <typename T>
absl::StatusOr<xla::XlaOp> CreateRangeTensor(
    const xla::LiteralSlice& start_literal,
    const xla::LiteralSlice& limit_literal,
    const xla::LiteralSlice& delta_literal, xla::XlaBuilder* builder) {
  T start = start_literal.Get<T>({});
  T limit = limit_literal.Get<T>({});
  T delta = delta_literal.Get<T>({});

  if (delta == 0) {
    return errors::InvalidArgument("Requires delta != 0: ", delta);
  }
  if (delta > 0) {
    if (start > limit) {
      return errors::InvalidArgument(
          "Requires start <= limit when delta > 0: ", start, "/", limit);
    }
  } else {
    if (start < limit) {
      return errors::InvalidArgument(
          "Requires start >= limit when delta < 0: ", start, "/", limit);
    }
  }
  int64_t size =
      (std::is_integral<T>::value
           ? static_cast<T>(
                 limit == start
                     ? 0
                     : (std::abs(limit - start) - 1) / std::abs(delta) + 1)
           : std::ceil(std::abs((limit - start) / delta)));

  return xla::ConstantR0(builder, start) +
         xla::ConstantR0(builder, delta) *
             xla::Iota(builder, xla::primitive_util::NativeToPrimitiveType<T>(),
                       size);
}

class RangeOp : public XlaOpKernel {
 public:
  explicit RangeOp(OpKernelConstruction* ctx) : XlaOpKernel(ctx) {}

  void Compile(XlaOpKernelContext* ctx) override {
    const TensorShape start_in_shape = ctx->InputShape(0);
    const TensorShape limit_in_shape = ctx->InputShape(1);
    const TensorShape delta_in_shape = ctx->InputShape(2);
    OP_REQUIRES(ctx, TensorShapeUtils::IsScalar(start_in_shape),
                errors::InvalidArgument("start must be a scalar, not shape ",
                                        start_in_shape.DebugString()));
    OP_REQUIRES(ctx, TensorShapeUtils::IsScalar(limit_in_shape),
                errors::InvalidArgument("limit must be a scalar, not shape ",
                                        limit_in_shape.DebugString()));
    OP_REQUIRES(ctx, TensorShapeUtils::IsScalar(delta_in_shape),
                errors::InvalidArgument("delta must be a scalar, not shape ",
                                        delta_in_shape.DebugString()));
    xla::Literal start, limit, delta;
    OP_REQUIRES_OK(ctx, ctx->ConstantInput(
                            0, &start, xla::ValueInferenceMode::kLowerBound));
    OP_REQUIRES_OK(ctx, ctx->ConstantInput(
                            1, &limit, xla::ValueInferenceMode::kUpperBound));
    OP_REQUIRES_OK(ctx, ctx->ConstantInput(2, &delta));

    DataType type = input_type(0);
    absl::StatusOr<xla::XlaOp> output;
    switch (type) {
      case DT_INT32:
        output = CreateRangeTensor<int32>(start, limit, delta, ctx->builder());
        break;
      case DT_INT64:
        output =
            CreateRangeTensor<int64_t>(start, limit, delta, ctx->builder());
        break;
      case DT_FLOAT:
        output = CreateRangeTensor<float>(start, limit, delta, ctx->builder());
        break;
      case DT_DOUBLE:
        output = CreateRangeTensor<double>(start, limit, delta, ctx->builder());
        break;
      default:
        output = errors::InvalidArgument("Invalid type for Range ",
                                         DataTypeString(type));
    }
    OP_REQUIRES_OK(ctx, output.status());
    bool start_is_dynamic = false;
    OP_REQUIRES_OK(ctx,
                   ctx->ResolveInputDynamismIntoPred(0, &start_is_dynamic));
    bool limit_is_dynamic = false;
    OP_REQUIRES_OK(ctx,
                   ctx->ResolveInputDynamismIntoPred(1, &limit_is_dynamic));

    if (start_is_dynamic || limit_is_dynamic) {
      xla::XlaOp delta = ctx->Input(2);
      xla::XlaOp limit = ctx->Input(1);
      xla::XlaOp start = ctx->Input(0);
      if (type == DT_INT32 || type == DT_INT64) {
        auto dynamic_size = (xla::Abs(limit - start) + xla::Abs(delta) -
                             xla::One(ctx->builder(), ctx->input_xla_type(0))) /
                            xla::Abs(delta);
        dynamic_size = xla::ConvertElementType(dynamic_size, xla::S32);
        output = xla::SetDimensionSize(output.value(), dynamic_size, 0);
      } else {
        auto dynamic_size = (xla::Ceil(xla::Abs((limit - start) / delta)));
        dynamic_size = xla::ConvertElementType(dynamic_size, xla::S32);
        output = xla::SetDimensionSize(output.value(), dynamic_size, 0);
      }
    }

    ctx->SetOutput(0, output.value());
  }
};

REGISTER_XLA_OP(Name("Range")
                    .CompileTimeConstantInput("start")
                    .CompileTimeConstantInput("limit")
                    .CompileTimeConstantInput("delta"),
                RangeOp);

class LinSpaceOp : public XlaOpKernel {
 public:
  explicit LinSpaceOp(OpKernelConstruction* ctx) : XlaOpKernel(ctx) {}

  void Compile(XlaOpKernelContext* ctx) override {
    const TensorShape start_in_shape = ctx->InputShape("start");
    const TensorShape stop_in_shape = ctx->InputShape("stop");
    const TensorShape num_in_shape = ctx->InputShape("num");
    OP_REQUIRES(ctx, TensorShapeUtils::IsScalar(start_in_shape),
                errors::InvalidArgument("start must be a scalar, not shape ",
                                        start_in_shape.DebugString()));
    OP_REQUIRES(ctx, TensorShapeUtils::IsScalar(stop_in_shape),
                errors::InvalidArgument("stop must be a scalar, not shape ",
                                        stop_in_shape.DebugString()));
    OP_REQUIRES(ctx, TensorShapeUtils::IsScalar(num_in_shape),
                errors::InvalidArgument("num must be a scalar, not shape ",
                                        num_in_shape.DebugString()));

    int64_t num;
    OP_REQUIRES_OK(ctx, ctx->ConstantInputAsIntScalar("num", &num));
    OP_REQUIRES(ctx, num > 0,
                errors::InvalidArgument("Requires num > 0: ", num));
    xla::XlaOp start = ctx->Input("start");
    xla::XlaOp stop = ctx->Input("stop");
    xla::XlaOp iota = xla::Iota(ctx->builder(), ctx->output_xla_type(0), num);
    xla::XlaOp step =
        (stop - start) / xla::ScalarLike(start, (num > 1 ? num - 1 : num));
    xla::XlaOp result = iota * step + start;
    if (num > 1) {
      // According to linspace spec, start has to be the first element and end
      // has to be last element.
      xla::XlaOp mask = xla::Iota(ctx->builder(), xla::S64, num);
      xla::XlaOp eq = xla::Eq(mask, xla::ScalarLike(mask, num - 1));
      result = xla::Select(eq, stop, result);
    }
    ctx->SetOutput(0, result);
  }
};

REGISTER_XLA_OP(Name("LinSpace").CompileTimeConstantInput("num"), LinSpaceOp);

}  // namespace
}  // namespace tensorflow
