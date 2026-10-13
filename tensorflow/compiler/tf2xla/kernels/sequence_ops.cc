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

#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <limits>
#include <type_traits>

#include "absl/status/statusor.h"
#include "tensorflow/compiler/tf2xla/xla_op_kernel.h"
#include "tensorflow/compiler/tf2xla/xla_op_registry.h"
#include "xla/hlo/builder/lib/constants.h"
#include "xla/hlo/builder/value_inference.h"
#include "xla/hlo/builder/xla_builder.h"
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

// The type-specific part of the implementation of Range: checks the arguments
// and returns delta * [0, 1, ..., size - 1] for the range's size.
template <typename T>
absl::StatusOr<xla::XlaOp> CreateRangeTensor(
    const xla::LiteralSlice& start_literal,
    const xla::LiteralSlice& limit_literal,
    const xla::LiteralSlice& delta_literal, xla::XlaBuilder* builder) {
  T start = start_literal.Get<T>({});
  T limit = limit_literal.Get<T>({});
  T delta = delta_literal.Get<T>({});

  if (delta == static_cast<T>(0)) {
    return errors::InvalidArgument("Requires delta != 0: ", delta);
  }
  if (delta > static_cast<T>(0)) {
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
  int64_t size;
  if constexpr (std::is_integral_v<T>) {
    // Take |limit - start| and |delta| in uint64_t, which holds them for every
    // T, as the CPU kernel does, so that a range wider than T can hold doesn't
    // overflow.
    const bool count_up = delta > static_cast<T>(0);
    const uint64_t range =
        count_up ? static_cast<uint64_t>(limit) - static_cast<uint64_t>(start)
                 : static_cast<uint64_t>(start) - static_cast<uint64_t>(limit);
    const uint64_t abs_delta = count_up ? static_cast<uint64_t>(delta)
                                        : -static_cast<uint64_t>(delta);
    const uint64_t size_unsigned = range == 0 ? 0 : (range - 1) / abs_delta + 1;
    if (size_unsigned > std::numeric_limits<int64_t>::max()) {
      return errors::InvalidArgument("Requires ((limit - start) / delta) <= ",
                                     std::numeric_limits<int64_t>::max());
    }
    size = static_cast<int64_t>(size_unsigned);
  } else {
    double start_f = static_cast<double>(start);
    double limit_f = static_cast<double>(limit);
    double delta_f = static_cast<double>(delta);
    size = static_cast<int64_t>(
        std::ceil(std::abs((limit_f - start_f) / delta_f)));
  }

  return xla::ConstantR0(builder, delta) *
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
    OP_REQUIRES_OK(ctx, ctx->ConstantInput(2, &delta));
    // Size the output from the bounds of start and limit that give the longest
    // range: the lowest start and highest limit when counting up, and the
    // highest start and lowest limit when counting down.
    const xla::PrimitiveType delta_type = delta.shape().element_type();
    const bool count_down =
        xla::primitive_util::IsFloatingPointType(delta_type)
            ? delta.GetAsDouble({}).value_or(0) < 0
            : xla::primitive_util::IsSignedIntegralType(delta_type) &&
                  delta.GetIntegralAsS64({}).value_or(0) < 0;
    OP_REQUIRES_OK(ctx, ctx->ConstantInput(
                            0, &start,
                            count_down ? xla::ValueInferenceMode::kUpperBound
                                       : xla::ValueInferenceMode::kLowerBound));
    OP_REQUIRES_OK(ctx, ctx->ConstantInput(
                            1, &limit,
                            count_down ? xla::ValueInferenceMode::kLowerBound
                                       : xla::ValueInferenceMode::kUpperBound));

    DataType type = input_type(0);
    absl::StatusOr<xla::XlaOp> output;
    switch (type) {
      case DT_INT8:
        output = CreateRangeTensor<int8_t>(start, limit, delta, ctx->builder());
        break;
      case DT_INT16:
        output =
            CreateRangeTensor<int16_t>(start, limit, delta, ctx->builder());
        break;
      case DT_INT32:
        output =
            CreateRangeTensor<int32_t>(start, limit, delta, ctx->builder());
        break;
      case DT_INT64:
        output =
            CreateRangeTensor<int64_t>(start, limit, delta, ctx->builder());
        break;
      case DT_UINT16:
        output =
            CreateRangeTensor<uint16_t>(start, limit, delta, ctx->builder());
        break;
      case DT_UINT32:
        output =
            CreateRangeTensor<uint32_t>(start, limit, delta, ctx->builder());
        break;
      case DT_UINT64:
        output =
            CreateRangeTensor<uint64_t>(start, limit, delta, ctx->builder());
        break;
      case DT_HALF:
        output =
            CreateRangeTensor<Eigen::half>(start, limit, delta, ctx->builder());
        break;
      case DT_BFLOAT16:
        output =
            CreateRangeTensor<bfloat16>(start, limit, delta, ctx->builder());
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
    // If start is dynamic, its literal is a bound, so add the input instead.
    output = (start_is_dynamic ? ctx->Input(0)
                               : xla::ConstantLiteral(ctx->builder(), start)) +
             output.value();

    if (start_is_dynamic || limit_is_dynamic) {
      xla::XlaOp delta = ctx->Input(2);
      xla::XlaOp limit = ctx->Input(1);
      xla::XlaOp start = ctx->Input(0);
      // The runtime values can be in the wrong order for delta's sign, which
      // the CPU kernel rejects. Give such a range size 0 rather than a size
      // past the output's static size.
      const xla::XlaOp is_empty =
          count_down ? xla::Le(start, limit) : xla::Ge(start, limit);
      xla::XlaOp dynamic_size;
      if (DataTypeIsInteger(type)) {
        // Take |limit - start| and |delta| in the unsigned type of the same
        // width, which holds them. XLA's Abs doesn't take unsigned types, whose
        // delta is positive anyway.
        const xla::PrimitiveType xla_type = ctx->input_xla_type(0);
        const xla::PrimitiveType unsigned_type =
            xla::primitive_util::UnsignedIntegralTypeForBitWidth(
                xla::primitive_util::BitWidth(xla_type));
        const auto as_unsigned = [&](xla::XlaOp x) {
          return xla::BitcastConvertType(x, unsigned_type);
        };
        const xla::XlaOp range = count_down
                                     ? as_unsigned(start) - as_unsigned(limit)
                                     : as_unsigned(limit) - as_unsigned(start);
        const xla::XlaOp abs_delta =
            as_unsigned(xla::primitive_util::IsSignedIntegralType(xla_type)
                            ? xla::Abs(delta)
                            : delta);
        const xla::XlaOp one = xla::One(ctx->builder(), unsigned_type);
        dynamic_size =
            xla::Select(is_empty, xla::Zero(ctx->builder(), unsigned_type),
                        (range - one) / abs_delta + one);
      } else {
        dynamic_size = xla::Select(
            is_empty, xla::Zero(ctx->builder(), ctx->input_xla_type(0)),
            xla::Ceil(xla::Abs((limit - start) / delta)));
      }
      dynamic_size = xla::ConvertElementType(dynamic_size, xla::S32);
      output = xla::SetDimensionSize(output.value(), dynamic_size, 0);
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
