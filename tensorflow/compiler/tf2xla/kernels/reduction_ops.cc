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

// XLA-specific reduction Ops.

#include "tensorflow/compiler/tf2xla/kernels/reduction_ops.h"

#include <cstdint>
#include <limits>
#include <vector>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "tensorflow/compiler/tf2xla/xla_helpers.h"
#include "tensorflow/compiler/tf2xla/xla_op_registry.h"
#include "xla/hlo/builder/lib/constants.h"
#include "xla/hlo/builder/lib/math.h"
#include "xla/hlo/builder/xla_builder.h"
#include "xla/hlo/builder/xla_computation.h"
#include "xla/primitive_util.h"
#include "xla/shape.h"
#include "xla/shape_util.h"
#include "xla/xla_data.pb.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/op_requires.h"
#include "tensorflow/core/framework/types.pb.h"
#include "tensorflow/core/platform/errors.h"

namespace tensorflow {
namespace {

class SumOp : public XlaReductionOp {
 public:
  explicit SumOp(OpKernelConstruction* ctx)
      : XlaReductionOp(ctx,
                       XlaHelpers::SumAccumulationType(ctx->input_type(0))) {}
  xla::XlaOp InitialValue(xla::XlaBuilder* builder) override {
    return xla::Zero(builder, xla_reduction_type_);
  }
  void BuildReducer(xla::XlaBuilder* builder, const xla::XlaOp& scalar_lhs,
                    const xla::XlaOp& scalar_rhs) override {
    xla::Add(scalar_lhs, scalar_rhs);
  }
};

REGISTER_XLA_OP(Name("Sum").CompileTimeConstantInput("reduction_indices"),
                SumOp);

class ProdOp : public XlaReductionOp {
 public:
  explicit ProdOp(OpKernelConstruction* ctx)
      : XlaReductionOp(ctx,
                       XlaHelpers::SumAccumulationType(ctx->input_type(0))) {}

  xla::XlaOp InitialValue(xla::XlaBuilder* builder) override {
    return xla::One(builder, xla_reduction_type_);
  }

  void BuildReducer(xla::XlaBuilder* builder, const xla::XlaOp& scalar_lhs,
                    const xla::XlaOp& scalar_rhs) override {
    xla::Mul(scalar_lhs, scalar_rhs);
  }

  xla::XlaOp BuildFinalizer(
      xla::XlaBuilder* builder, const xla::XlaOp& input,
      const xla::XlaOp& reduce_output,
      const std::vector<int64_t>& dimensions_to_reduce) override {
    return builder->ReportErrorOrReturn([&]() -> absl::StatusOr<xla::XlaOp> {
      xla::XlaOp final_output = XlaReductionOp::BuildFinalizer(
          builder, input, reduce_output, dimensions_to_reduce);

      if (xla::primitive_util::IsComplexType(xla_reduction_type_)) {
        // For complex types (C64/C128), inspect real and imaginary parts
        // separately since IsInf/IsNan operate on real floating-point scalars.
        xla::XlaOp zero = xla::Zero(builder, xla_reduction_type_);
        xla::XlaOp is_zero = xla::Eq(input, zero);
        xla::XlaOp real = xla::Real(input);
        xla::XlaOp imag = xla::Imag(input);
        xla::XlaOp is_nan_or_inf = xla::Or(xla::IsNan(real), xla::IsInf(real),
                                           xla::IsNan(imag), xla::IsInf(imag));

        xla::XlaBuilder r("prod_complex_finalizer_reduction");
        xla::Shape pred_shape = xla::ShapeUtil::MakeShape(xla::PRED, {});
        xla::XlaOp lhs_zero = xla::Parameter(&r, 0, pred_shape, "lhs_zero");
        xla::XlaOp lhs_nan_inf =
            xla::Parameter(&r, 1, pred_shape, "lhs_nan_inf");
        xla::XlaOp rhs_zero = xla::Parameter(&r, 2, pred_shape, "rhs_zero");
        xla::XlaOp rhs_nan_inf =
            xla::Parameter(&r, 3, pred_shape, "rhs_nan_inf");
        xla::Tuple(&r, {xla::Or(lhs_zero, rhs_zero),
                        xla::Or(lhs_nan_inf, rhs_nan_inf)});
        absl::StatusOr<xla::XlaComputation> finalizer_comp = r.Build();
        if (!finalizer_comp.ok()) {
          return finalizer_comp.status();
        }

        xla::XlaOp false_val = xla::ConstantR0<bool>(builder, false);
        xla::XlaOp reduced_flags = xla::Reduce(
            builder, {is_zero, is_nan_or_inf}, {false_val, false_val},
            *finalizer_comp, dimensions_to_reduce);
        xla::XlaOp any_zero = xla::GetTupleElement(reduced_flags, 0);
        xla::XlaOp any_nan_or_inf = xla::GetTupleElement(reduced_flags, 1);

        xla::XlaOp out_real = xla::Real(final_output);
        xla::XlaOp out_imag = xla::Imag(final_output);
        xla::XlaOp output_is_nan =
            xla::Or(xla::IsNan(out_real), xla::IsNan(out_imag));
        xla::XlaOp should_be_zero = xla::And(
            output_is_nan, xla::And(any_zero, xla::Not(any_nan_or_inf)));
        return xla::Select(should_be_zero, xla::ZerosLike(final_output),
                           final_output);
      }

      if (!xla::primitive_util::IsFloatingPointType(xla_reduction_type_)) {
        return final_output;
      }

      xla::XlaOp zero = xla::Zero(builder, xla_reduction_type_);
      xla::XlaOp is_zero = xla::Eq(input, zero);
      xla::XlaOp is_nan_or_inf = xla::Or(xla::IsNan(input), xla::IsInf(input));
      xla::XlaOp is_negative =
          xla::Or(xla::Lt(input, zero), xla::IsNegZero(input));

      xla::XlaBuilder r("prod_finalizer_reduction");
      xla::Shape pred_shape = xla::ShapeUtil::MakeShape(xla::PRED, {});
      xla::XlaOp lhs_zero = xla::Parameter(&r, 0, pred_shape, "lhs_zero");
      xla::XlaOp lhs_nan_inf = xla::Parameter(&r, 1, pred_shape, "lhs_nan_inf");
      xla::XlaOp lhs_neg = xla::Parameter(&r, 2, pred_shape, "lhs_neg");
      xla::XlaOp rhs_zero = xla::Parameter(&r, 3, pred_shape, "rhs_zero");
      xla::XlaOp rhs_nan_inf = xla::Parameter(&r, 4, pred_shape, "rhs_nan_inf");
      xla::XlaOp rhs_neg = xla::Parameter(&r, 5, pred_shape, "rhs_neg");
      xla::Tuple(
          &r, {xla::Or(lhs_zero, rhs_zero), xla::Or(lhs_nan_inf, rhs_nan_inf),
               xla::Xor(lhs_neg, rhs_neg)});
      absl::StatusOr<xla::XlaComputation> finalizer_comp = r.Build();
      if (!finalizer_comp.ok()) {
        return finalizer_comp.status();
      }

      xla::XlaOp false_val = xla::ConstantR0<bool>(builder, false);
      xla::XlaOp reduced_flags =
          xla::Reduce(builder, {is_zero, is_nan_or_inf, is_negative},
                      {false_val, false_val, false_val}, *finalizer_comp,
                      dimensions_to_reduce);
      xla::XlaOp any_zero = xla::GetTupleElement(reduced_flags, 0);
      xla::XlaOp any_nan_or_inf = xla::GetTupleElement(reduced_flags, 1);
      xla::XlaOp odd_negatives = xla::GetTupleElement(reduced_flags, 2);

      xla::XlaOp output_is_nan = xla::IsNan(final_output);

      xla::XlaOp should_be_zero =
          xla::And(output_is_nan, xla::And(any_zero, xla::Not(any_nan_or_inf)));

      xla::XlaOp pos_zero = xla::ZerosLike(final_output);
      xla::XlaOp signed_zero =
          xla::Select(odd_negatives, xla::Neg(pos_zero), pos_zero);

      return xla::Select(should_be_zero, signed_zero, final_output);
    });
  }
};

REGISTER_XLA_OP(Name("Prod").CompileTimeConstantInput("reduction_indices"),
                ProdOp);

class MinOp : public XlaReductionOp {
 public:
  explicit MinOp(OpKernelConstruction* ctx)
      : XlaReductionOp(ctx, ctx->input_type(0)) {}

  xla::XlaOp InitialValue(xla::XlaBuilder* builder) override {
    return xla::MaxValue(builder, xla_reduction_type_);
  }

  void BuildReducer(xla::XlaBuilder* builder, const xla::XlaOp& scalar_lhs,
                    const xla::XlaOp& scalar_rhs) override {
    xla::Min(scalar_lhs, scalar_rhs);
  }
};

REGISTER_XLA_OP(Name("Min").CompileTimeConstantInput("reduction_indices"),
                MinOp);

class MaxOp : public XlaReductionOp {
 public:
  explicit MaxOp(OpKernelConstruction* ctx)
      : XlaReductionOp(ctx, ctx->input_type(0)) {
    OP_REQUIRES_OK(ctx, PrimitiveTypeCheck(xla_reduction_type_));
  }

  static absl::Status PrimitiveTypeCheck(
      xla::PrimitiveType xla_reduction_type) {
    if (xla_reduction_type == xla::C64 || xla_reduction_type == xla::C128 ||
        xla_reduction_type == xla::TUPLE ||
        xla_reduction_type == xla::OPAQUE_TYPE) {
      return errors::InvalidArgument(
          "Unsupported PrimitiveType in MaxOp: '",
          xla::PrimitiveType_Name(xla_reduction_type), "'");
    } else {
      return absl::OkStatus();
    }
  }

  xla::XlaOp InitialValue(xla::XlaBuilder* builder) override {
    return xla::MinValue(builder, xla_reduction_type_);
  }

  void BuildReducer(xla::XlaBuilder* builder, const xla::XlaOp& scalar_lhs,
                    const xla::XlaOp& scalar_rhs) override {
    xla::Max(scalar_lhs, scalar_rhs);
  }
};

REGISTER_XLA_OP(Name("Max").CompileTimeConstantInput("reduction_indices"),
                MaxOp);

// Mean divides the accumulated sum by the number of reduced elements, so an
// accumulator that wraps does not merely produce a wrapped sum the way Sum
// does, it produces a quotient that is wrong in both sign and magnitude. That
// is why Mean needs a wider accumulator than the other reductions:
// `SumAccumulationType` already widens the 8 and 16 bit integer types to avoid
// overflow, but leaves the 32 bit ones alone, so reducing int32 or uint32 gave
// a negative mean for strictly positive inputs and disagreed with the eager
// kernel. The 64 bit types have nothing wider to accumulate in and keep
// wrapping, which is what the eager kernel does for them as well.
DataType MeanAccumulationType(const DataType& dtype) {
  if (dtype == DT_INT32) return DT_INT64;
  if (dtype == DT_UINT32) return DT_UINT64;
  return XlaHelpers::SumAccumulationType(dtype);
}

class MeanOp : public XlaReductionOp {
 public:
  explicit MeanOp(OpKernelConstruction* ctx)
      : XlaReductionOp(ctx, MeanAccumulationType(ctx->input_type(0))) {}

  xla::XlaOp InitialValue(xla::XlaBuilder* builder) override {
    return xla::Zero(builder, xla_reduction_type_);
  }
  void BuildReducer(xla::XlaBuilder* builder, const xla::XlaOp& scalar_lhs,
                    const xla::XlaOp& scalar_rhs) override {
    xla::Add(scalar_lhs, scalar_rhs);
  }

  xla::XlaOp BuildFinalizer(
      xla::XlaBuilder* builder, const xla::XlaOp& input,
      const xla::XlaOp& reduce_output,
      const std::vector<int64_t>& dimensions_to_reduce) override {
    if (dimensions_to_reduce.empty()) {
      return reduce_output;
    }
    xla::XlaOp result = reduce_output;
    xla::Shape bounded_shape = builder->GetShape(input).value();
    int64_t divisor_value = bounded_shape.dimensions(dimensions_to_reduce[0]);
    auto divisor = xla::GetDimensionSize(input, dimensions_to_reduce[0]);
    for (int i = 1; i < dimensions_to_reduce.size(); i++) {
      int64_t size_value = bounded_shape.dimensions(dimensions_to_reduce[i]);
      auto size = xla::GetDimensionSize(input, dimensions_to_reduce[i]);
      if (size_value * divisor_value > std::numeric_limits<int32_t>::max()) {
        result = result / xla::ConvertElementType(divisor, xla_reduction_type_);
        divisor_value = size_value;
        divisor = size;
      } else {
        divisor = xla::Mul(divisor, size);
        divisor_value = size_value * divisor_value;
      }
    }
    divisor = xla::ConvertElementType(divisor, xla_reduction_type_);
    return XlaHelpers::ConvertElementType(result / divisor, input_type(0));
  }
};

REGISTER_XLA_OP(Name("Mean").CompileTimeConstantInput("reduction_indices"),
                MeanOp);

class AllOp : public XlaReductionOp {
 public:
  explicit AllOp(OpKernelConstruction* ctx)
      : XlaReductionOp(ctx, ctx->input_type(0)) {}

  xla::XlaOp InitialValue(xla::XlaBuilder* builder) override {
    return xla::ConstantR0<bool>(builder, true);
  }

  void BuildReducer(xla::XlaBuilder* builder, const xla::XlaOp& scalar_lhs,
                    const xla::XlaOp& scalar_rhs) override {
    xla::And(scalar_lhs, scalar_rhs);
  }
};

REGISTER_XLA_OP(Name("All").CompileTimeConstantInput("reduction_indices"),
                AllOp);

class AnyOp : public XlaReductionOp {
 public:
  explicit AnyOp(OpKernelConstruction* ctx)
      : XlaReductionOp(ctx, ctx->input_type(0)) {}

  xla::XlaOp InitialValue(xla::XlaBuilder* builder) override {
    return xla::ConstantR0<bool>(builder, false);
  }

  void BuildReducer(xla::XlaBuilder* builder, const xla::XlaOp& scalar_lhs,
                    const xla::XlaOp& scalar_rhs) override {
    xla::Or(scalar_lhs, scalar_rhs);
  }
};

REGISTER_XLA_OP(Name("Any").CompileTimeConstantInput("reduction_indices"),
                AnyOp);

class EuclideanNormOp : public XlaReductionOp {
 public:
  explicit EuclideanNormOp(OpKernelConstruction* ctx)
      : XlaReductionOp(ctx,
                       XlaHelpers::SumAccumulationType(ctx->input_type(0))) {}
  xla::XlaOp InitialValue(xla::XlaBuilder* builder) override {
    return xla::Zero(builder, xla_reduction_type_);
  }

  xla::XlaOp PreprocessInput(xla::XlaBuilder* /*builder*/,
                             const xla::XlaOp& data) override {
    return xla::Mul(data, xla::MaybeConjugate(data, true));
  }

  void BuildReducer(xla::XlaBuilder* builder, const xla::XlaOp& scalar_lhs,
                    const xla::XlaOp& scalar_rhs) override {
    xla::Add(scalar_lhs, scalar_rhs);
  }

  xla::XlaOp BuildFinalizer(
      xla::XlaBuilder* /*builder*/, const xla::XlaOp& input,
      const xla::XlaOp& reduce_output,
      const std::vector<int64_t>& dimensions_to_reduce) override {
    if (xla::primitive_util::IsIntegralType(xla_reduction_type_)) {
      // XLA only supports float and complex sqrt.
      // Thus, cast integral type to F32 for computation.
      return XlaHelpers::ConvertElementType(
          xla::Sqrt(xla::ConvertElementType(reduce_output, xla::F32)),
          input_type(0));
    }
    return XlaHelpers::ConvertElementType(xla::Sqrt(reduce_output),
                                          input_type(0));
  }
};

REGISTER_XLA_OP(
    Name("EuclideanNorm").CompileTimeConstantInput("reduction_indices"),
    EuclideanNormOp);

}  // namespace
}  // namespace tensorflow
