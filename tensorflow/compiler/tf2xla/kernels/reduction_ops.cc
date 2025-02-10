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
#include "tensorflow/compiler/tf2xla/xla_helpers.h"
#include "tensorflow/compiler/tf2xla/xla_op_registry.h"
#include "xla/hlo/builder/lib/constants.h"
#include "xla/hlo/builder/xla_builder.h"
#include "xla/shape.h"
#include "xla/xla_data.pb.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/op_requires.h"
#include "tensorflow/core/platform/errors.h"
#include "tensorflow/core/platform/status.h"

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

class MeanOp : public XlaReductionOp {
 public:
  explicit MeanOp(OpKernelConstruction* ctx)
      : XlaReductionOp(ctx,
                       XlaHelpers::SumAccumulationType(ctx->input_type(0))) {}

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

}  // namespace
}  // namespace tensorflow
