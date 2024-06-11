/* Copyright 2019 The OpenXLA Authors.

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

#include "xla/client/lib/comparators.h"

#include <limits>
#include <optional>
#include <string>
#include <vector>

#include "absl/log/check.h"
#include "absl/strings/str_cat.h"
#include "absl/types/span.h"
#include "xla/client/lib/constants.h"
#include "xla/client/xla_builder.h"
#include "xla/client/xla_computation.h"
#include "xla/primitive_util.h"
#include "xla/shape_util.h"
#include "xla/types.h"
#include "xla/util.h"
#include "xla/xla_data.pb.h"

namespace xla {
namespace {

using XlaCompareOp = XlaOp (*)(XlaOp, XlaOp, absl::Span<const int64_t>);

XlaComputation CreateScalarComparisonComputation(
    const std::string& name, const std::vector<PrimitiveType>& operand_types,
    XlaBuilder* builder, XlaCompareOp generator) {
  CHECK_NE(operand_types.size(), 0);
  std::vector<std::optional<XlaCompareOp>> generators(operand_types.size());
  generators[0] = generator;
  return CreateScalarComparisonComputation(name, operand_types, generators,
                                           builder);
}
}  // namespace

XlaComputation CreateScalarComparisonComputation(
    const std::string& name, const std::vector<PrimitiveType>& operand_types,
    const std::vector<std::optional<XlaCompareOp>>& generators,
    XlaBuilder* builder) {
  // Create a default computation where we compare only the first two
  // parameters of type 'operand_types[0]'.
  auto b = builder->CreateSubBuilder(name);
  if (operand_types.empty()) {
    b->ReportError(InvalidArgument("operand_types should not be empty"));
    return b->BuildAndNoteError();
  }

  CHECK_EQ(operand_types.size(), generators.size());
  int parameter_count = 0;
  int last_generator_index = 0;
  std::vector<XlaOp> lhs_params;
  std::vector<XlaOp> rhs_params;

  // For each type in 'operand_types' we create two parameters of this type. The
  // idea is that this computation can be used by n-ary Sort, and potentially
  // should support comparing also the other operands of sort. In this default
  // computation, however, we will not actually use any parameters except the
  // first two.
  for (auto operand_type : operand_types) {
    auto scalar_shape = ShapeUtil::MakeShape(operand_type, {});
    auto lhs_param = Parameter(b.get(), parameter_count * 2, scalar_shape,
                               absl::StrCat("p.", parameter_count, ".lhs"));
    auto rhs_param = Parameter(b.get(), parameter_count * 2 + 1, scalar_shape,
                               absl::StrCat("p.", parameter_count, ".rhs"));
    lhs_params.emplace_back(lhs_param);
    rhs_params.emplace_back(rhs_param);
    if (generators[parameter_count].has_value()) {
      last_generator_index = parameter_count;
    }
    parameter_count++;
  }

  CHECK_NE(parameter_count, 0);

  XlaOp result;
  XlaOp prev_equal;

  for (int i = 0; i < parameter_count; i++) {
    if (generators[i].has_value()) {
      XlaOp cmp_op = generators[i].value()(lhs_params[i], rhs_params[i], {});
      result = prev_equal.valid() ? Select(prev_equal, cmp_op, result) : cmp_op;
      if (i != last_generator_index) {
        XlaOp eq_op = EqTotalOrder(lhs_params[i], rhs_params[i]);
        prev_equal = prev_equal.valid() ? And(prev_equal, eq_op) : eq_op;
      }
    }
  }

  CHECK(result.valid());
  return b->BuildAndNoteError();
}

// Creates a scalar less-than computation and returns it.
XlaComputation CreateScalarLtComputation(
    const std::vector<PrimitiveType>& operand_types, XlaBuilder* builder) {
  return CreateScalarComparisonComputation("compare-less-than", operand_types,
                                           builder, LtTotalOrder);
}

// Creates a scalar greater-than computation and returns it.
XlaComputation CreateScalarGtComputation(
    const std::vector<PrimitiveType>& operand_types, XlaBuilder* builder) {
  return CreateScalarComparisonComputation(
      "compare-greater-than", operand_types, builder, GtTotalOrder);
}

}  // namespace xla
