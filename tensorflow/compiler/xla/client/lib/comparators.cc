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

#include "tensorflow/compiler/xla/client/lib/comparators.h"

#include <limits>
#include <string>
#include <vector>

#include "absl/strings/str_cat.h"
#include "absl/types/span.h"
#include "tensorflow/compiler/xla/client/lib/constants.h"
#include "tensorflow/compiler/xla/client/xla_builder.h"
#include "tensorflow/compiler/xla/client/xla_computation.h"
#include "tensorflow/compiler/xla/primitive_util.h"
#include "tensorflow/compiler/xla/shape_util.h"
#include "tensorflow/compiler/xla/types.h"
#include "tensorflow/compiler/xla/xla_data.pb.h"

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
  int64_t parameter_count = 0;
  int64_t last_generator_index = 0;
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

  auto shape_or = b->GetShape(lhs_params[0]);
  if (!shape_or.ok()) {
    b->ReportError(shape_or.status());
    return {};
  }
  Shape shape = shape_or.ValueOrDie();
  shape.set_element_type(PRED);
  XlaOp param_equal =
      Broadcast(One(b.get(), shape.element_type()), shape.dimensions());
  XlaOp result = param_equal;

  for (int64_t i = 0; i < parameter_count; i++) {
    if (generators[i].has_value()) {
      result = Select(param_equal,
                      generators[i].value()(lhs_params[i], rhs_params[i], {}),
                      result);
      if (i != last_generator_index) {
        param_equal =
            And(param_equal, EqTotalOrder(lhs_params[i], rhs_params[i]));
      }
    }
  }

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
