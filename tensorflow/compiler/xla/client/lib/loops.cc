/* Copyright 2018 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/compiler/xla/client/lib/loops.h"

#include "tensorflow/compiler/xla/client/lib/constants.h"
#include "tensorflow/compiler/xla/client/xla_builder.h"
#include "tensorflow/compiler/xla/shape_util.h"
#include "tensorflow/compiler/xla/status_macros.h"

namespace xla {

StatusOr<std::vector<XlaOp>> WhileLoopHelper(
    const WhileLoopHelperConditionFunction& condition_function,
    const WhileLoopHelperBodyFunction& body_function,
    absl::Span<const XlaOp> initial_values, absl::string_view name,
    XlaBuilder* builder) {
  int arity = initial_values.size();
  std::vector<Shape> var_shapes;
  var_shapes.reserve(arity);
  for (const XlaOp& input : initial_values) {
    TF_ASSIGN_OR_RETURN(auto shape, builder->GetShape(input));
    var_shapes.push_back(std::move(shape));
  }
  Shape tuple_shape = ShapeUtil::MakeTupleShape(var_shapes);

  // Unpacks a tuple into its component parts.
  auto unpack_tuple = [](XlaOp tuple, int arity, XlaBuilder* builder) {
    std::vector<XlaOp> elements(arity);
    for (int i = 0; i < arity; ++i) {
      elements[i] = GetTupleElement(tuple, i);
    }
    return elements;
  };

  // Build the condition.
  std::unique_ptr<XlaBuilder> cond_builder =
      builder->CreateSubBuilder(absl::StrCat(name, "_condition"));
  {
    auto parameter = Parameter(cond_builder.get(), 0, tuple_shape, "parameter");

    TF_RETURN_IF_ERROR(
        condition_function(unpack_tuple(parameter, arity, cond_builder.get()),
                           cond_builder.get())
            .status());
  }
  TF_ASSIGN_OR_RETURN(auto cond, cond_builder->Build());

  // Build the body.
  std::unique_ptr<XlaBuilder> body_builder =
      builder->CreateSubBuilder(absl::StrCat(name, "_body"));
  {
    auto parameter = Parameter(body_builder.get(), 0, tuple_shape, "parameter");

    TF_ASSIGN_OR_RETURN(
        auto result,
        body_function(unpack_tuple(parameter, arity, body_builder.get()),
                      body_builder.get()));

    TF_RET_CHECK(result.size() == initial_values.size());
    Tuple(body_builder.get(), result);
  }
  TF_ASSIGN_OR_RETURN(auto body, body_builder->Build());

  auto outputs = While(cond, body, Tuple(builder, initial_values));

  return unpack_tuple(outputs, arity, builder);
}

StatusOr<std::vector<XlaOp>> ForEachIndex(
    int64 num_iterations, PrimitiveType num_iterations_type,
    const ForEachIndexBodyFunction& body_function,
    absl::Span<const XlaOp> initial_values, absl::string_view name,
    XlaBuilder* builder) {
  auto while_cond_fn = [&](absl::Span<const XlaOp> values,
                           XlaBuilder* cond_builder) -> StatusOr<XlaOp> {
    return Lt(values[0], ConstantR0WithType(cond_builder, num_iterations_type,
                                            num_iterations));
  };
  auto while_body_fn =
      [&](absl::Span<const XlaOp> values,
          XlaBuilder* body_builder) -> StatusOr<std::vector<XlaOp>> {
    XlaOp iteration = values[0];

    std::vector<XlaOp> updated_values;
    updated_values.reserve(values.size());
    updated_values.push_back(Add(
        iteration,
        ConstantLiteral(body_builder, LiteralUtil::One(num_iterations_type))));

    values.remove_prefix(1);
    TF_ASSIGN_OR_RETURN(std::vector<XlaOp> body_outputs,
                        body_function(iteration, values, body_builder));
    updated_values.insert(updated_values.end(), body_outputs.begin(),
                          body_outputs.end());
    return updated_values;
  };

  std::vector<XlaOp> values;
  values.reserve(initial_values.size() + 1);
  values.push_back(
      ConstantLiteral(builder, LiteralUtil::Zero(num_iterations_type)));
  values.insert(values.end(), initial_values.begin(), initial_values.end());

  TF_ASSIGN_OR_RETURN(values, WhileLoopHelper(while_cond_fn, while_body_fn,
                                              values, name, builder));
  values.erase(values.begin(), values.begin() + 1);
  return values;
}

}  // namespace xla
