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

#include "tensorflow/compiler/tf2xla/lib/while_loop.h"
#include "tensorflow/compiler/tf2xla/lib/util.h"
#include "tensorflow/compiler/xla/client/xla_client/xla_builder.h"
#include "tensorflow/compiler/xla/shape_util.h"
#include "tensorflow/compiler/xla/status_macros.h"

namespace tensorflow {

xla::StatusOr<std::vector<xla::XlaOp>> XlaWhileLoop(
    const LoopConditionFunction& condition_function,
    const LoopBodyFunction& body_function,
    gtl::ArraySlice<xla::XlaOp> initial_values, StringPiece name,
    xla::XlaBuilder* builder) {
  int arity = initial_values.size();
  std::vector<xla::Shape> var_shapes;
  var_shapes.reserve(arity);
  for (const xla::XlaOp& input : initial_values) {
    TF_ASSIGN_OR_RETURN(auto shape, builder->GetShape(input));
    var_shapes.push_back(std::move(shape));
  }
  xla::Shape tuple_shape = xla::ShapeUtil::MakeTupleShape(var_shapes);

  // Unpacks a tuple into its component parts.
  auto unpack_tuple = [](xla::XlaOp tuple, int arity,
                         xla::XlaBuilder* builder) {
    std::vector<xla::XlaOp> elements(arity);
    for (int i = 0; i < arity; ++i) {
      elements[i] = xla::GetTupleElement(tuple, i);
    }
    return elements;
  };

  // Build the condition.
  std::unique_ptr<xla::XlaBuilder> cond_builder =
      builder->CreateSubBuilder(strings::StrCat(name, "_condition"));
  {
    auto parameter =
        xla::Parameter(cond_builder.get(), 0, tuple_shape, "parameter");

    TF_RETURN_IF_ERROR(
        condition_function(unpack_tuple(parameter, arity, cond_builder.get()),
                           cond_builder.get())
            .status());
  }
  TF_ASSIGN_OR_RETURN(auto cond, cond_builder->Build());

  // Build the body.
  std::unique_ptr<xla::XlaBuilder> body_builder =
      builder->CreateSubBuilder(strings::StrCat(name, "_body"));
  {
    auto parameter =
        xla::Parameter(body_builder.get(), 0, tuple_shape, "parameter");

    TF_ASSIGN_OR_RETURN(
        auto result,
        body_function(unpack_tuple(parameter, arity, body_builder.get()),
                      body_builder.get()));

    TF_RET_CHECK(result.size() == initial_values.size());
    xla::Tuple(body_builder.get(), result);
  }
  TF_ASSIGN_OR_RETURN(auto body, body_builder->Build());

  auto outputs = xla::While(cond, body, xla::Tuple(builder, initial_values));

  return unpack_tuple(outputs, arity, builder);
}

xla::StatusOr<std::vector<xla::XlaOp>> XlaForEachIndex(
    int64 num_iterations, xla::PrimitiveType num_iterations_type,
    const ForEachIndexBodyFunction& body_function,
    gtl::ArraySlice<xla::XlaOp> initial_values, StringPiece name,
    xla::XlaBuilder* builder) {
  auto while_cond_fn =
      [&](gtl::ArraySlice<xla::XlaOp> values,
          xla::XlaBuilder* cond_builder) -> xla::StatusOr<xla::XlaOp> {
    return xla::Lt(values[0], IntegerLiteral(cond_builder, num_iterations_type,
                                             num_iterations));
  };
  auto while_body_fn = [&](gtl::ArraySlice<xla::XlaOp> values,
                           xla::XlaBuilder* body_builder)
      -> xla::StatusOr<std::vector<xla::XlaOp>> {
    xla::XlaOp iteration = values[0];

    std::vector<xla::XlaOp> updated_values;
    updated_values.reserve(values.size());
    updated_values.push_back(xla::Add(
        iteration,
        xla::ConstantLiteral(body_builder,
                             xla::LiteralUtil::One(num_iterations_type))));

    values.remove_prefix(1);
    TF_ASSIGN_OR_RETURN(std::vector<xla::XlaOp> body_outputs,
                        body_function(iteration, values, body_builder));
    updated_values.insert(updated_values.end(), body_outputs.begin(),
                          body_outputs.end());
    return updated_values;
  };

  std::vector<xla::XlaOp> values;
  values.reserve(initial_values.size() + 1);
  values.push_back(xla::ConstantLiteral(
      builder, xla::LiteralUtil::Zero(num_iterations_type)));
  values.insert(values.end(), initial_values.begin(), initial_values.end());

  TF_ASSIGN_OR_RETURN(values, XlaWhileLoop(while_cond_fn, while_body_fn, values,
                                           name, builder));
  values.erase(values.begin(), values.begin() + 1);
  return values;
}

}  // namespace tensorflow
