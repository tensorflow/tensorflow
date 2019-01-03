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

#include "tensorflow/compiler/xla/client/lib/arithmetic.h"

#include <string>

#include "absl/strings/str_cat.h"
#include "tensorflow/compiler/xla/client/lib/constants.h"
#include "tensorflow/compiler/xla/client/xla_builder.h"
#include "tensorflow/compiler/xla/client/xla_computation.h"
#include "tensorflow/compiler/xla/shape_util.h"
#include "tensorflow/compiler/xla/status_macros.h"
#include "tensorflow/compiler/xla/types.h"
#include "tensorflow/compiler/xla/xla_data.pb.h"

namespace xla {
namespace {

using XlaOpGenerator = XlaOp (*)(XlaBuilder*, const XlaOp&, const XlaOp&);

XlaComputation CreateScalarComputation(const string& name, PrimitiveType type,
                                       XlaBuilder* builder,
                                       XlaOpGenerator generator) {
  std::unique_ptr<XlaBuilder> b;
  if (type == PRED) {
    b = builder->CreateSubBuilder(name);
  } else {
    b = builder->CreateSubBuilder(
        absl::StrCat(name, "_", PrimitiveType_Name(type)));
  }

  const Shape scalar = ShapeUtil::MakeShape(type, {});
  auto lhs = Parameter(b.get(), 0, scalar, "lhs");
  auto rhs = Parameter(b.get(), 1, scalar, "rhs");
  generator(b.get(), lhs, rhs);
  return b->BuildAndNoteError();
}

}  // namespace

XlaComputation CreateScalarAddComputation(PrimitiveType type,
                                          XlaBuilder* builder) {
  return CreateScalarComputation(
      "add", type, builder,
      [](XlaBuilder* b, const XlaOp& lhs, const XlaOp& rhs) {
        return Add(lhs, rhs);
      });
}

XlaComputation CreateScalarMultiplyComputation(PrimitiveType type,
                                               XlaBuilder* builder) {
  return CreateScalarComputation(
      "mul", type, builder,
      [](XlaBuilder* b, const XlaOp& lhs, const XlaOp& rhs) {
        return Mul(lhs, rhs);
      });
}

XlaComputation CreateScalarGeComputation(PrimitiveType type,
                                         XlaBuilder* builder) {
  return CreateScalarComputation("ge", type, builder,
                                 [](XlaBuilder* b, const XlaOp& lhs,
                                    const XlaOp& rhs) { return Ge(lhs, rhs); });
}

XlaComputation CreateScalarMaxComputation(PrimitiveType type,
                                          XlaBuilder* builder) {
  return CreateScalarComputation(
      "max", type, builder,
      [](XlaBuilder* b, const XlaOp& lhs, const XlaOp& rhs) {
        return Max(lhs, rhs);
      });
}

XlaComputation CreateScalarMinComputation(PrimitiveType type,
                                          XlaBuilder* builder) {
  return CreateScalarComputation(
      "min", type, builder,
      [](XlaBuilder* b, const XlaOp& lhs, const XlaOp& rhs) {
        return Min(lhs, rhs);
      });
}

XlaComputation CreateScalarAndComputation(PrimitiveType type,
                                          XlaBuilder* builder) {
  return CreateScalarComputation(
      "and", type, builder,
      [](XlaBuilder* b, const XlaOp& lhs, const XlaOp& rhs) {
        return And(lhs, rhs);
      });
}

XlaComputation CreateScalarOrComputation(PrimitiveType type,
                                         XlaBuilder* builder) {
  return CreateScalarComputation("or", type, builder,
                                 [](XlaBuilder* b, const XlaOp& lhs,
                                    const XlaOp& rhs) { return Or(lhs, rhs); });
}

XlaOp Any(XlaOp predicates) {
  XlaBuilder* builder = predicates.builder();
  return builder->ReportErrorOrReturn([&]() -> StatusOr<XlaOp> {
    auto f = ConstantR0<bool>(builder, false);
    XlaComputation logical_or = CreateScalarOrComputation(PRED, builder);
    TF_ASSIGN_OR_RETURN(const Shape& predicates_shape,
                        builder->GetShape(predicates));
    std::vector<int64> all_dimensions(predicates_shape.rank());
    std::iota(all_dimensions.begin(), all_dimensions.end(), 0);
    return Reduce(predicates, f, logical_or, all_dimensions);
  });
}

}  // namespace xla
