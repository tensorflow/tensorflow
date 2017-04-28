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

#include "tensorflow/compiler/xla/client/computation_builder.h"
#include "tensorflow/compiler/xla/shape_util.h"
#include "tensorflow/compiler/xla/status_macros.h"
#include "tensorflow/compiler/xla/types.h"
#include "tensorflow/compiler/xla/xla_data.pb.h"

namespace xla {

Computation CreateScalarAddComputation(PrimitiveType type,
                                       ComputationBuilder* builder) {
  const Shape scalar = ShapeUtil::MakeShape(type, {});
  auto b = builder->CreateSubBuilder("add_" + PrimitiveType_Name(type));
  auto lhs = b->Parameter(0, scalar, "lhs");
  auto rhs = b->Parameter(1, scalar, "rhs");
  b->Add(lhs, rhs);
  return b->BuildAndNoteError();
}

Computation CreateScalarGeComputation(PrimitiveType type,
                                      ComputationBuilder* builder) {
  const Shape scalar = ShapeUtil::MakeShape(type, {});
  auto b = builder->CreateSubBuilder("ge_" + PrimitiveType_Name(type));
  auto lhs = b->Parameter(0, scalar, "lhs");
  auto rhs = b->Parameter(1, scalar, "rhs");
  b->Ge(lhs, rhs);
  return b->BuildAndNoteError();
}

Computation CreateScalarMaxComputation(PrimitiveType type,
                                       ComputationBuilder* builder) {
  const Shape scalar = ShapeUtil::MakeShape(type, {});
  auto b = builder->CreateSubBuilder("max_" + PrimitiveType_Name(type));
  auto lhs = b->Parameter(0, scalar, "lhs");
  auto rhs = b->Parameter(1, scalar, "rhs");
  b->Max(lhs, rhs);
  return b->BuildAndNoteError();
}

Computation CreateScalarMinComputation(PrimitiveType type,
                                       ComputationBuilder* builder) {
  const Shape scalar = ShapeUtil::MakeShape(type, {});
  auto b = builder->CreateSubBuilder("min_" + PrimitiveType_Name(type));
  auto lhs = b->Parameter(0, scalar, "lhs");
  auto rhs = b->Parameter(1, scalar, "rhs");
  b->Min(lhs, rhs);
  return b->BuildAndNoteError();
}

Computation CreateScalarLogicalAndComputation(ComputationBuilder* builder) {
  const Shape scalar = ShapeUtil::MakeShape(PRED, {});
  auto b = builder->CreateSubBuilder("logical_and");
  auto lhs = b->Parameter(0, scalar, "lhs");
  auto rhs = b->Parameter(1, scalar, "rhs");
  b->LogicalAnd(lhs, rhs);
  return b->BuildAndNoteError();
}

Computation CreateScalarLogicalOrComputation(ComputationBuilder* builder) {
  const Shape scalar = ShapeUtil::MakeShape(PRED, {});
  auto b = builder->CreateSubBuilder("logical_or");
  auto lhs = b->Parameter(0, scalar, "lhs");
  auto rhs = b->Parameter(1, scalar, "rhs");
  b->LogicalOr(lhs, rhs);
  return b->BuildAndNoteError();
}

StatusOr<ComputationDataHandle> Any(const ComputationDataHandle& predicates,
                                    ComputationBuilder* builder) {
  auto f = builder->ConstantR0<bool>(false);
  Computation logical_or = CreateScalarLogicalOrComputation(builder);
  TF_ASSIGN_OR_RETURN(std::unique_ptr<Shape> predicates_shape,
                      builder->GetShape(predicates));
  std::vector<int64> all_dimensions(ShapeUtil::Rank(*predicates_shape));
  std::iota(all_dimensions.begin(), all_dimensions.end(), 0);
  return builder->Reduce(predicates, f, logical_or, all_dimensions);
}

}  // namespace xla
