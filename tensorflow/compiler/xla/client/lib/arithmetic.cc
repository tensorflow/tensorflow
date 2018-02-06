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
#include "tensorflow/core/lib/strings/strcat.h"

namespace xla {
namespace {
using InstructionGenerator =
    ComputationDataHandle (*)(ComputationBuilder*, const ComputationDataHandle&,
                              const ComputationDataHandle&);

Computation CreateScalarComputation(const string& name, PrimitiveType type,
                                    ComputationBuilder* builder,
                                    InstructionGenerator generator) {
  std::unique_ptr<ComputationBuilder> b;
  if (type == PRED) {
    b = builder->CreateSubBuilder(name);
  } else {
    b = builder->CreateSubBuilder(
        tensorflow::strings::StrCat(name, "_", PrimitiveType_Name(type)));
  }

  const Shape scalar = ShapeUtil::MakeShape(type, {});
  auto lhs = b->Parameter(0, scalar, "lhs");
  auto rhs = b->Parameter(1, scalar, "rhs");
  generator(b.get(), lhs, rhs);
  return b->BuildAndNoteError();
}
}  // namespace

Computation CreateScalarAddComputation(PrimitiveType type,
                                       ComputationBuilder* builder) {
  return CreateScalarComputation(
      "add", type, builder,
      [](ComputationBuilder* b, const ComputationDataHandle& lhs,
         const ComputationDataHandle& rhs) { return b->Add(lhs, rhs); });
}

Computation CreateScalarMultiplyComputation(PrimitiveType type,
                                            ComputationBuilder* builder) {
  return CreateScalarComputation(
      "add", type, builder,
      [](ComputationBuilder* b, const ComputationDataHandle& lhs,
         const ComputationDataHandle& rhs) { return b->Mul(lhs, rhs); });
}

Computation CreateScalarGeComputation(PrimitiveType type,
                                      ComputationBuilder* builder) {
  return CreateScalarComputation(
      "ge", type, builder,
      [](ComputationBuilder* b, const ComputationDataHandle& lhs,
         const ComputationDataHandle& rhs) { return b->Ge(lhs, rhs); });
}

Computation CreateScalarMaxComputation(PrimitiveType type,
                                       ComputationBuilder* builder) {
  return CreateScalarComputation(
      "max", type, builder,
      [](ComputationBuilder* b, const ComputationDataHandle& lhs,
         const ComputationDataHandle& rhs) { return b->Max(lhs, rhs); });
}

Computation CreateScalarMinComputation(PrimitiveType type,
                                       ComputationBuilder* builder) {
  return CreateScalarComputation(
      "min", type, builder,
      [](ComputationBuilder* b, const ComputationDataHandle& lhs,
         const ComputationDataHandle& rhs) { return b->Min(lhs, rhs); });
}

Computation CreateScalarAndComputation(ComputationBuilder* builder) {
  return CreateScalarComputation(
      "and", PRED, builder,
      [](ComputationBuilder* b, const ComputationDataHandle& lhs,
         const ComputationDataHandle& rhs) { return b->And(lhs, rhs); });
}

Computation CreateScalarOrComputation(ComputationBuilder* builder) {
  return CreateScalarComputation(
      "or", PRED, builder,
      [](ComputationBuilder* b, const ComputationDataHandle& lhs,
         const ComputationDataHandle& rhs) { return b->Or(lhs, rhs); });
}

StatusOr<ComputationDataHandle> Any(const ComputationDataHandle& predicates,
                                    ComputationBuilder* builder) {
  auto f = builder->ConstantR0<bool>(false);
  Computation logical_or = CreateScalarOrComputation(builder);
  TF_ASSIGN_OR_RETURN(std::unique_ptr<Shape> predicates_shape,
                      builder->GetShape(predicates));
  std::vector<int64> all_dimensions(ShapeUtil::Rank(*predicates_shape));
  std::iota(all_dimensions.begin(), all_dimensions.end(), 0);
  return builder->Reduce(predicates, f, logical_or, all_dimensions);
}

}  // namespace xla
