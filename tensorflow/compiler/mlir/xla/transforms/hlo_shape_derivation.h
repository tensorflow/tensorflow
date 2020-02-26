/* Copyright 2020 The TensorFlow Authors. All Rights Reserved.

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

#ifndef TENSORFLOW_COMPILER_MLIR_XLA_TRANSFORMS_HLO_SHAPE_DERIVATION_H_
#define TENSORFLOW_COMPILER_MLIR_XLA_TRANSFORMS_HLO_SHAPE_DERIVATION_H_

#include "mlir/Dialect/StandardOps/IR/Ops.h"  // TF:llvm-project
#include "mlir/IR/Attributes.h"  // TF:llvm-project
#include "mlir/IR/Builders.h"  // TF:llvm-project
#include "mlir/IR/Location.h"  // TF:llvm-project
#include "mlir/IR/MLIRContext.h"  // TF:llvm-project
#include "mlir/IR/Operation.h"  // TF:llvm-project
#include "mlir/Transforms/DialectConversion.h"  // TF:llvm-project
#include "tensorflow/compiler/mlir/xla/ir/hlo_ops.h"

namespace mlir {
namespace xla_hlo {

// This file contains implementations for shape derivation functions that,
// given some operation and a result number, produce IR that computes the
// shape of the given result at runtime based on operands of the provided
// operation.
// These should be generated at some point based on annotations on the HLO
// using the new shape dialect. While this is still in the works, we hardcode
// the expected IR here to unblock progress.
// The implementation is based on templates to allow for using these derivation
// functions in templated code.

namespace impl {

struct UnknownShape {
  // Default shape derivation function that simply fails with a runtime error.
  static Value deriveShapeFromOp(Operation* op, int operand_position,
                                 ConversionPatternRewriter* rewriter) {
    op->emitOpError()
        << "dynamic result shapes cannot be derived for this operation";
    return {};
  }
};

struct SameShapeAsFirstOperand {
  // Shape derivation function that computes the shape of the result based on
  // the first argument. For a 2-dimensional input tensor, this produces IR of
  // the form
  //
  //  %0 = dim %arg0, 0 : memref<?x?xf32>
  //  %1 = index_cast %0 : index to i64
  //  %2 = dim %arg0, 1 : memref<?x?xf32>
  //  %3 = index_cast %2 : index to i64
  //  %4 = "xla_hlo.scalars_to_dimension_tensor"(%1, %3)
  //    : (i64, i64) -> tensor<2xi64>
  //
  // and returns %4 as the shape value.
  static Value deriveShapeFromOp(Operation* op, int result_postion,
                                 ConversionPatternRewriter* rewriter) {
    Value operand = op->getOperand(0);
    ShapedType operand_type = operand.getType().dyn_cast<ShapedType>();
    if (!operand_type) {
      op->emitOpError() << "first operand has no shaped type";
      return {};
    }
    auto loc = op->getLoc();
    SmallVector<Value, 4> shape_values;
    shape_values.reserve(operand_type.getRank());
    auto shape_scalar_type = rewriter->getIntegerType(64);
    for (auto element : llvm::enumerate(operand_type.getShape())) {
      if (element.value() == ShapedType::kDynamicSize) {
        Value dim = rewriter->create<DimOp>(loc, operand, element.index());
        shape_values.push_back(
            rewriter->create<IndexCastOp>(loc, dim, shape_scalar_type));
      } else {
        shape_values.push_back(rewriter->create<ConstantOp>(
            loc, rewriter->getI64IntegerAttr(element.value())));
      }
    }
    return rewriter->create<ScalarsToDimensionTensorOp>(
        loc, RankedTensorType::get({operand_type.getRank()}, shape_scalar_type),
        shape_values);
  }
};

}  // namespace impl

// Default template to cover HLO operations whose shape derivation is unknown.
template <typename HloOpTy>
struct ShapeDerivation {
  using impl = impl::UnknownShape;
};

// Element-wise operations that have the shape of their first operand.

#define SAME_SHAPE_AS_FIRST_OPERAND(Op)         \
  template <>                                   \
  struct ShapeDerivation<Op> {                  \
    using impl = impl::SameShapeAsFirstOperand; \
  };

SAME_SHAPE_AS_FIRST_OPERAND(AbsOp)
SAME_SHAPE_AS_FIRST_OPERAND(AddOp)
SAME_SHAPE_AS_FIRST_OPERAND(AndOp)
SAME_SHAPE_AS_FIRST_OPERAND(CeilOp)
SAME_SHAPE_AS_FIRST_OPERAND(CosOp)
SAME_SHAPE_AS_FIRST_OPERAND(DivOp)
SAME_SHAPE_AS_FIRST_OPERAND(ExpOp)
SAME_SHAPE_AS_FIRST_OPERAND(MaxOp)
SAME_SHAPE_AS_FIRST_OPERAND(MinOp)
SAME_SHAPE_AS_FIRST_OPERAND(MulOp)
SAME_SHAPE_AS_FIRST_OPERAND(NegOp)
SAME_SHAPE_AS_FIRST_OPERAND(RemOp)
SAME_SHAPE_AS_FIRST_OPERAND(SubOp)
SAME_SHAPE_AS_FIRST_OPERAND(TanhOp)

#undef SAME_SHAPE_AS_FIRST_OPERAND

}  // namespace xla_hlo
}  // namespace mlir

#endif  // TENSORFLOW_COMPILER_MLIR_XLA_TRANSFORMS_HLO_SHAPE_DERIVATION_H_
