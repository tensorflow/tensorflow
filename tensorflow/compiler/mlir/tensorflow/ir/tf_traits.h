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

// This file defines the op traits used in the MLIR TensorFlow dialect.

#ifndef TENSORFLOW_COMPILER_MLIR_TENSORFLOW_IR_TF_TRAITS_H_
#define TENSORFLOW_COMPILER_MLIR_TENSORFLOW_IR_TF_TRAITS_H_

#include "mlir/IR/BuiltinTypes.h"  // from @llvm-project
#include "mlir/IR/OpDefinition.h"  // from @llvm-project
#include "mlir/IR/TypeUtilities.h"  // from @llvm-project
#include "mlir/Interfaces/SideEffectInterfaces.h"  // from @llvm-project
#include "mlir/Support/LogicalResult.h"  // from @llvm-project
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_op_interfaces.h"
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_types.h"

namespace mlir {
namespace OpTrait {
namespace TF {

// Verifies if 'ref_type' is a REF type corresponding to 'type'.
static inline LogicalResult VerifyRefTypeMatch(mlir::Type type,
                                               mlir::Type maybe_ref_type) {
  if (auto ref_type =
          maybe_ref_type.dyn_cast<mlir::tf_type::TensorFlowRefType>())
    return success(ref_type.RemoveRef().getTypeID() == type.getTypeID());
  return failure();
}

// This class provides verification for ops that are known to have the same
// result types and all operands are either of the same type as result or a REF
// type corresponding to the result type.
// TODO(jpienaar): Update the name and the description.
template <typename ConcreteType>
class OperandsSameAsResultsTypeOrRef
    : public TraitBase<ConcreteType, OperandsSameAsResultsTypeOrRef> {
 public:
  static LogicalResult verifyTrait(Operation* op) {
    LogicalResult shapeMatch = impl::verifySameOperandsAndResultShape(op);
    if (failed(shapeMatch)) return shapeMatch;
    Type type = op->getResult(0).getType();
    // Verify that the first result type is same as the rest of the results.
    // We skip the comparison against itself.
    for (auto result_type : llvm::drop_begin(op->getResultTypes(), 1)) {
      if (!mlir::tf_type::HasCompatibleElementTypes(type, result_type))
        return op->emitOpError()
               << "requires all return types to have compatible element types";
    }
    for (auto operand_type : op->getOperandTypes()) {
      if (!mlir::tf_type::HasCompatibleElementTypes(
              operand_type, type, /*may_ignore_ref_type_lhs=*/true))
        return op->emitError() << "requires all operands and results to have "
                                  "compatible element types";
    }
    return success();
  }
};

namespace detail {
inline LogicalResult verifySameOperandsAndResultElementTypeResolveRef(
    Operation* op) {
  Type element_type;
  if (op->getNumResults() > 0) {
    element_type = mlir::tf_type::GetElementTypeOrSelfResolveRef(
        op->getResult(0).getType());
  } else if (op->getNumOperands() > 0) {
    element_type = mlir::tf_type::GetElementTypeOrSelfResolveRef(
        op->getOperand(0).getType());
  } else {
    // Nothing to check.
    return success();
  }
  // Verify that all result element types are compatible to `element_type`.
  for (const auto& result_type : op->getResultTypes()) {
    if (mlir::tf_type::GetElementTypeOrSelfResolveRef(result_type) !=
        element_type) {
      return op->emitOpError(
          "requires compatible element types for all operands and results");
    }
  }
  // Verify that all operand element types are compatible to `element_type`.
  for (const auto& operand_type : op->getOperandTypes()) {
    if (mlir::tf_type::GetElementTypeOrSelfResolveRef(operand_type) !=
        element_type) {
      return op->emitOpError(
          "requires compatible element types for all operands and results");
    }
  }
  return success();
}
}  // namespace detail

// Verifies that op has the same operand and result element types (or type
// itself, if scalar) after resolving reference types (i.e., after converting
// reference types to their corresponding TensorFlow or standard types).
template <typename ConcreteType>
class SameOperandsAndResultElementTypeResolveRef
    : public TraitBase<ConcreteType,
                       SameOperandsAndResultElementTypeResolveRef> {
 public:
  static LogicalResult verifyTrait(Operation* op) {
    return detail::verifySameOperandsAndResultElementTypeResolveRef(op);
  }
};

// Verifies that op has the same operand and result types after resolving
// reference types (i.e., after converting reference types to their
// corresponding TensorFlow or standard types).
template <typename ConcreteType>
class SameOperandsAndResultTypeResolveRef
    : public TraitBase<ConcreteType, SameOperandsAndResultTypeResolveRef> {
 public:
  static LogicalResult verifyTrait(Operation* op) {
    if (failed(impl::verifySameOperandsAndResultShape(op))) return failure();
    return detail::verifySameOperandsAndResultElementTypeResolveRef(op);
  }
};

// Layout agnostic operations do not depend on the operands data layout (data
// format), as and example all element wise operations are layout agnostic.
template <typename ConcreteType>
class LayoutAgnostic : public TraitBase<ConcreteType, LayoutAgnostic> {};

// Trait to indicate operations that cannot be duplicated as they might carry
// certain state around within their implementations.
template <typename ConcreteType>
class CannotDuplicate : public TraitBase<ConcreteType, CannotDuplicate> {
 public:
  static LogicalResult verifyTrait(Operation* op) {
    if (MemoryEffectOpInterface::hasNoEffect(op))
      return op->emitError(
          "operations with no side effects cannot have CannotDuplicate trait");
    return success();
  }
};

// Trait to indicate an operation cannot be constant folded.
template <typename ConcreteType>
class NoConstantFold : public TraitBase<ConcreteType, NoConstantFold> {};

// Coefficient-wise binary operation with implicit broadcasting support, for
// example tf.Sub operation.
template <typename ConcreteType>
class CwiseBinary : public TraitBase<ConcreteType, CwiseBinary> {};

// Coefficient-wise unary operation, for example tf.Sqrt operation.
template <typename ConcreteType>
class CwiseUnary : public TraitBase<ConcreteType, CwiseUnary> {};

// Indicates that any returned resource is unique.
template <typename ConcreteType>
class UniqueResourceAllocation
    : public TraitBase<ConcreteType, UniqueResourceAllocation> {
 public:
  // Implements method required for `ResourceHandleAllocatorInterface`.
  llvm::SmallVector<mlir::TF::ResourceHandleValueAndId>
  GetResourceHandleValueAndIdList(
      llvm::SmallDenseMap<mlir::TF::ResourceHandle, int64_t>&
          resource_handle_id_map,
      int64_t& next_id) {
    llvm::SmallVector<mlir::TF::ResourceHandleValueAndId> resource_vec;
    for (Value resource :
         mlir::tf_type::filter_resources(this->getOperation()->getResults())) {
      resource_vec.push_back({resource, next_id++});
    }
    return resource_vec;
  }
};

}  // namespace TF
}  // namespace OpTrait
}  // namespace mlir

#endif  // TENSORFLOW_COMPILER_MLIR_TENSORFLOW_IR_TF_TRAITS_H_
