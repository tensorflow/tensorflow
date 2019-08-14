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

#include "mlir/IR/OpDefinition.h"  // TF:local_config_mlir
#include "mlir/IR/StandardTypes.h"  // TF:local_config_mlir
#include "mlir/IR/TypeUtilities.h"  // TF:local_config_mlir
#include "mlir/Support/LogicalResult.h"  // TF:local_config_mlir
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_types.h"

namespace mlir {
namespace OpTrait {
namespace TF {

// Verifies if 'ref_type' is a REF type corresponding to 'type'.
static inline LogicalResult VerifyRefTypeMatch(mlir::Type type,
                                               mlir::Type ref_type) {
  auto ref_type_kind = ref_type.getKind();
  switch (type.getKind()) {
    case mlir::StandardTypes::F16:
      return success(ref_type_kind == mlir::TF::TensorFlowTypes::HALF_REF);
    case mlir::StandardTypes::F32:
      return success(ref_type_kind == mlir::TF::TensorFlowTypes::FLOAT_REF);
    case mlir::StandardTypes::F64:
      return success(ref_type_kind == mlir::TF::TensorFlowTypes::DOUBLE_REF);
    case mlir::StandardTypes::BF16:
      return success(ref_type_kind == mlir::TF::TensorFlowTypes::BFLOAT16_REF);
    case mlir::StandardTypes::Integer: {
      const auto& itype = type.cast<mlir::IntegerType>();
      switch (itype.getWidth()) {
        case 1:
          return success(ref_type_kind == mlir::TF::TensorFlowTypes::BOOL_REF);
        case 8:
          return success(ref_type_kind == mlir::TF::TensorFlowTypes::INT8_REF);
        case 16:
          return success(ref_type_kind == mlir::TF::TensorFlowTypes::INT16_REF);
        case 32:
          return success(ref_type_kind == mlir::TF::TensorFlowTypes::INT32_REF);
        case 64:
          return success(ref_type_kind == mlir::TF::TensorFlowTypes::INT64_REF);
        default:
          return failure();
      }
    }
#define HANDLE_TF_TYPE(tftype, enumerant, name) \
  case mlir::TF::TensorFlowTypes::enumerant:    \
    return success(ref_type_kind == mlir::TF::TensorFlowTypes::enumerant##_REF);

#define HANDLE_TF_REF_TYPE(tftype, enumerant, name)
// NOLINTNEXTLINE
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_types.def"
    default:
      return failure();
  }
}

// This class provides verification for ops that are known to have the same
// result types and all operands are either of the same type as result or a REF
// type corresponding to the result type.
template <typename ConcreteType>
class OperandsSameAsResultsTypeOrRef
    : public TraitBase<ConcreteType, OperandsSameAsResultsTypeOrRef> {
 public:
  static LogicalResult verifyTrait(Operation* op) {
    LogicalResult shapeMatch = impl::verifySameOperandsAndResultShape(op);
    if (failed(shapeMatch)) return shapeMatch;

    auto type = getElementTypeOrSelf(op->getResult(0)->getType());

    // Verify that the first result type is same as the rest of the results.
    // We skip the comparison against itself.
    for (auto resultType : llvm::drop_begin(op->getResultTypes(), 1)) {
      resultType = getElementTypeOrSelf(resultType);
      if (resultType != type)
        return op->emitOpError() << "requires the same type for all results";
    }

    for (auto opType : op->getOperandTypes()) {
      opType = getElementTypeOrSelf(opType);
      if (opType != type && failed(VerifyRefTypeMatch(type, opType))) {
        return op->emitError() << "requires all operands to be either same "
                                  "as or ref type of results";
      }
    }
    return success();
  }
};

}  // namespace TF
}  // namespace OpTrait
}  // namespace mlir

#endif  // TENSORFLOW_COMPILER_MLIR_TENSORFLOW_IR_TF_TRAITS_H_
