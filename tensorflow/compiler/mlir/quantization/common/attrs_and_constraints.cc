/* Copyright 2022 The TensorFlow Authors. All Rights Reserved.

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
#include "tensorflow/compiler/mlir/quantization/common/attrs_and_constraints.h"

#include <cstdint>

#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/MathExtras.h"
#include "mlir/IR/Builders.h"  // from @llvm-project
#include "mlir/IR/BuiltinAttributes.h"  // from @llvm-project
#include "mlir/IR/BuiltinTypeInterfaces.h"  // from @llvm-project
#include "mlir/IR/Operation.h"  // from @llvm-project
#include "mlir/IR/Types.h"  // from @llvm-project
#include "mlir/IR/Value.h"  // from @llvm-project
#include "mlir/Support/LLVM.h"  // from @llvm-project
#include "mlir/Support/LogicalResult.h"  // from @llvm-project
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_ops.h"
#include "tensorflow/compiler/mlir/tensorflow/utils/xla_call_module_attrs.h"

namespace mlir::quant {

bool HasStaticShape(Value value) {
  auto shaped_type = value.getType().dyn_cast<ShapedType>();
  if (!shaped_type) return false;

  return shaped_type.hasStaticShape();
}

bool HasStaticShapeAtDims(Value value, const ArrayRef<int> dims) {
  auto shaped_type = value.getType().dyn_cast<ShapedType>();
  if (!shaped_type || !shaped_type.hasRank()) return false;

  for (auto dim : dims) {
    if (shaped_type.isDynamicDim(dim)) return false;
  }
  return true;
}

Type CloneTypeWithNewElementType(Type old_type, Type element_type) {
  if (!old_type.isa<ShapedType>()) return {};

  return old_type.cast<ShapedType>().clone(element_type);
}

SmallVector<Value> CloneOpWithReplacedOperands(
    OpBuilder& builder, Operation* op, const ArrayRef<Value> new_operands) {
  IRMapping mapping;
  for (const auto& arg : enumerate(new_operands)) {
    mapping.map(op->getOperand(arg.index()), arg.value());
  }
  return builder.clone(*op, mapping)->getResults();
}

FailureOr<int32_t> CastI64ToI32(const int64_t value) {
  if (!llvm::isInt<32>(value)) {
    DEBUG_WITH_TYPE(
        "mlir-quant-attrs-and-constraints",
        llvm::dbgs()
            << "Tried to cast " << value
            << "from int64 to int32, but lies out of range of int32.\n");
    return failure();
  }
  return static_cast<int32_t>(value);
}

FailureOr<SmallVector<int32_t>> CastI64ArrayToI32(
    const ArrayRef<int64_t> int64_array) {
  SmallVector<int32_t> int32_array{};
  int32_array.reserve(int64_array.size());

  for (const int64_t i64 : int64_array) {
    FailureOr<int32_t> cast_i32 = CastI64ToI32(i64);
    if (failed(cast_i32)) return failure();

    int32_array.push_back(*cast_i32);
  }
  return int32_array;
}

StringRef GetEntryFunctionName(TF::XlaCallModuleOp op) {
  if (!op->hasAttrOfType<FlatSymbolRefAttr>(
          TF::kStablehloEntryFunctionAttrName)) {
    return StringRef();
  }
  return op
      ->getAttrOfType<FlatSymbolRefAttr>(TF::kStablehloEntryFunctionAttrName)
      .getValue();
}

}  // namespace mlir::quant
