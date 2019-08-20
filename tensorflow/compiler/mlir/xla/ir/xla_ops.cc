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

// This file defines the operations used in the XLA dialect.

#include "tensorflow/compiler/mlir/xla/ir/xla_ops.h"

#include <assert.h>
#include <stddef.h>
#include <stdint.h>

#include "llvm/ADT/APFloat.h"
#include "llvm/ADT/APInt.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/FormatVariadic.h"
#include "mlir/IR/Attributes.h"  // TF:local_config_mlir
#include "mlir/IR/Builders.h"  // TF:local_config_mlir
#include "mlir/IR/Dialect.h"  // TF:local_config_mlir
#include "mlir/IR/Location.h"  // TF:local_config_mlir
#include "mlir/IR/MLIRContext.h"  // TF:local_config_mlir
#include "mlir/IR/OpDefinition.h"  // TF:local_config_mlir
#include "mlir/IR/OpImplementation.h"  // TF:local_config_mlir
#include "mlir/IR/Operation.h"  // TF:local_config_mlir
#include "mlir/IR/OperationSupport.h"  // TF:local_config_mlir
#include "mlir/IR/PatternMatch.h"  // TF:local_config_mlir
#include "mlir/IR/StandardTypes.h"  // TF:local_config_mlir
#include "mlir/IR/TypeUtilities.h"  // TF:local_config_mlir
#include "mlir/IR/Types.h"  // TF:local_config_mlir
#include "mlir/IR/Value.h"  // TF:local_config_mlir
#include "tensorflow/compiler/mlir/xla/ir/xla_ops.h.inc"

using namespace mlir;
using namespace mlir::XLA;

XlaHloDialect::XlaHloDialect(MLIRContext* context)
    : Dialect(getDialectNamespace(), context) {
  addOperations<
#define GET_OP_LIST
#include "tensorflow/compiler/mlir/xla/ir/xla_ops.cc.inc"
      >();

  // Support unknown operations because not all XLA operations are registered.
  allowUnknownOperations();
}

Operation* XlaHloDialect::materializeConstant(OpBuilder& builder,
                                              Attribute value, Type type,
                                              Location loc) {
  // If this is an opaque elements attribute, then generate an xla_hlo.constant.
  if (value.isa<OpaqueElementsAttr>())
    return builder.create<XLA::ConstOp>(loc, type, value.cast<ElementsAttr>());
  return nullptr;
}

#define GET_OP_CLASSES
#include "tensorflow/compiler/mlir/xla/ir/xla_ops.cc.inc"

//===----------------------------------------------------------------------===//
// ConstOp
//===----------------------------------------------------------------------===//

OpFoldResult ConstOp::fold(ArrayRef<Attribute> operands) {
  assert(operands.empty() && "constant has no operands");

  // Return the held attribute value.
  return value();
}

// Builds a constant op with the specified attribute `value`.
void ConstOp::build(Builder* builder, OperationState* result, Attribute value) {
  Type type;
  if (auto elemAttr = value.dyn_cast<ElementsAttr>()) {
    type = elemAttr.getType();
  } else if (value.isa<BoolAttr>() || value.isa<FloatAttr>() ||
             value.isa<IntegerAttr>()) {
    // All XLA types must be tensor types. In the build() method, we want to
    // provide more flexiblity by allowing attributes of scalar types. But we
    // need to wrap it up with ElementsAttr to construct valid XLA constants.
    type = RankedTensorType::get(/*shape=*/{}, value.getType());
    value = DenseElementsAttr::get(type.cast<TensorType>(), value);
  }

  // TODO: support other XLA specific types.
  assert(type && "unsupported attribute type for building xla_hlo.constant");
  result->types.push_back(type);
  result->addAttribute("value", value);
}

//===----------------------------------------------------------------------===//
// ConvertOp
//===----------------------------------------------------------------------===//

OpFoldResult ConvertOp::fold(ArrayRef<Attribute> operands) {
  assert(operands.size() == 1 && "convert must take one operand");
  auto operand = operands[0];

  if (!operand) return {};

  if (auto elementsAttr = operand.dyn_cast<ElementsAttr>()) {
    auto inType = elementsAttr.getType();
    auto outType = getResult()->getType().cast<ShapedType>();

    if (inType == outType) {
      return operand;
    }

    auto inElement = inType.getElementType();
    auto outElement = outType.getElementType();
    size_t bitWidth =
        outElement.isBF16() ? 64 : outElement.getIntOrFloatBitWidth();

    if (inElement.isa<FloatType>()) {
      if (outElement.isa<IntegerType>()) {
        auto func = [&](const APFloat& floatValue) -> APInt {
          return APInt(bitWidth, FloatAttr::getValueAsDouble(floatValue));
        };
        llvm::function_ref<APInt(const APFloat&)> func_ref = func;
        return elementsAttr.mapValues(outType.getElementType(), func_ref);
      }

      if (outElement.isa<FloatType>()) {
        auto& semantics = outElement.cast<FloatType>().getFloatSemantics();
        auto func = [&](const APFloat& floatValue) -> APInt {
          APFloat newDouble(FloatAttr::getValueAsDouble(floatValue));
          bool losesInfo = false;
          newDouble.convert(semantics, llvm::APFloat::rmNearestTiesToEven,
                            &losesInfo);
          return newDouble.bitcastToAPInt();
        };
        llvm::function_ref<APInt(const APFloat&)> func_ref = func;
        return elementsAttr.mapValues(outType.getElementType(), func_ref);
      }
    }

    if (inElement.isa<IntegerType>()) {
      if (outElement.isa<IntegerType>()) {
        auto func = [&](const APInt& val) -> APInt {
          return APInt(bitWidth, val.getLimitedValue());
        };
        llvm::function_ref<APInt(const APInt&)> func_ref = func;
        return elementsAttr.mapValues(outType.getElementType(), func_ref);
      }

      if (outElement.isa<FloatType>()) {
        auto& semantics = outElement.cast<FloatType>().getFloatSemantics();
        auto func = [&](const APInt& val) -> APInt {
          APFloat newDouble(static_cast<double>(val.getLimitedValue()));
          bool losesInfo = false;
          newDouble.convert(semantics, llvm::APFloat::rmNearestTiesToEven,
                            &losesInfo);
          return newDouble.bitcastToAPInt();
        };
        llvm::function_ref<APInt(const APInt&)> func_ref = func;
        return elementsAttr.mapValues(outType.getElementType(), func_ref);
      }
    }
  }

  return {};
}

//===----------------------------------------------------------------------===//
// IotaOp
//===----------------------------------------------------------------------===//

OpFoldResult IotaOp::fold(ArrayRef<Attribute> operands) {
  const auto output_type = getResult()->getType().cast<ShapedType>();
  const auto output_size = output_type.getNumElements();
  const auto dimension = iota_dimension().getLimitedValue();
  const auto max_dim_size = output_type.getDimSize(dimension);
  int bitwidth = output_type.getElementType().getIntOrFloatBitWidth();

  llvm::SmallVector<APInt, 10> values;
  values.reserve(output_size);

  int64_t increase_stride = output_size;
  for (int i = 0; i <= dimension; i++) {
    increase_stride /= output_type.getDimSize(i);
  }

  int64_t current_value = 0;
  for (int i = 0; i < output_size; i++) {
    int64_t value = (current_value / increase_stride) % max_dim_size;
    values.push_back(APInt(bitwidth, value));
    ++current_value;
  }

  return DenseIntElementsAttr::get(output_type, values);
}

//===----------------------------------------------------------------------===//
// ReshapeOp
//===----------------------------------------------------------------------===//

OpFoldResult ReshapeOp::fold(ArrayRef<Attribute> operands) {
  if (getOperand()->getType() == getType()) {
    return getOperand();
  }

  if (auto prev_op =
          dyn_cast_or_null<ReshapeOp>(getOperand()->getDefiningOp())) {
    setOperand(prev_op.getOperand());
    return getResult();
  }

  if (auto elements = operands.front().dyn_cast_or_null<DenseElementsAttr>()) {
    return elements.reshape(getResult()->getType().cast<ShapedType>());
  }

  return {};
}

//===----------------------------------------------------------------------===//
// TransposeOp
//===----------------------------------------------------------------------===//

OpFoldResult TransposeOp::fold(ArrayRef<Attribute> operands) {
  for (auto it : llvm::enumerate(permutation().cast<DenseIntElementsAttr>())) {
    if (it.index() != it.value()) {
      return {};
    }
  }
  return getOperand();
}
