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

#ifndef TENSORFLOW_COMPILER_MLIR_XLA_IR_HLO_UTILS_H_
#define TENSORFLOW_COMPILER_MLIR_XLA_IR_HLO_UTILS_H_

#include "mlir/IR/Attributes.h"  // TF:local_config_mlir
#include "mlir/IR/Builders.h"  // TF:local_config_mlir
#include "mlir/IR/PatternMatch.h"  // TF:local_config_mlir
#include "mlir/IR/StandardTypes.h"  // TF:local_config_mlir
#include "mlir/IR/TypeUtilities.h"  // TF:local_config_mlir
#include "tensorflow/compiler/mlir/xla/convert_op_folder.h"

namespace mlir {
namespace xla {

// Computes the broadcast dimensions attr for an elementwise binary operator
// between two ranked tensors.
mlir::DenseIntElementsAttr getBroadcastDimensionsAttr(mlir::Builder* b,
                                                      mlir::Value x,
                                                      mlir::Value y);

/// Get a constant splat for the given value type.
template <typename T>
static ElementsAttr getSplat(Builder* b, Value val, T constant) {
  auto valType = val->getType().cast<TensorType>();
  auto valElementType = getElementTypeOrSelf(val->getType());

  // Handle integer elements.
  Attribute elementAttr;
  if (valElementType.isa<IntegerType>())
    elementAttr = b->getIntegerAttr(valElementType, constant);
  else if (valElementType.isa<FloatType>())
    elementAttr = b->getFloatAttr(valElementType, constant);
  else
    llvm_unreachable("unhandled element type");

  return DenseElementsAttr::get(valType, elementAttr);
}

// Returns DenseElementsAttr of rank zero with the given element type and the
// value.
// Requires `ty` to be either FloatType of IntegerType.
DenseElementsAttr GetScalarOfType(Type ty, int64_t raw_value);

}  // namespace xla
}  // namespace mlir

#endif  // TENSORFLOW_COMPILER_MLIR_XLA_IR_HLO_UTILS_H_
