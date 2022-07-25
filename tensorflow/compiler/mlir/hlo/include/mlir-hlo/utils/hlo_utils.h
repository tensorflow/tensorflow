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

#ifndef MLIR_HLO_UTILS_HLO_UTILS_H
#define MLIR_HLO_UTILS_HLO_UTILS_H

#include <string>

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/TypeUtilities.h"

namespace mlir {
namespace hlo {
// Computes the broadcast dimensions attr for an elementwise binary operator
// between two ranked tensors.
// If `allow_empty` is true, then null can be returned to mean that the
// broadcast is an "identity".
mlir::DenseIntElementsAttr getBroadcastDimensionsAttr(mlir::Builder* b,
                                                      mlir::Value x,
                                                      mlir::Value y,
                                                      bool allowEmpty = true);

// Get a constant splat for the given value of type. Requires value to be of
// type static shaped RankedTensorType.
template <typename T>
static ElementsAttr getSplat(Builder* b, RankedTensorType ty, T constant) {
  Type elementTy = getElementTypeOrSelf(ty);

  if (elementTy.isSignlessInteger())
    return DenseElementsAttr::get(ty, b->getIntegerAttr(elementTy, constant));

  if (elementTy.isa<FloatType>())
    return DenseElementsAttr::get(ty, b->getFloatAttr(elementTy, constant));

  if (auto complexTy = elementTy.dyn_cast<ComplexType>()) {
    auto complexElementTy = complexTy.getElementType();
    if (complexElementTy.isF32())
      return DenseElementsAttr::get(ty,
                                    static_cast<std::complex<float>>(constant));
    if (complexElementTy.isF64())
      return DenseElementsAttr::get(
          ty, static_cast<std::complex<double>>(constant));
  }
  llvm_unreachable("unhandled element type");
}

template <typename T>
static ElementsAttr getSplat(Builder* b, Value val, T constant) {
  return getSplat(b, val.getType().cast<RankedTensorType>(), constant);
}

// Returns DenseElementsAttr of rank zero with the given element type and the
// value.
// Requires `ty` to be either FloatType, IntegerType, or ComplexType.
DenseElementsAttr getScalarOfType(Type ty, int64_t rawValue);

// Returns DenseElementsAttr of rank zero with the given element type and the
// value which is the neutral element for additions.
// Requires `ty` to be either FloatType, IntegerType, or ComplexType.
DenseElementsAttr getScalarNegZeroOfType(Type ty);

// Enum type used to specify scalar argument to GetScalarLimitOfType.
enum ScalarLimit {
  kLowest,          // The scalar corresponding to numeric_limits<T>::lowest.
  kInfinityLowest,  // Like kLowest, but returns -infinity where available.
  kMax,             // The scalar corresponding to numeric_limits<T>::max.
  kInfinityMax,     // Like kMax, but returns infinity where available.
};

// Returns a scalar limit value for the given type.
//
// The argument 'limit' describes which scalar value to return.
//
// Requires `ty` to be either FloatType or IntegerType.
DenseElementsAttr getScalarLimitOfType(Type ty, ScalarLimit limit);

// Given `op_name` from LMHLO, returns the corresponding op name in MHLO.
// Returns empty string if no such op exists.
std::string lmhloToMhloOpName(llvm::StringRef opName,
                              mlir::MLIRContext* context);

// Return true if Attr has values [0, 1, ...].
bool isSequenceStartingWith0(Attribute attr);

// Returns the argument index for the giving FuncOp and its operand value.
int64_t getArgumentIndex(func::FuncOp op, Value value);

/// Computes the memory usage of the given allocations.
std::pair<size_t, size_t> computeMemory(const std::vector<Value>& allocs);

}  // namespace hlo
}  // namespace mlir

#endif  // MLIR_HLO_UTILS_HLO_UTILS_H
