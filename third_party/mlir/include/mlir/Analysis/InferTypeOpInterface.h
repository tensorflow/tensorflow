//===- InferTypeOpInterface.h - Infer Type Interfaces -----------*- C++ -*-===//
//
// Part of the MLIR Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file contains the definitions of the infer op interfaces defined in
// `InferTypeOpInterface.td`.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_ANALYSIS_INFERTYPEOPINTERFACE_H_
#define MLIR_ANALYSIS_INFERTYPEOPINTERFACE_H_

#include "mlir/IR/Attributes.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/Types.h"
#include "mlir/Support/LLVM.h"
#include "llvm/ADT/SmallVector.h"

namespace mlir {

#include "mlir/Analysis/InferTypeOpInterface.h.inc"

namespace OpTrait {
template <typename ConcreteType>
class TypeOpInterfaceDefault
    : public TraitBase<ConcreteType, TypeOpInterfaceDefault> {
public:
  /// Returns whether two arrays are equal as strongest check for compatibility
  /// by default.
  static bool isCompatibleReturnTypes(ArrayRef<Type> lhs, ArrayRef<Type> rhs) {
    return lhs == rhs;
  };
};
} // namespace OpTrait

} // namespace mlir

#endif // MLIR_ANALYSIS_INFERTYPEOPINTERFACE_H_
