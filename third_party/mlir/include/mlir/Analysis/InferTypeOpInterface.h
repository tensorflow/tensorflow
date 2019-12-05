//===- InferTypeOpInterface.h - Infer Type Interfaces -----------*- C++ -*-===//
//
// Copyright 2019 The MLIR Authors.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//   http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
// =============================================================================
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
