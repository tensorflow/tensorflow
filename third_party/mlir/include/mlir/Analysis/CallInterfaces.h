//===- CallInterfaces.h - Call Interfaces for MLIR --------------*- C++ -*-===//
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
// This file contains the definitions of the call interfaces defined in
// `CallInterfaces.td`.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_ANALYSIS_CALLINTERFACES_H
#define MLIR_ANALYSIS_CALLINTERFACES_H

#include "mlir/IR/OpDefinition.h"
#include "llvm/ADT/PointerUnion.h"

namespace mlir {

/// A callable is either a symbol, or an SSA value, that is referenced by a
/// call-like operation. This represents the destination of the call.
struct CallInterfaceCallable : public PointerUnion<SymbolRefAttr, Value> {
  using PointerUnion<SymbolRefAttr, Value>::PointerUnion;
};

#include "mlir/Analysis/CallInterfaces.h.inc"
} // end namespace mlir

#endif // MLIR_ANALYSIS_CALLINTERFACES_H
