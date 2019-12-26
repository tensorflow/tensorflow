//===- CallInterfaces.h - Call Interfaces for MLIR --------------*- C++ -*-===//
//
// Part of the MLIR Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
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
