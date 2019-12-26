//===- SideEffectsInterface.h - dialect interface modeling side effects ---===//
//
// Part of the MLIR Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file specifies a dialect interface to model side-effects.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_TRANSFORMS_SIDEEFFECTSINTERFACE_H_
#define MLIR_TRANSFORMS_SIDEEFFECTSINTERFACE_H_

#include "mlir/IR/DialectInterface.h"
#include "mlir/IR/Operation.h"

namespace mlir {

/// Specifies an interface for basic side-effect modelling that is used by the
/// loop-invariant code motion pass.
///
/// TODO: This interface should be replaced by a more general solution.
class SideEffectsDialectInterface
    : public DialectInterface::Base<SideEffectsDialectInterface> {
public:
  SideEffectsDialectInterface(Dialect *dialect) : Base(dialect) {}

  enum SideEffecting {
    Never,     /* the operation has no side-effects */
    Recursive, /* the operation has side-effects if a contained operation has */
    Always     /* the operation has side-effects */
  };

  /// Checks whether the given operation has side-effects.
  virtual SideEffecting isSideEffecting(Operation *op) const {
    if (op->hasNoSideEffect())
      return Never;
    return Always;
  };
};

class SideEffectsInterface
    : public DialectInterfaceCollection<SideEffectsDialectInterface> {
public:
  using SideEffecting = SideEffectsDialectInterface::SideEffecting;
  explicit SideEffectsInterface(MLIRContext *ctx)
      : DialectInterfaceCollection<SideEffectsDialectInterface>(ctx) {}

  SideEffecting isSideEffecting(Operation *op) const {
    // First check generic trait.
    if (op->hasNoSideEffect())
      return SideEffecting::Never;
    if (auto handler = getInterfaceFor(op))
      return handler->isSideEffecting(op);

    return SideEffecting::Always;
  }
};

} // namespace mlir

#endif // MLIR_TRANSFORMS_SIDEEFFECTSINTERFACE_H_
