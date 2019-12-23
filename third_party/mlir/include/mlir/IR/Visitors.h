//===- Visitors.h - Utilities for visiting operations -----------*- C++ -*-===//
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
// This file defines utilities for walking and visiting operations.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_IR_VISITORS_H
#define MLIR_IR_VISITORS_H

#include "mlir/Support/LLVM.h"
#include "mlir/Support/LogicalResult.h"
#include "llvm/ADT/STLExtras.h"

namespace mlir {
class Diagnostic;
class InFlightDiagnostic;
class Operation;

/// A utility result that is used to signal if a walk method should be
/// interrupted or advance.
class WalkResult {
  enum ResultEnum { Interrupt, Advance } result;

public:
  WalkResult(ResultEnum result) : result(result) {}

  /// Allow LogicalResult to interrupt the walk on failure.
  WalkResult(LogicalResult result)
      : result(failed(result) ? Interrupt : Advance) {}

  /// Allow diagnostics to interrupt the walk.
  WalkResult(Diagnostic &&) : result(Interrupt) {}
  WalkResult(InFlightDiagnostic &&) : result(Interrupt) {}

  bool operator==(const WalkResult &rhs) const { return result == rhs.result; }

  static WalkResult interrupt() { return {Interrupt}; }
  static WalkResult advance() { return {Advance}; }

  /// Returns if the walk was interrupted.
  bool wasInterrupted() const { return result == Interrupt; }
};

namespace detail {
/// Helper templates to deduce the first argument of a callback parameter.
template <typename Ret, typename Arg> Arg first_argument_type(Ret (*)(Arg));
template <typename Ret, typename F, typename Arg>
Arg first_argument_type(Ret (F::*)(Arg));
template <typename Ret, typename F, typename Arg>
Arg first_argument_type(Ret (F::*)(Arg) const);
template <typename F>
decltype(first_argument_type(&F::operator())) first_argument_type(F);

/// Type definition of the first argument to the given callable 'T'.
template <typename T>
using first_argument = decltype(first_argument_type(std::declval<T>()));

/// Walk all of the operations nested under and including the given operation.
void walkOperations(Operation *op, function_ref<void(Operation *op)> callback);

/// Walk all of the operations nested under and including the given operation.
/// This methods walks operations until an interrupt result is returned by the
/// callback.
WalkResult walkOperations(Operation *op,
                          function_ref<WalkResult(Operation *op)> callback);

// Below are a set of functions to walk nested operations. Users should favor
// the direct `walk` methods on the IR classes(Operation/Block/etc) over these
// methods. They are also templated to allow for statically dispatching based
// upon the type of the callback function.

/// Walk all of the operations nested under and including the given operation.
/// This method is selected for callbacks that operation on Operation*.
///
/// Example:
///   op->walk([](Operation *op) { ... });
template <
    typename FuncTy, typename ArgT = detail::first_argument<FuncTy>,
    typename RetT = decltype(std::declval<FuncTy>()(std::declval<ArgT>()))>
typename std::enable_if<std::is_same<ArgT, Operation *>::value, RetT>::type
walkOperations(Operation *op, FuncTy &&callback) {
  return detail::walkOperations(op, function_ref<RetT(ArgT)>(callback));
}

/// Walk all of the operations of type 'ArgT' nested under and including the
/// given operation. This method is selected for void returning callbacks that
/// operate on a specific derived operation type.
///
/// Example:
///   op->walk([](ReturnOp op) { ... });
template <
    typename FuncTy, typename ArgT = detail::first_argument<FuncTy>,
    typename RetT = decltype(std::declval<FuncTy>()(std::declval<ArgT>()))>
typename std::enable_if<!std::is_same<ArgT, Operation *>::value &&
                            std::is_same<RetT, void>::value,
                        RetT>::type
walkOperations(Operation *op, FuncTy &&callback) {
  auto wrapperFn = [&](Operation *op) {
    if (auto derivedOp = dyn_cast<ArgT>(op))
      callback(derivedOp);
  };
  return detail::walkOperations(op, function_ref<RetT(Operation *)>(wrapperFn));
}

/// Walk all of the operations of type 'ArgT' nested under and including the
/// given operation. This method is selected for WalkReturn returning
/// interruptible callbacks that operate on a specific derived operation type.
///
/// Example:
///   op->walk([](ReturnOp op) {
///     if (some_invariant)
///       return WalkResult::interrupt();
///     return WalkResult::advance();
///   });
template <
    typename FuncTy, typename ArgT = detail::first_argument<FuncTy>,
    typename RetT = decltype(std::declval<FuncTy>()(std::declval<ArgT>()))>
typename std::enable_if<!std::is_same<ArgT, Operation *>::value &&
                            std::is_same<RetT, WalkResult>::value,
                        RetT>::type
walkOperations(Operation *op, FuncTy &&callback) {
  auto wrapperFn = [&](Operation *op) {
    if (auto derivedOp = dyn_cast<ArgT>(op))
      return callback(derivedOp);
    return WalkResult::advance();
  };
  return detail::walkOperations(op, function_ref<RetT(Operation *)>(wrapperFn));
}

/// Utility to provide the return type of a templated walk method.
template <typename FnT>
using walkResultType = decltype(walkOperations(nullptr, std::declval<FnT>()));
} // end namespace detail

} // namespace mlir

#endif
