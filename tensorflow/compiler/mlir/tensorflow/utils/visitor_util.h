/* Copyright 2020 The TensorFlow Authors. All Rights Reserved.

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
#ifndef TENSORFLOW_COMPILER_MLIR_TENSORFLOW_UTILS_VISITOR_UTIL_H_
#define TENSORFLOW_COMPILER_MLIR_TENSORFLOW_UTILS_VISITOR_UTIL_H_

#include <utility>

#include "mlir/IR/Visitors.h"  // from @llvm-project

// This file defines generic (pre/in/post)-order MLIR IR visitors/walkers. The
// walk() utility that MLIR core provides traverses operations in a block/
// blocks in a region in the program order, and these walkers do the same. When
// operations have regions attached to them, the core MLIR walkers visit the
// regions attached to an Op first, and then visit the op. So within the context
// of a single Op, the traversal is post-order (considering the Op as the parent
// node and regions as the children). For certain use cases, it may be more
// efficient/desirable to visit the parent Op before visiting the attached
// regions. As an example, if the attached regions have region arguments that
// are related to the operation inputs (tf.WhileRegion is an example), then we
// may want to propagate some information from the Op inputs to the region
// inputs and then visit the regions to continue progagating that information
// within the regions. With just post-order traversal, to acheive the same we
// may need to schedule another walk so make sure child regions get visited.
// A pre-order walk (within the context of a single operation) will avoid that.
// Similarly, for certain operations, we may want to visit the Op both before
// and after all regions have been visited (say to propagate information from
// inputs -> region arguments and then from region results -> outputs).

// In general, since the data flow between an operation and its regions is
// opaque in MLIR, we may need to visit the operation in-between regions as well
// if say region0 is transferring control back to the Op and from then to
// region1. So a more general walker that supports pre/in/post-order walk is
// desirable. To support this, the generic walkers defined below will invoke
// the walk callback on the parent Op at each stage of the child region walk,
// i.e., before visiting any region, in between regions, and after visiting all
// regions. To indicate the current walk stage, the callback will also get a
// `WalkState` parameter. The callback can inspect the current walk stage and
// decide to take appropriate actions (incuding not doing anything). With this
// the walker below can support pre/in/post-order walks as well as combined
// walks (pre+in+post)-order walk.

namespace tensorflow {

// A class to indicate the current walk stage.
class WalkStage {
 public:
  explicit WalkStage(mlir::Operation *op);

  bool IsBeforeAllRegions() const { return next_region_ == 0; }
  bool IsBeforeRegion(int region) const { return next_region_ == region; }
  bool IsAfterRegion(int region) const { return next_region_ == region + 1; }
  bool IsAfterAllRegions() const { return next_region_ == num_regions_; }
  void Advance() { next_region_++; }
  int GetNextRegion() const { return next_region_; }

 private:
  const int num_regions_;
  int next_region_;
};

namespace detail {
// This is similar to MLIR version, but works with multiple argument functions.
// Helper templates to deduce the first argument of a callback parameter.
template <typename Ret, typename Arg, typename... Rest>
Arg first_argument_type(Ret (*)(Arg, Rest...));
template <typename Ret, typename F, typename Arg, typename... Rest>
Arg first_argument_type(Ret (F::*)(Arg, Rest...));
template <typename Ret, typename F, typename Arg, typename... Rest>
Arg first_argument_type(Ret (F::*)(Arg, Rest...) const);
template <typename F>
decltype(first_argument_type(&F::operator())) first_argument_type(F);

/// Type definition of the first argument to the given callable 'T'.
template <typename T>
using first_argument = decltype(first_argument_type(std::declval<T>()));

using VoidCallback =
    llvm::function_ref<void(mlir::Operation *, const WalkStage &)>;
using InterruptCallback =
    llvm::function_ref<mlir::WalkResult(mlir::Operation *, const WalkStage &)>;

// Walk all of the operations nested under and including the given operation.
void WalkOperations(mlir::Operation *op, VoidCallback callback);

// Walk all of the operations nested under and including the given operation.
// This methods walks operations until an interrupt result is returned by the
// callback.
mlir::WalkResult WalkOperations(mlir::Operation *op,
                                InterruptCallback callback);

}  // namespace detail

// Walk all of the operations nested under and including the given operation.
// This method is selected for stage-aware callbacks that operate on Operation*.
//
// Example:
//   tensorflow::walk(op, [](Operation *op, const WalkStage &stage) { ... });
template <typename FuncTy, typename ArgT = detail::first_argument<FuncTy>,
          typename RetT = decltype(std::declval<FuncTy>()(
              std::declval<ArgT>(), std::declval<const WalkStage &>()))>
typename std::enable_if<std::is_same<ArgT, mlir::Operation *>::value,
                        RetT>::type
GenericWalk(mlir::Operation *op, FuncTy &&callback) {
  return detail::WalkOperations(
      op, llvm::function_ref<RetT(ArgT, const WalkStage &)>(callback));
}

// Walk all of the operations of type 'ArgT' nested under and including the
// given operation. This method is selected for void returning callbacks that
// operate on a specific derived operation type.
//
// Example:
//   tensorflow::walk(op, [](ReturnOp op, const WalkStage &stage) { ... });
template <typename FuncTy, typename ArgT = detail::first_argument<FuncTy>,
          typename RetT = decltype(std::declval<FuncTy>()(
              std::declval<ArgT>(), std::declval<const WalkStage &>()))>
typename std::enable_if<!std::is_same<ArgT, mlir::Operation *>::value &&
                            std::is_same<RetT, void>::value,
                        RetT>::type
GenericWalk(mlir::Operation *op, FuncTy &&callback) {
  auto wrapperFn = [&](mlir::Operation *op, const WalkStage &stage) {
    if (auto derivedOp = llvm::dyn_cast<ArgT>(op)) callback(derivedOp, stage);
  };
  return detail::WalkOperations(op,
                                static_cast<detail::VoidCallback>(wrapperFn));
}

// Walk all of the operations of type 'ArgT' nested under and including the
// given operation. This method is selected for WalkReturn returning
// interruptible callbacks that operate on a specific derived operation type.
//
// Example:
//   tensorflow::walk(op, [](ReturnOp op, const WalkStage &stage) {
//     if (some_invariant)
//       return WalkResult::interrupt();
//     return WalkResult::advance();
//   });
template <typename FuncTy, typename ArgT = detail::first_argument<FuncTy>,
          typename RetT = decltype(std::declval<FuncTy>()(
              std::declval<ArgT>(), std::declval<const WalkStage &>()))>
typename std::enable_if<!std::is_same<ArgT, mlir::Operation *>::value &&
                            std::is_same<RetT, mlir::WalkResult>::value,
                        RetT>::type
GenericWalk(mlir::Operation *op, FuncTy &&callback) {
  auto wrapperFn = [&](mlir::Operation *op, const WalkStage &stage) {
    if (auto derivedOp = llvm::dyn_cast<ArgT>(op))
      return callback(derivedOp, stage);
    return mlir::WalkResult::advance();
  };
  return detail::WalkOperations(
      op, static_cast<detail::InterruptCallback>(wrapperFn));
}

}  // namespace tensorflow

#endif  // TENSORFLOW_COMPILER_MLIR_TENSORFLOW_UTILS_VISITOR_UTIL_H_
