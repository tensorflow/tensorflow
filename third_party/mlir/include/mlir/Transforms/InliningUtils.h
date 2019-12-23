//===- InliningUtils.h - Inliner utilities ----------------------*- C++ -*-===//
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
// This header file defines interfaces for various inlining utility methods.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_TRANSFORMS_INLINING_UTILS_H
#define MLIR_TRANSFORMS_INLINING_UTILS_H

#include "mlir/IR/DialectInterface.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/Region.h"

namespace mlir {

class Block;
class BlockAndValueMapping;
class CallableOpInterface;
class CallOpInterface;
class FuncOp;
class OpBuilder;
class Operation;
class Region;
class Value;

//===----------------------------------------------------------------------===//
// InlinerInterface
//===----------------------------------------------------------------------===//

/// This is the interface that must be implemented by the dialects of operations
/// to be inlined. This interface should only handle the operations of the
/// given dialect.
class DialectInlinerInterface
    : public DialectInterface::Base<DialectInlinerInterface> {
public:
  DialectInlinerInterface(Dialect *dialect) : Base(dialect) {}

  //===--------------------------------------------------------------------===//
  // Analysis Hooks
  //===--------------------------------------------------------------------===//

  /// Returns true if the given region 'src' can be inlined into the region
  /// 'dest' that is attached to an operation registered to the current dialect.
  /// 'valueMapping' contains any remapped values from within the 'src' region.
  /// This can be used to examine what values will replace entry arguments into
  /// the 'src' region for example.
  virtual bool isLegalToInline(Region *dest, Region *src,
                               BlockAndValueMapping &valueMapping) const {
    return false;
  }

  /// Returns true if the given operation 'op', that is registered to this
  /// dialect, can be inlined into the given region, false otherwise.
  /// 'valueMapping' contains any remapped values from within the 'src' region.
  /// This can be used to examine what values may potentially replace the
  /// operands to 'op'.
  virtual bool isLegalToInline(Operation *op, Region *dest,
                               BlockAndValueMapping &valueMapping) const {
    return false;
  }

  /// This hook is invoked on an operation that contains regions. It should
  /// return true if the analyzer should recurse within the regions of this
  /// operation when computing legality and cost, false otherwise. The default
  /// implementation returns true.
  virtual bool shouldAnalyzeRecursively(Operation *op) const { return true; }

  //===--------------------------------------------------------------------===//
  // Transformation Hooks
  //===--------------------------------------------------------------------===//

  /// Handle the given inlined terminator by replacing it with a new operation
  /// as necessary. This overload is called when the inlined region has more
  /// than one block. The 'newDest' block represents the new final branching
  /// destination of blocks within this region, i.e. operations that release
  /// control to the parent operation will likely now branch to this block.
  /// Its block arguments correspond to any values that need to be replaced by
  /// terminators within the inlined region.
  virtual void handleTerminator(Operation *op, Block *newDest) const {
    llvm_unreachable("must implement handleTerminator in the case of multiple "
                     "inlined blocks");
  }

  /// Handle the given inlined terminator by replacing it with a new operation
  /// as necessary. This overload is called when the inlined region only
  /// contains one block. 'valuesToReplace' contains the previously returned
  /// values of the call site before inlining. These values must be replaced by
  /// this callback if they had any users (for example for traditional function
  /// calls, these are directly replaced with the operands of the `return`
  /// operation). The given 'op' will be removed by the caller, after this
  /// function has been called.
  virtual void handleTerminator(Operation *op,
                                ArrayRef<Value> valuesToReplace) const {
    llvm_unreachable(
        "must implement handleTerminator in the case of one inlined block");
  }

  /// Attempt to materialize a conversion for a type mismatch between a call
  /// from this dialect, and a callable region. This method should generate an
  /// operation that takes 'input' as the only operand, and produces a single
  /// result of 'resultType'. If a conversion can not be generated, nullptr
  /// should be returned. For example, this hook may be invoked in the following
  /// scenarios:
  ///   func @foo(i32) -> i32 { ... }
  ///
  ///   // Mismatched input operand
  ///   ... = foo.call @foo(%input : i16) -> i32
  ///
  ///   // Mismatched result type.
  ///   ... = foo.call @foo(%input : i32) -> i16
  ///
  /// NOTE: This hook may be invoked before the 'isLegal' checks above.
  virtual Operation *materializeCallConversion(OpBuilder &builder, Value input,
                                               Type resultType,
                                               Location conversionLoc) const {
    return nullptr;
  }
};

/// This interface provides the hooks into the inlining interface.
/// Note: this class automatically collects 'DialectInlinerInterface' objects
/// registered to each dialect within the given context.
class InlinerInterface
    : public DialectInterfaceCollection<DialectInlinerInterface> {
public:
  using Base::Base;

  /// Process a set of blocks that have been inlined. This callback is invoked
  /// *before* inlined terminator operations have been processed.
  virtual void
  processInlinedBlocks(iterator_range<Region::iterator> inlinedBlocks) {}

  /// These hooks mirror the hooks for the DialectInlinerInterface, with default
  /// implementations that call the hook on the handler for the dialect 'op' is
  /// registered to.

  //===--------------------------------------------------------------------===//
  // Analysis Hooks
  //===--------------------------------------------------------------------===//

  virtual bool isLegalToInline(Region *dest, Region *src,
                               BlockAndValueMapping &valueMapping) const;
  virtual bool isLegalToInline(Operation *op, Region *dest,
                               BlockAndValueMapping &valueMapping) const;
  virtual bool shouldAnalyzeRecursively(Operation *op) const;

  //===--------------------------------------------------------------------===//
  // Transformation Hooks
  //===--------------------------------------------------------------------===//

  virtual void handleTerminator(Operation *op, Block *newDest) const;
  virtual void handleTerminator(Operation *op,
                                ArrayRef<Value> valuesToRepl) const;
};

//===----------------------------------------------------------------------===//
// Inline Methods.
//===----------------------------------------------------------------------===//

/// This function inlines a region, 'src', into another. This function returns
/// failure if it is not possible to inline this function. If the function
/// returned failure, then no changes to the module have been made.
///
/// The provided 'inlinePoint' must be within a region, and corresponds to the
/// location where the 'src' region should be inlined. 'mapping' contains any
/// remapped operands that are used within the region, and *must* include
/// remappings for the entry arguments to the region. 'resultsToReplace'
/// corresponds to any results that should be replaced by terminators within the
/// inlined region. 'inlineLoc' is an optional Location that, if provided, will
/// be used to update the inlined operations' location information.
/// 'shouldCloneInlinedRegion' corresponds to whether the source region should
/// be cloned into the 'inlinePoint' or spliced directly.
LogicalResult inlineRegion(InlinerInterface &interface, Region *src,
                           Operation *inlinePoint, BlockAndValueMapping &mapper,
                           ArrayRef<Value> resultsToReplace,
                           Optional<Location> inlineLoc = llvm::None,
                           bool shouldCloneInlinedRegion = true);

/// This function is an overload of the above 'inlineRegion' that allows for
/// providing the set of operands ('inlinedOperands') that should be used
/// in-favor of the region arguments when inlining.
LogicalResult inlineRegion(InlinerInterface &interface, Region *src,
                           Operation *inlinePoint,
                           ArrayRef<Value> inlinedOperands,
                           ArrayRef<Value> resultsToReplace,
                           Optional<Location> inlineLoc = llvm::None,
                           bool shouldCloneInlinedRegion = true);

/// This function inlines a given region, 'src', of a callable operation,
/// 'callable', into the location defined by the given call operation. This
/// function returns failure if inlining is not possible, success otherwise. On
/// failure, no changes are made to the module. 'shouldCloneInlinedRegion'
/// corresponds to whether the source region should be cloned into the 'call' or
/// spliced directly.
LogicalResult inlineCall(InlinerInterface &interface, CallOpInterface call,
                         CallableOpInterface callable, Region *src,
                         bool shouldCloneInlinedRegion = true);

} // end namespace mlir

#endif // MLIR_TRANSFORMS_INLINING_UTILS_H
