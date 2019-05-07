//===- SliceOp.h - Linalg dialect SliceOp operation definition ------------===//
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

#ifndef LINALG1_SLICEOP_H_
#define LINALG1_SLICEOP_H_

#include "linalg1/Types.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/Support/LLVM.h"

namespace linalg {

/// A SliceOp is used to create a "sub-View" from a ViewType. It results in a
/// new ViewType which is contained within its parent ViewType.
class SliceOp : public mlir::Op<SliceOp, mlir::OpTrait::NOperands<2>::Impl,
                                mlir::OpTrait::OneResult,
                                mlir::OpTrait::HasNoSideEffect> {
public:
  friend mlir::Operation;
  using Op::Op;

  //////////////////////////////////////////////////////////////////////////////
  // Hooks to customize the behavior of this op.
  //////////////////////////////////////////////////////////////////////////////
  static llvm::StringRef getOperationName() { return "linalg.slice"; }
  static void build(mlir::Builder *b, mlir::OperationState *result,
                    mlir::Value *view, mlir::Value *indexing, unsigned dim);
  mlir::LogicalResult verify();
  static mlir::ParseResult parse(mlir::OpAsmParser *parser,
                                 mlir::OperationState *result);
  void print(mlir::OpAsmPrinter *p);

  //////////////////////////////////////////////////////////////////////////////
  // Op-specific functionality.
  //////////////////////////////////////////////////////////////////////////////
  enum { FirstIndexingOperand = 1 };
  /// Returns the attribute name that describes which dimension of the input
  /// view that this SliceOp slices.
  static llvm::StringRef getSlicingDimAttrName() { return "dim"; }
  /// Returns the unique result of the parent SliceOp of ViewOp instruction that
  /// created the view on which this SliceOp operates.
  mlir::Value *getParentView() { return getOperand(0); }
  /// Returns the indexing operand of the current SliceOp.
  /// This operands may either be:
  ///   1. A range, in which case the operand comes from a RangeOp. This SliceOp
  ///      does not reduce the dimension of the input ViewType.
  ///   2. An index, in which case the operand comes from any possible producer
  ///      of an index. This SliceOp reduces the dimension of the input ViewType
  ///      by 1.
  mlir::Value *getIndexing() { return getOperand(1); }
  /// Returns the dim of the parent ViewType that is sliced by this SliceOp.
  unsigned getSlicingDim() {
    return getAttrOfType<mlir::IntegerAttr>(getSlicingDimAttrName()).getInt();
  }
  /// Returns the ViewType resulting from this SliceOp.
  ViewType getViewType();
  /// Returns the rank of the current ViewType.
  unsigned getRank();
  /// Return the element type of the current ViewType.
  mlir::Type getElementType();

  /// Returns the ViewType of `getParentView()`.
  ViewType getParentViewType();
  /// Returns the rank of the ViewType of `getParentView()`.
  unsigned getParentRank();
  /// Returns the element Type of the ViewType of `getParentView()`.
  mlir::Type getParentElementType();

  /// Returns true if the rank of the part view is greater than the rank of
  /// the child view.
  bool isRankDecreasing();

  // Get all the indexings in this slice.
  mlir::Operation::operand_range getIndexings();
};

} // namespace linalg

#endif // LINALG1_SLICEOP_H_
