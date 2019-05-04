//===- ViewOp.h - Linalg dialect ViewOp operation definition ------------===//
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

#ifndef LINALG1_VIEWOP_H_
#define LINALG1_VIEWOP_H_

#include "mlir/IR/OpDefinition.h"
#include "mlir/Support/LLVM.h"

namespace linalg {

class ViewType;

/// A `ViewOp` produces a `ViewType` which is a multi-dimensional range
/// abstraction on top of an underlying data type. For now we use the existing
/// mlir::MemRef for the underlying data type.
class ViewOp : public mlir::Op<ViewOp, mlir::OpTrait::VariadicOperands,
                               mlir::OpTrait::OneResult,
                               mlir::OpTrait::HasNoSideEffect> {
public:
  friend mlir::Operation;
  using Op::Op;

  //////////////////////////////////////////////////////////////////////////////
  // Hooks to customize the behavior of this op.
  //////////////////////////////////////////////////////////////////////////////
  static llvm::StringRef getOperationName() { return "linalg.view"; }
  static void build(mlir::Builder *b, mlir::OperationState *result,
                    mlir::Value *memRef,
                    llvm::ArrayRef<mlir::Value *> indexings);
  mlir::LogicalResult verify();
  static bool parse(mlir::OpAsmParser *parser, mlir::OperationState *result);
  void print(mlir::OpAsmPrinter *p);

  //////////////////////////////////////////////////////////////////////////////
  // Op-specific functionality.
  //////////////////////////////////////////////////////////////////////////////
  enum { FirstIndexingOperand = 1 };
  unsigned getRank();
  mlir::Type getElementType();
  ViewType getViewType();
  // May be something else than a MemRef in the future.
  mlir::Value *getSupportingMemRef();
  // Get the underlying indexing at a given rank.
  mlir::Value *getIndexing(unsigned rank);
  // Get all the indexings of type RangeOp.
  llvm::SmallVector<mlir::Value *, 8> getRanges();
  // Get all the indexings in this view.
  mlir::Operation::operand_range getIndexings();
};

} // namespace linalg

#endif // LINALG1_VIEWOP_H_
