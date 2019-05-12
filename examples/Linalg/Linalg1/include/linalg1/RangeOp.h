//===- RangeOp.h - Linalg dialect RangeOp operation definition ------------===//
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

#ifndef LINALG1_RANGEOP_H_
#define LINALG1_RANGEOP_H_

#include "mlir/IR/OpDefinition.h"
#include "mlir/Support/LLVM.h"

namespace linalg {

/// A RangeOp is used to create a value of RangeType from 3 values of type index
/// that represent the min, max and step values of the range.
/// Note: step must be an mlir::ConstantIndexOp for now due to current
/// `affine.for` limitations.
class RangeOp : public mlir::Op<RangeOp, mlir::OpTrait::NOperands<3>::Impl,
                                mlir::OpTrait::OneResult,
                                mlir::OpTrait::HasNoSideEffect> {
public:
  using Op::Op;

  //////////////////////////////////////////////////////////////////////////////
  // Hooks to customize the behavior of this op.
  //////////////////////////////////////////////////////////////////////////////
  static llvm::StringRef getOperationName() { return "linalg.range"; }
  static void build(mlir::Builder *b, mlir::OperationState *result,
                    mlir::Value *min, mlir::Value *max, mlir::Value *step);
  mlir::LogicalResult verify();
  static mlir::ParseResult parse(mlir::OpAsmParser *parser,
                                 mlir::OperationState *result);
  void print(mlir::OpAsmPrinter *p);

  //////////////////////////////////////////////////////////////////////////////
  // Op-specific functionality.
  //////////////////////////////////////////////////////////////////////////////
  mlir::Value *getMin() { return getOperand(0); }
  mlir::Value *getMax() { return getOperand(1); }
  mlir::Value *getStep() { return getOperand(2); }
};

} // namespace linalg

#endif // LINALG1_RANGEOP_H_
