//===- LinalgOps.h - Linalg Operations --------------------------*- C++ -*-===//
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

#ifndef MLIR_LINALG_LINALGOPS_H_
#define MLIR_LINALG_LINALGOPS_H_

#include "mlir/IR/OpDefinition.h"
#include "mlir/Support/LLVM.h"

namespace mlir {

/// A RangeOp is used to create a value of RangeType from 3 values of type index
/// that represent the min, max and step values of the range.
class RangeOp : public Op<RangeOp, OpTrait::NOperands<3>::Impl,
                          OpTrait::OneResult, OpTrait::HasNoSideEffect> {
public:
  using Op::Op;

  //////////////////////////////////////////////////////////////////////////////
  // Hooks to customize the behavior of this op.
  //////////////////////////////////////////////////////////////////////////////
  static llvm::StringRef getOperationName() { return "linalg.range"; }
  static void build(Builder *b, OperationState *result, Value *min, Value *max,
                    Value *step);
  LogicalResult verify();
  static bool parse(OpAsmParser *parser, OperationState *result);
  void print(OpAsmPrinter *p);

  //////////////////////////////////////////////////////////////////////////////
  // Op-specific functionality.
  //////////////////////////////////////////////////////////////////////////////
  Value *min() { return getOperand(0); }
  Value *max() { return getOperand(1); }
  Value *step() { return getOperand(2); }
};

} // namespace mlir

#endif // MLIR_LINALG_LINALGOPS_H_
