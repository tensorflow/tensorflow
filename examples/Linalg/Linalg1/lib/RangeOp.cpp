//===- RangeOp.cpp - Implementation of the linalg RangeOp operation -------===//
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
// This file implements a simple IR operation to create a new RangeType in the
// linalg dialect.
//
//===----------------------------------------------------------------------===//

#include "linalg1/Ops.h"
#include "linalg1/Types.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/StandardTypes.h"

using mlir::Builder;
using mlir::IndexType;
using mlir::OpAsmParser;
using mlir::OpAsmPrinter;
using mlir::OperationState;
using mlir::Value;

// Minimal example for a new RangeOp operating on RangeType.
void linalg::RangeOp::build(Builder *b, OperationState *result, Value *min,
                            Value *max, Value *step) {
  result->addOperands({min, max, step});
  result->addTypes({linalg::RangeType::get(b->getContext())});
}

// Verification is simply that a RangeOp takes 3 index ssa-value.
mlir::LogicalResult linalg::RangeOp::verify() {
  if (!getMin() || !getMin()->getType().isa<IndexType>())
    return emitOpError("first operand should be of type index");
  if (!getMax() || !getMax()->getType().isa<IndexType>())
    return emitOpError("second operand should be of type index");
  if (!getStep() || !getStep()->getType().isa<IndexType>())
    return emitOpError("third operand should be of type index");
  return mlir::success();
}

// Parsing of the linalg dialect is not supported in this tutorial.
bool linalg::RangeOp::parse(OpAsmParser *parser, OperationState *result) {
  llvm_unreachable("Parsing linalg dialect is not supported in this tutorial");
}

// A RangeOp prints as:
//
// ```{.mlir}
//   linalg.range %arg0:%arg1:%c42 : !linalg.range
// ```
void linalg::RangeOp::print(OpAsmPrinter *p) {
  *p << getOperationName() << " " << *getMin() << ":" << *getMax() << ":"
     << *getStep() << " : " << getType();
}
