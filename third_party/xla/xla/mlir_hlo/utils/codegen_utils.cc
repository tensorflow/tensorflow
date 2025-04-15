/* Copyright 2021 The OpenXLA Authors.

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

#include "utils/codegen_utils.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Location.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/LLVM.h"

using llvm::SmallVector;

namespace mlir {
namespace codegen_utils {

Value emitNumElementsComputation(OpBuilder& b, Location loc, Value memref) {
  int rank = mlir::cast<MemRefType>(memref.getType()).getRank();
  Value numElements;
  numElements = b.create<mlir::arith::ConstantOp>(
      loc, b.getIndexType(), b.getIntegerAttr(b.getIndexType(), 1));
  for (int r = 0; r < rank; ++r) {
    auto dimSize = b.create<memref::DimOp>(loc, memref, r);
    numElements = b.create<arith::MulIOp>(loc, numElements, dimSize);
  }
  return numElements;
}

Value emitNumElementsComputation(OpBuilder& b, Location loc, Operation* op) {
  // only const rank is supported for now
  assert(op->getDialect()->getNamespace() == "lmhlo");
  int numOperands = op->getNumOperands();
  Value resultMemref = op->getOperand(numOperands - 1);
  return emitNumElementsComputation(b, loc, resultMemref);
}

SmallVector<Value> calcMultiDimIndex(OpBuilder& b, Location loc,
                                     Value linearIndex, ArrayRef<Value> shape) {
  int rank = shape.size();
  SmallVector<Value> result;
  if (rank == 0) return result;
  if (rank == 1) {
    result.push_back(linearIndex);
    return result;
  }

  // dim_acc_mul_vec = [d, c*d, b*c*d]
  SmallVector<Value> dimAccMulVec;
  Value tmpAccMul = shape[rank - 1];
  dimAccMulVec.push_back(tmpAccMul);
  for (int i = rank - 2; i > 0; --i) {
    tmpAccMul = b.create<arith::MulIOp>(loc, tmpAccMul, shape[i]);
    dimAccMulVec.push_back(tmpAccMul);
  }
  Value blockIndex = linearIndex;
  for (int i = 0; i < rank; ++i) {
    Value index;
    if (i == rank - 1) {
      index = blockIndex;
    } else {
      index = b.create<arith::DivUIOp>(loc, blockIndex, dimAccMulVec.back());
      blockIndex =
          b.create<arith::RemUIOp>(loc, blockIndex, dimAccMulVec.back());
      dimAccMulVec.pop_back();
    }
    result.push_back(index);
  }
  return result;
}

SmallVector<Value> calcMultiDimIndex(OpBuilder& b, Location loc,
                                     Value linearIndex, Value memref) {
  int rank = mlir::cast<MemRefType>(memref.getType()).getRank();
  SmallVector<Value> result;
  if (rank == 0) return result;
  if (rank == 1) {
    result.push_back(linearIndex);
    return result;
  }
  // shape = [a, b, c, d]
  SmallVector<Value, 4> shapeVec;
  for (int i = 0; i < rank; ++i) {
    shapeVec.push_back(b.create<memref::DimOp>(loc, memref, i));
  }

  return calcMultiDimIndex(b, loc, linearIndex, shapeVec);
}

static SmallVector<Value> calcMultiDimIndexForFirstOperand(OpBuilder& b,
                                                           Location loc,
                                                           Value linearIndex,
                                                           Operation* op) {
  assert(op->getDialect()->getNamespace() == "lmhlo");
  Value operandMemref = op->getOperand(0);
  return calcMultiDimIndex(b, loc, linearIndex, operandMemref);
}

}  // namespace codegen_utils
}  // namespace mlir
