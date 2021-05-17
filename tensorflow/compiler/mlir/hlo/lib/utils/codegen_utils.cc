/* Copyright 2019 The TensorFlow Authors. All Rights Reserved.

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

#include "mlir-hlo/utils/codegen_utils.h"

#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Location.h"
#include "mlir/Pass/Pass.h"

using llvm::SmallVector;

namespace mlir {
namespace codegen_utils {

Value calcNumElements(OpBuilder& b, Location loc, Value memref) {
  auto rank = memref.getType().cast<MemRefType>().getRank();
  Value num_elements;
  num_elements = b.create<mlir::ConstantOp>(
      loc, b.getIndexType(), b.getIntegerAttr(b.getIndexType(), 1));
  for (int r = 0; r < rank; ++r) {
    auto dim_size = b.create<memref::DimOp>(loc, memref, r);
    num_elements = b.create<MulIOp>(loc, num_elements, dim_size);
  }
  return num_elements;
}

Value calcNumElements(OpBuilder& b, Location loc, Operation* op) {
  // only const rank is supported for now
  assert(op->getDialect()->getNamespace() == "lmhlo");
  auto num_operands = op->getNumOperands();
  auto result_memref = op->getOperand(num_operands - 1);
  return calcNumElements(b, loc, result_memref);
}

Value calcNumElementsForFirstOperand(OpBuilder& b, Location loc,
                                     Operation* op) {
  assert(op->getDialect()->getNamespace() == "lmhlo");
  auto operand_memref = op->getOperand(0);
  return calcNumElements(b, loc, operand_memref);
}

SmallVector<Value, 4> calcMultiDimIndex(OpBuilder& b, Location loc,
                                        Value linear_index,
                                        ArrayRef<Value> shape) {
  auto rank = shape.size();
  SmallVector<Value, 4> result;
  if (rank == 0) {
    return result;
  } else if (rank == 1) {
    result.push_back(linear_index);
    return result;
  }

  // dim_acc_mul_vec = [d, c*d, b*c*d]
  std::vector<Value> dim_acc_mul_vec;
  Value tmp_acc_mul = shape[rank - 1];
  dim_acc_mul_vec.emplace_back(tmp_acc_mul);
  for (int i = rank - 2; i > 0; --i) {
    tmp_acc_mul = b.create<MulIOp>(loc, tmp_acc_mul, shape[i]);
    dim_acc_mul_vec.emplace_back(tmp_acc_mul);
  }
  Value block_index = linear_index;
  for (int i = 0; i < rank; ++i) {
    Value index;
    if (i == rank - 1) {
      index = block_index;
    } else {
      index =
          b.create<UnsignedDivIOp>(loc, block_index, dim_acc_mul_vec.back());
      block_index =
          b.create<UnsignedRemIOp>(loc, block_index, dim_acc_mul_vec.back());
      dim_acc_mul_vec.pop_back();
    }
    result.push_back(index);
  }
  return result;
}

SmallVector<Value, 4> calcMultiDimIndex(OpBuilder& b, Location loc,
                                        Value linear_index, Value memref) {
  auto rank = memref.getType().cast<MemRefType>().getRank();
  SmallVector<Value, 4> result;
  if (rank == 0) {
    return result;
  } else if (rank == 1) {
    result.push_back(linear_index);
    return result;
  }
  // shape = [a, b, c, d]
  SmallVector<Value, 4> shape_vec;
  for (int i = 0; i < rank; ++i) {
    shape_vec.push_back(b.create<memref::DimOp>(loc, memref, i));
  }

  return calcMultiDimIndex(b, loc, linear_index, shape_vec);
}

SmallVector<Value, 4> calcMultiDimIndex(OpBuilder& b, Location loc,
                                        Value linear_index, Operation* op) {
  assert(op->getDialect()->getNamespace() == "lmhlo");
  auto num_operands = op->getNumOperands();
  auto result_memref = op->getOperand(num_operands - 1);
  return calcMultiDimIndex(b, loc, linear_index, result_memref);
}

SmallVector<Value, 4> calcMultiDimIndexForFirstOperand(OpBuilder& b,
                                                       Location loc,
                                                       Value linear_index,
                                                       Operation* op) {
  assert(op->getDialect()->getNamespace() == "lmhlo");
  auto operand_memref = op->getOperand(0);
  return calcMultiDimIndex(b, loc, linear_index, operand_memref);
}

Value calcLinearIndex(OpBuilder& b, Location loc,
                      llvm::ArrayRef<Value> multi_index, Operation* op) {
  auto num_operands = op->getNumOperands();
  auto result_memref = op->getOperand(num_operands - 1);
  auto rank = result_memref.getType().cast<MemRefType>().getRank();

  if (rank == 0) {
    return b.create<ConstantIndexOp>(loc, 0);
  } else if (rank == 1) {
    return multi_index[0];
  }

  SmallVector<Value, 4> shape_vec;
  for (int i = 0; i < rank; ++i) {
    shape_vec.push_back(b.create<memref::DimOp>(loc, result_memref, i));
  }

  return calcLinearIndex(b, loc, multi_index, shape_vec);
}

Value calcLinearIndex(OpBuilder& b, Location loc,
                      llvm::ArrayRef<Value> multi_index,
                      llvm::ArrayRef<Value> shape) {
  assert(multi_index.size() == shape.size());
  if (shape.empty()) {
    return b.create<ConstantIndexOp>(loc, 0);
  }

  Value linear = multi_index[0];
  for (size_t i = 1; i < shape.size(); ++i) {
    // linear = (linear * shape[i]) + multi_index[i]
    linear = b.create<AddIOp>(loc, b.create<MulIOp>(loc, linear, shape[i]),
                              multi_index[i]);
  }
  return linear;
}

}  // namespace codegen_utils
}  // namespace mlir
