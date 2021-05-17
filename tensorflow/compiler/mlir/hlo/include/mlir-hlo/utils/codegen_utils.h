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

#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/IR/MLIRContext.h"  // TF:llvm-project

#ifndef TENSORFLOW_COMPILER_MLIR_HLO_INCLUDE_MLIR_HLO_UTILS_CODEGEN_UTILS_H_
#define TENSORFLOW_COMPILER_MLIR_HLO_INCLUDE_MLIR_HLO_UTILS_CODEGEN_UTILS_H_

namespace mlir {
class Value;
class ValueRange;
class OpBuilder;
class Location;
class Operation;
namespace codegen_utils {

Value calcNumElements(OpBuilder& b, Location loc, Operation* op);
Value calcNumElements(OpBuilder& b, Location loc, Value memref);

Value calcNumElementsForFirstOperand(OpBuilder& b, Location loc, Operation* op);

llvm::SmallVector<Value, 4> calcMultiDimIndex(OpBuilder& b, Location loc,
                                              Value linear_index,
                                              Operation* op);

llvm::SmallVector<Value, 4> calcMultiDimIndex(OpBuilder& b, Location loc,
                                              Value linear_index, Value memref);

llvm::SmallVector<Value, 4> calcMultiDimIndex(OpBuilder& b, Location loc,
                                              Value linear_index,
                                              llvm::ArrayRef<Value> shape);

llvm::SmallVector<Value, 4> calcMultiDimIndexForFirstOperand(OpBuilder& b,
                                                             Location loc,
                                                             Value linear_index,
                                                             Operation* op);

Value calcLinearIndex(OpBuilder& b, Location loc,
                      llvm::ArrayRef<Value> multi_index, Operation* op);

Value calcLinearIndex(OpBuilder& b, Location loc,
                      llvm::ArrayRef<Value> multi_index,
                      llvm::ArrayRef<Value> shape);

}  // namespace codegen_utils
}  // namespace mlir

#endif  // TENSORFLOW_COMPILER_MLIR_HLO_INCLUDE_MLIR_HLO_UTILS_CODEGEN_UTILS_H_
