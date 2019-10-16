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

#include "tensorflow/compiler/mlir/tensorflow/ir/tf_device.h"

#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/SMLoc.h"
#include "mlir/IR/MLIRContext.h"  // TF:local_config_mlir
#include "mlir/IR/OpImplementation.h"  // TF:local_config_mlir
#include "mlir/IR/OperationSupport.h"  // TF:local_config_mlir
#include "mlir/IR/Types.h"  // TF:local_config_mlir
#include "mlir/Support/STLExtras.h"  // TF:local_config_mlir

namespace mlir {
namespace tf_device {

TensorFlowDeviceDialect::TensorFlowDeviceDialect(MLIRContext* context)
    : Dialect(/*name=*/"tf_device", context) {
  addOperations<
#define GET_OP_LIST
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_device.cc.inc"
      >();
}

//===----------------------------------------------------------------------===//
// tf_device.return
//===----------------------------------------------------------------------===//

namespace {
ParseResult ParseReturnOp(OpAsmParser* parser, OperationState* state) {
  llvm::SmallVector<OpAsmParser::OperandType, 2> op_info;
  llvm::SmallVector<Type, 2> types;
  llvm::SMLoc loc = parser->getCurrentLocation();
  return failure(parser->parseOperandList(op_info) ||
                 (!op_info.empty() && parser->parseColonTypeList(types)) ||
                 parser->resolveOperands(op_info, types, loc, state->operands));
}

void Print(ReturnOp op, OpAsmPrinter* p) {
  *p << op.getOperationName();
  if (op.getNumOperands() > 0) {
    *p << ' ';
    p->printOperands(op.getOperands());
    *p << " : ";
    interleaveComma(op.getOperandTypes(), *p);
  }
}
}  // anonymous namespace

//===----------------------------------------------------------------------===//
// TableGen'd op method definitions
//===----------------------------------------------------------------------===//

#define GET_OP_CLASSES
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_device.cc.inc"

}  // namespace tf_device
}  // namespace mlir
