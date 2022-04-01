/* Copyright 2020 The TensorFlow Authors. All Rights Reserved.

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

// This file defines the operations used in the tf_framework dialect.

#include "tensorflow/compiler/mlir/tools/kernel_gen/ir/tf_framework_ops.h"

#include "mlir/Dialect/Bufferization/IR/Bufferization.h"  // from @llvm-project
#include "mlir/Dialect/Func/IR/FuncOps.h"  // from @llvm-project
#include "mlir/Dialect/MemRef/IR/MemRef.h"  // from @llvm-project
#include "mlir/IR/Builders.h"  // from @llvm-project
#include "mlir/IR/DialectImplementation.h"  // from @llvm-project
#include "tensorflow/compiler/mlir/tools/kernel_gen/ir/tf_status.cc.inc"

// Generated dialect definitions.
#include "tensorflow/compiler/mlir/tools/kernel_gen/ir/tf_framework_dialect.cc.inc"

namespace mlir {
namespace kernel_gen {
namespace tf_framework {

void TFFrameworkDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "tensorflow/compiler/mlir/tools/kernel_gen/ir/tf_framework_ops.cc.inc"
      >();
  addTypes<JITCallableType, OpKernelContextType>();
}

/// Parse a type registered to this dialect.
Type TFFrameworkDialect::parseType(DialectAsmParser &parser) const {
  StringRef keyword;
  if (parser.parseKeyword(&keyword)) return Type();

  if (keyword == "op_kernel_context") {
    return OpKernelContextType::get(getContext());
  }
  if (keyword == "jit_callable") {
    return JITCallableType::get(getContext());
  }

  parser.emitError(parser.getNameLoc(), "unknown TF Framework type: ")
      << keyword;
  return Type();
}

/// Print a type registered to this dialect.
void TFFrameworkDialect::printType(Type type, DialectAsmPrinter &os) const {
  if (type.isa<OpKernelContextType>()) {
    os << "op_kernel_context";
    return;
  }
  if (type.isa<JITCallableType>()) {
    os << "jit_callable";
    return;
  }
  llvm_unreachable("unexpected TF Framework type kind");
}

//===----------------------------------------------------------------------===//
// TFAllocOp
//===----------------------------------------------------------------------===//
LogicalResult TFAllocOp::verify() {
  TFAllocOp op = *this;
  // Check that the total number of operands matches the number of dynamic
  // dimensions specified in the memref type.
  unsigned result_dyn_dims = op.getType().getNumDynamicDims();
  unsigned dyn_sizes_count = op.dyn_sizes().size();
  if (dyn_sizes_count != result_dyn_dims)
    return op.emitOpError()
           << "`dyn_sizes` count " << dyn_sizes_count
           << " does not match dynamic dimensions count in the result type"
           << op.getType();
  return success();
}

Optional<Operation *> TFAllocOp::buildDealloc(OpBuilder &builder, Value alloc) {
  auto funcop = alloc.getParentRegion()->getParentOfType<func::FuncOp>();
  return builder
      .create<TFDeallocOp>(alloc.getLoc(), funcop.getArgument(0), alloc)
      .getOperation();
}

Optional<Value> TFAllocOp::buildClone(OpBuilder &builder, Value alloc) {
  // TODO(herhut): We should have our own clone op if one of these survives.
  return builder.create<mlir::bufferization::CloneOp>(alloc.getLoc(), alloc)
      .getResult();
}

::tensorflow::error::Code ConvertAttrToEnumValue(ErrorCode error_code) {
  using ::tensorflow::error::Code;
  switch (error_code) {
    case ErrorCode::OK:
      return Code::OK;
    case ErrorCode::CANCELLED:
      return Code::CANCELLED;
    case ErrorCode::UNKNOWN:
      return Code::UNKNOWN;
    case ErrorCode::INVALID_ARGUMENT:
      return Code::INVALID_ARGUMENT;
    case ErrorCode::DEADLINE_EXCEEDED:
      return Code::DEADLINE_EXCEEDED;
    case ErrorCode::NOT_FOUND:
      return Code::NOT_FOUND;
    case ErrorCode::ALREADY_EXISTS:
      return Code::ALREADY_EXISTS;
    case ErrorCode::PERMISSION_DENIED:
      return Code::PERMISSION_DENIED;
    case ErrorCode::UNAUTHENTICATED:
      return Code::UNAUTHENTICATED;
    case ErrorCode::RESOURCE_EXHAUSTED:
      return Code::RESOURCE_EXHAUSTED;
    case ErrorCode::FAILED_PRECONDITION:
      return Code::FAILED_PRECONDITION;
    case ErrorCode::ABORTED:
      return Code::ABORTED;
    case ErrorCode::OUT_OF_RANGE:
      return Code::OUT_OF_RANGE;
    case ErrorCode::UNIMPLEMENTED:
      return Code::UNIMPLEMENTED;
    case ErrorCode::INTERNAL:
      return Code::INTERNAL;
    case ErrorCode::UNAVAILABLE:
      return Code::UNAVAILABLE;
    case ErrorCode::DATA_LOSS:
      return Code::DATA_LOSS;
  }
}

}  // namespace tf_framework
}  // namespace kernel_gen
}  // namespace mlir

#define GET_OP_CLASSES
#include "tensorflow/compiler/mlir/tools/kernel_gen/ir/tf_framework_ops.cc.inc"
