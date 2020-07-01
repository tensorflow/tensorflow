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

// This file defines the operations used in the XLA dialect.

#include "tensorflow/compiler/mlir/xla/ir/lhlo_ops.h"

#include <assert.h>
#include <stddef.h>
#include <stdint.h>

#include "llvm/ADT/APFloat.h"
#include "llvm/ADT/APInt.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/FormatVariadic.h"
#include "mlir/IR/Attributes.h"  // from @llvm-project
#include "mlir/IR/Builders.h"  // from @llvm-project
#include "mlir/IR/Dialect.h"  // from @llvm-project
#include "mlir/IR/Location.h"  // from @llvm-project
#include "mlir/IR/MLIRContext.h"  // from @llvm-project
#include "mlir/IR/OpDefinition.h"  // from @llvm-project
#include "mlir/IR/OpImplementation.h"  // from @llvm-project
#include "mlir/IR/Operation.h"  // from @llvm-project
#include "mlir/IR/OperationSupport.h"  // from @llvm-project
#include "mlir/IR/PatternMatch.h"  // from @llvm-project
#include "mlir/IR/StandardTypes.h"  // from @llvm-project
#include "mlir/IR/TypeUtilities.h"  // from @llvm-project
#include "mlir/IR/Types.h"  // from @llvm-project
#include "mlir/IR/Value.h"  // from @llvm-project
#include "tensorflow/compiler/mlir/xla/ir/lhlo_ops.h.inc"

namespace mlir {
#include "tensorflow/compiler/mlir/xla/ir/lhlo_structs.cc.inc"
namespace xla_lhlo {

XlaLhloDialect::XlaLhloDialect(MLIRContext *context)
    : Dialect(getDialectNamespace(), context) {
  addOperations<
#define GET_OP_LIST
#include "tensorflow/compiler/mlir/xla/ir/lhlo_ops.cc.inc"
      >();
}

//===----------------------------------------------------------------------===//
// StaticMemRefCastOp
//===----------------------------------------------------------------------===//

Value StaticMemRefCastOp::getViewSource() { return *getODSOperands(0).begin(); }

static LogicalResult Verify(StaticMemRefCastOp op) {
  if (!op.operand().getType().cast<ShapedType>().hasStaticShape())
    return op.emitOpError("operand must have static shape");
  if (!op.getType().hasStaticShape())
    return op.emitOpError("result must have static shape");
  return success();
}

//===----------------------------------------------------------------------===//
// DynamicMemRefCastOp
//===----------------------------------------------------------------------===//

Value DynamicMemRefCastOp::getViewSource() {
  return *getODSOperands(0).begin();
}

static LogicalResult Verify(DynamicMemRefCastOp op) {
  // Check if `sizes` and `strides` args are compatible with the result type.
  if (op.sizes().size() != op.getType().getRank())
    return op.emitOpError(
        "`sizes` args count must be equal to the rank of the output memref");
  return success();
}

#define GET_OP_CLASSES
#include "tensorflow/compiler/mlir/xla/ir/lhlo_ops.cc.inc"

// TODO(cheshire): Support folding, reuse code from hlo_ops.cc.

void FusionOp::build(OpBuilder &builder, OperationState &result,
                     ArrayRef<NamedAttribute> attributes) {
  result.addAttributes(attributes);
  Region *bodyRegion = result.addRegion();
  FusionOp::ensureTerminator(*bodyRegion, builder, result.location);
}

}  // namespace xla_lhlo
}  // namespace mlir
