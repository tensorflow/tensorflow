/* Copyright 2024 The TensorFlow Authors. All Rights Reserved.

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

#include "xla/service/gpu/fusions/mlir/ir/xla_gpu_ops.h"

#include "llvm/ADT/TypeSwitch.h"  // IWYU pragma: keep
#include "mlir/Dialect/Func/IR/FuncOps.h"  // from @llvm-project
#include "mlir/IR/Builders.h"  // from @llvm-project  // IWYU pragma: keep
#include "mlir/IR/DialectImplementation.h"  // from @llvm-project  // IWYU pragma: keep
#include "mlir/IR/MLIRContext.h"  // from @llvm-project  // IWYU pragma: keep
#include "mlir/IR/PatternMatch.h"  // from @llvm-project  // IWYU pragma: keep
#include "mlir/IR/SymbolTable.h"  // from @llvm-project
#include "mlir/IR/TypeUtilities.h"  // from @llvm-project  // IWYU pragma: keep
#include "xla/service/gpu/fusions/mlir/ir/xla_gpu_dialect.cc.inc"

namespace xla {
namespace gpu {

void XlaGpuDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "xla/service/gpu/fusions/mlir/ir/xla_gpu_ops.cc.inc"
#undef GET_OP_LIST
      >();
}

mlir::LogicalResult PureCallOp::verifySymbolUses(
    mlir::SymbolTableCollection &symbolTable) {
  auto callee = getCalleeAttr();
  auto function =
      symbolTable.lookupNearestSymbolFrom<mlir::func::FuncOp>(*this, callee);
  if (!function) {
    return emitError("'f' attribute refers to an undefined function: ")
           << callee;
  }

  int func_arg_count = function.getFunctionType().getNumInputs();
  int arg_count = getOperands().size();

  if (arg_count != func_arg_count) {
    return emitError() << "argument count mismatch: 'operands' has "
                       << arg_count << " arguments, but '" << callee
                       << "' expects " << func_arg_count;
  }

  return mlir::success();
}

}  // namespace gpu
}  // namespace xla

#define GET_OP_CLASSES
#include "xla/service/gpu/fusions/mlir/ir/xla_gpu_ops.cc.inc"
