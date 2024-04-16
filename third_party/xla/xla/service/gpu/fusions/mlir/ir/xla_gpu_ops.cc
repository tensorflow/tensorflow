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
#include "mlir/Support/LLVM.h"  // from @llvm-project
#include "mlir/Transforms/InliningUtils.h"  // from @llvm-project
#include "xla/service/gpu/fusions/mlir/ir/xla_gpu_dialect.cc.inc"

namespace xla {
namespace gpu {
namespace {
struct XlaGpuInlinerInterface : public mlir::DialectInlinerInterface {
  using DialectInlinerInterface::DialectInlinerInterface;
  // Returns true if the given operation 'callable', that implements the
  // 'CallableOpInterface', can be inlined into the position given call
  // operation 'call', that is registered to the current dialect and implements
  // the `CallOpInterface`. 'wouldBeCloned' is set to true if the region of the
  // given 'callable' is set to be cloned during the inlining process, or false
  // if the region is set to be moved in-place (i.e. no duplicates would be
  // created).
  bool isLegalToInline(mlir::Operation *call, mlir::Operation *callable,
                       bool wouldBeCloned) const final {
    if (!wouldBeCloned) {
      // If no duplicate would be created, 'call' is likely the only caller of
      // 'callable'.
      return true;
    }
    // Otherwise, inline only if the called function is small. We could
    // theoretically also inline if there is no other caller in the function
    // that contains the callee that has a call path to the callable, but that
    // is more expensive to check.
    auto func_op = mlir::dyn_cast<mlir::func::FuncOp>(callable);
    if (!func_op) {
      return false;
    }
    auto region = func_op.getCallableRegion();
    if (!region) {
      return false;
    }
    const int kMaxOperationsToInline = 8;
    return region->front().getOperations().size() <= kMaxOperationsToInline;
  }
  // Returns true if the given operation 'op', that is registered to this
  // dialect, can be inlined into the given region, false otherwise.
  // 'wouldBeCloned' is set to true if the given 'op' is set to be cloned
  // during the inlining process, or false if the operation is set to be moved
  // in-place(i.e. no duplicates would be created). 'valueMapping' contains any
  // remapped values from within the 'src' region. This can be used to examine
  // what values may potentially replace the operands to 'op'.
  bool isLegalToInline(mlir::Operation *op, mlir::Region *dest,
                       bool wouldBeCloned,
                       mlir::IRMapping &valueMapping) const final {
    // We allow any op from the xla_gpu dialect to be inlined.
    return true;
  }
};
}  // namespace

void XlaGpuDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "xla/service/gpu/fusions/mlir/ir/xla_gpu_ops.cc.inc"
#undef GET_OP_LIST
      >();
  addInterfaces<XlaGpuInlinerInterface>();
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

//===----------------------------------------------------------------------===//
// AllocateSharedOp
//===----------------------------------------------------------------------===//

void AllocateSharedOp::getAsmResultNames(
    llvm::function_ref<void(mlir::Value, mlir::StringRef)> setNameFn) {
  setNameFn(getResult(), "shmem");
}

//===----------------------------------------------------------------------===//
// AtomicRMWOp
//===----------------------------------------------------------------------===//

void AtomicRMWOp::getAsmResultNames(
    llvm::function_ref<void(mlir::Value, mlir::StringRef)> setNameFn) {
  setNameFn(getResult(), "atomic_rmw");
}

using mlir::OpBuilder;
using mlir::OperationState;
using mlir::RankedTensorType;
using mlir::Region;
using mlir::Type;
using mlir::Value;
using mlir::ValueRange;

void AtomicRMWOp::build(OpBuilder &builder, OperationState &result,
                        Value tensor, ValueRange ivs) {
  OpBuilder::InsertionGuard g(builder);
  result.addOperands(tensor);
  result.addOperands(ivs);
  result.addTypes(tensor.getType());

  auto tensor_type = llvm::cast<RankedTensorType>(tensor.getType());
  Region *body = result.addRegion();
  builder.createBlock(body);
  body->addArgument(tensor_type.getElementType(), tensor.getLoc());
}

//===----------------------------------------------------------------------===//
// PureCallOp
//===----------------------------------------------------------------------===//

void PureCallOp::getAsmResultNames(
    llvm::function_ref<void(mlir::Value, mlir::StringRef)> setNameFn) {
  for (auto result : getResults()) {
    setNameFn(result, "pure_call");
  }
}

//===----------------------------------------------------------------------===//
// SyncThreadsOp
//===----------------------------------------------------------------------===//

void SyncThreadsOp::getAsmResultNames(
    llvm::function_ref<void(mlir::Value, mlir::StringRef)> setNameFn) {
  for (auto result : getResults()) {
    setNameFn(result, "synced_tensor");
  }
}

}  // namespace gpu
}  // namespace xla

#define GET_OP_CLASSES
#include "xla/service/gpu/fusions/mlir/ir/xla_gpu_ops.cc.inc"
