/* Copyright 2022 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/compiler/xla/mlir/runtime/ir/rt_ops.h"  // IWYU pragma: keep

#include <iterator>
#include <optional>

#include "llvm/ADT/STLExtras.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"  // from @llvm-project
#include "mlir/IR/Builders.h"  // from @llvm-project  // IWYU pragma: keep
#include "mlir/IR/BuiltinAttributes.h"  // from @llvm-project
#include "tensorflow/compiler/xla/mlir/runtime/ir/rt_interfaces.h"

namespace xla {
namespace runtime {

using namespace mlir;  // NOLINT

using llvm::Optional;

//===----------------------------------------------------------------------===//
// ExportOp
//===----------------------------------------------------------------------===//

void ExportOp::build(OpBuilder &builder, OperationState &result,
                     FunctionOpInterface function_ref) {
  result.addAttribute("function_ref", SymbolRefAttr::get(function_ref));
}

void ExportOp::build(OpBuilder &builder, OperationState &result,
                     FunctionOpInterface function_ref, unsigned ordinal) {
  build(builder, result, function_ref);
  result.addAttribute("ordinal", builder.getI32IntegerAttr(ordinal));
}

LogicalResult ExportOp::verifySymbolUses(SymbolTableCollection &symbolTable) {
  Operation *op = getOperation();
  auto func = symbolTable.lookupNearestSymbolFrom<FunctionOpInterface>(
      op, getFunctionRefAttr());

  // Function reference must reference a valid FuncOp operation.
  if (!func) {
    return op->emitError() << "func op named '" << getFunctionRef()
                           << "' not found for export";
  }

  return success();
}

Optional<unsigned> ExportOp::ordinal() {
  if (auto ordinal = getOrdinal()) return ordinal->getLimitedValue();
  return std::nullopt;
}

FunctionOpInterface ExportOp::exported(mlir::SymbolTable &sym_table) {
  return sym_table.lookupNearestSymbolFrom<FunctionOpInterface>(
      getOperation(), getFunctionRefAttr());
}

//===----------------------------------------------------------------------===//
// TraceOp
//===----------------------------------------------------------------------===//

void TraceOp::getSuccessorRegions(std::optional<unsigned> index,
                                  ArrayRef<Attribute> operands,
                                  SmallVectorImpl<RegionSuccessor> &regions) {
  // If the predecessor is the TraceOp, branch into the body.
  if (!index) {
    regions.push_back(RegionSuccessor(&getRegion()));
    return;
  }

  // Region branches back to the parent operation.
  regions.push_back(RegionSuccessor(getResults()));
}

LogicalResult TraceOp::verify() {
  if (getRegion().front().getNumArguments() > 0)
    return emitOpError("region cannot have any arguments");
  return success();
}

void TraceOp::build(OpBuilder &builder, OperationState &result,
                    TypeRange results, Value exec_ctx,
                    TraceAnnotationAttrInterface annotation,
                    function_ref<void(OpBuilder &, Location)> bodyBuilder) {
  result.addTypes(results);
  result.addOperands(exec_ctx);
  result.addAttribute("annotation", annotation);

  Region *bodyRegion = result.addRegion();
  Block &bodyBlock = bodyRegion->emplaceBlock();

  OpBuilder::InsertionGuard guard(builder);
  builder.setInsertionPointToStart(&bodyBlock);

  // Create the default terminator if the builder is not provided and if the
  // expected result is empty. Otherwise, leave this to the caller
  // because we don't know which values to return from the trace op.
  if (results.empty() && !bodyBuilder) {
    builder.create<YieldOp>(result.location, ValueRange());
  } else if (bodyBuilder) {
    bodyBuilder(builder, result.location);
  }
}

//===----------------------------------------------------------------------===//
// YieldOp
//===----------------------------------------------------------------------===//

MutableOperandRange YieldOp::getMutableSuccessorOperands(
    std::optional<unsigned> index) {
  return getArgumentsMutable();
}

}  // namespace runtime
}  // namespace xla

#define GET_OP_CLASSES
#include "tensorflow/compiler/xla/mlir/runtime/ir/rt_ops.cc.inc"
