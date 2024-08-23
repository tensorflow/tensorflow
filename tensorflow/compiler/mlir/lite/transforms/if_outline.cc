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

#include <memory>
#include <string>

#include "llvm/ADT/SetVector.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"  // from @llvm-project
#include "mlir/IR/Block.h"  // from @llvm-project
#include "mlir/IR/Builders.h"  // from @llvm-project
#include "mlir/IR/BuiltinOps.h"  // from @llvm-project
#include "mlir/IR/BuiltinTypes.h"  // from @llvm-project
#include "mlir/IR/Location.h"  // from @llvm-project
#include "mlir/IR/MLIRContext.h"  // from @llvm-project
#include "mlir/IR/Region.h"  // from @llvm-project
#include "mlir/IR/SymbolTable.h"  // from @llvm-project
#include "mlir/Pass/Pass.h"  // from @llvm-project
#include "mlir/Support/LLVM.h"  // from @llvm-project
#include "mlir/Support/TypeID.h"  // from @llvm-project
#include "mlir/Transforms/RegionUtils.h"  // from @llvm-project
#include "tensorflow/compiler/mlir/lite/ir/tfl_ops.h"
#include "tensorflow/compiler/mlir/lite/transforms/passes.h"
#include "tensorflow/compiler/mlir/lite/utils/utils.h"
#include "tensorflow/compiler/mlir/op_or_arg_name_mapper.h"

namespace mlir {
namespace TFL {
namespace {
#define GEN_PASS_DEF_IFOUTLINEPASS
#include "tensorflow/compiler/mlir/lite/transforms/passes.h.inc"

// This pass outlines the body region of the TFL IfOp into functions and
// replaces the regions with calls to these outlined functions.
class IfOutlinePass : public impl::IfOutlinePassBase<IfOutlinePass> {
 public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(IfOutlinePass)
  explicit IfOutlinePass() = default;

 private:
  void runOnOperation() override;

  // Outlines the regions of the IfOp's body and insert function
  // calls instead,
  void OutlineIf(IfOp if_op);

  // Get unique name by using the loc to name mapping.
  std::string GetName(Operation* op, StringRef suffix);

  tensorflow::OpOrArgLocNameMapper mapper_;
};

std::string IfOutlinePass::GetName(Operation* op, StringRef suffix) {
  return (mapper_.GetUniqueName(op) + suffix).str();
}

// Returns whether a Region is already outlined (e.g., only consists a call op
// and a yield op).
bool IsRegionAlreadyOutlined(Region& region) {
  auto it = region.front().begin();
  return isa<func::CallOp>(*it++) && isa<YieldOp>(*it);
}

func::FuncOp CreateOutlineFuncAndEraseRegion(
    StringRef name, Region& region, const llvm::SetVector<Value>& extern_values,
    ArrayRef<Type> types, Location loc) {
  MLIRContext* context = loc.getContext();
  OpBuilder func_builder(context);
  FunctionType type;
  SmallVector<Type> result_types;
  auto operands = region.front().getTerminator()->getOperandTypes();
  result_types.append(operands.begin(), operands.end());
  type = FunctionType::get(context, types, result_types);

  // Create outlined function and move region body to it.
  auto outlined_func = func_builder.create<func::FuncOp>(loc, name, type);
  outlined_func.getBody().takeBody(region);
  Region& func_region = outlined_func.getBody();

  // Replace all external uses with block args and update uses.
  Block& block = func_region.front();
  for (Value value : extern_values) {
    auto arg = block.addArgument(value.getType(), loc);
    replaceAllUsesInRegionWith(value, arg, func_region);
  }
  // Replace yield op with return.
  Operation* yield_op = outlined_func.getBody().front().getTerminator();
  OpBuilder return_builder(yield_op);
  return_builder.create<func::ReturnOp>(yield_op->getLoc(),
                                        yield_op->getOperands());
  yield_op->erase();

  SymbolTable(region.getParentOfType<ModuleOp>()).insert(outlined_func);
  outlined_func.setPrivate();
  return outlined_func;
}

// Replace region with call to outline function.
void ReplaceRegionWithCall(StringRef name, Region& region,
                           const llvm::SetVector<Value>& extern_values,
                           ArrayRef<Type> types, Location loc) {
  auto func =
      CreateOutlineFuncAndEraseRegion(name, region, extern_values, types, loc);
  OpBuilder b(region);

  // The body of the region is empty/has been outlined into the function.
  auto block = b.createBlock(&region);
  SmallVector<Value> new_operands;
  for (Type t : types.drop_back(extern_values.size())) {
    new_operands.push_back(block->addArgument(t, loc));
  }
  new_operands.append(extern_values.begin(), extern_values.end());
  auto call = b.create<func::CallOp>(loc, func, new_operands);
  b.create<YieldOp>(loc, call.getResults());
}

void IfOutlinePass::OutlineIf(IfOp if_op) {
  if (IsRegionAlreadyOutlined(if_op.getThenRegion()) &&
      IsRegionAlreadyOutlined(if_op.getElseRegion()))
    return;
  // Collect external values used by taking the union of all values defined
  // above the regions. Use same signature of function call for both regions.
  llvm::SetVector<Value> extern_values;
  for (auto* region : {&if_op.getThenRegion(), &if_op.getElseRegion()}) {
    getUsedValuesDefinedAbove(*region, extern_values);
  }
  // Collect new types.
  SmallVector<Type> types;
  for (auto value : extern_values) {
    types.push_back(value.getType());
  }
  ReplaceRegionWithCall(GetName(if_op.getOperation(), "_then"),
                        if_op.getThenRegion(), extern_values, types,
                        if_op.getLoc());
  ReplaceRegionWithCall(GetName(if_op.getOperation(), "_else"),
                        if_op.getElseRegion(), extern_values, types,
                        if_op.getLoc());
}

void IfOutlinePass::runOnOperation() {
  getOperation().walk([&](mlir::TFL::IfOp if_op) { OutlineIf(if_op); });
}
}  // namespace

// Creates an instance of the TensorFlow Lite dialect IfOp outline pass.
std::unique_ptr<OperationPass<ModuleOp>> CreateIfOutlinePass() {
  return std::make_unique<IfOutlinePass>();
}

}  // namespace TFL
}  // namespace mlir
