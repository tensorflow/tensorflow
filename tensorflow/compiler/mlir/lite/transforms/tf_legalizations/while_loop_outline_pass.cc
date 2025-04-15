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
#include "tensorflow/compiler/mlir/lite/transforms/tf_legalizations/while_loop_outline_pass.h"

#include <string>

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/Support/Casting.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"  // from @llvm-project
#include "mlir/IR/Builders.h"  // from @llvm-project
#include "mlir/IR/BuiltinOps.h"  // from @llvm-project
#include "mlir/IR/BuiltinTypes.h"  // from @llvm-project
#include "mlir/IR/Location.h"  // from @llvm-project
#include "mlir/IR/MLIRContext.h"  // from @llvm-project
#include "mlir/IR/Matchers.h"  // from @llvm-project
#include "mlir/IR/Region.h"  // from @llvm-project
#include "mlir/IR/SymbolTable.h"  // from @llvm-project
#include "mlir/IR/TypeUtilities.h"  // from @llvm-project
#include "mlir/Support/LLVM.h"  // from @llvm-project
#include "mlir/Transforms/RegionUtils.h"  // from @llvm-project
#include "tensorflow/compiler/mlir/lite/ir/tfl_ops.h"
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_ops.h"

namespace mlir {
namespace TFL {
namespace {

// Returns whether the WhileOp is already outlined (e.g., only consists of calls
// to functions).
bool IsAlreadyOutlined(WhileOp while_op) {
  auto just_call = [](Region& region) {
    auto it = region.front().begin();
    if (!isa<func::CallOp>(*it)) return false;
    ++it;
    if (!isa<YieldOp>(*it)) return false;
    return true;
  };
  return just_call(while_op.getBody()) && just_call(while_op.getCond());
}

bool IsCompatibleTypeWithTFLCastOp(Type type) {
  auto elemType = getElementTypeOrSelf(type);
  // F16, F32, F64, BF16 types are allowed.
  if (elemType.isBF16() || elemType.isF16() || elemType.isF32() ||
      elemType.isF64())
    return true;

  // I1, I4, I8, I16, I32, I64 types are allowed.
  if (elemType.isInteger(1) || elemType.isInteger(4) || elemType.isInteger(8) ||
      elemType.isInteger(16) || elemType.isInteger(32) ||
      elemType.isInteger(64))
    return true;

  // Complex<F<32>> is allowed.
  if (mlir::isa<ComplexType>(elemType) &&
      mlir::cast<ComplexType>(elemType).getElementType().isF32())
    return true;

  // QUINT8 and UI8 are allowed.
  if (mlir::isa<TF::Quint8Type>(elemType) ||
      (elemType.isInteger(8) && mlir::cast<IntegerType>(elemType).isUnsigned()))
    return true;

  return false;
}

func::FuncOp CreateOutlineFunc(StringRef name, Region& region,
                               bool passthru_extra_args, int num_loop_carried,
                               const llvm::SetVector<Value>& extern_values,
                               const SmallVectorImpl<Type>& types,
                               Location loc) {
  MLIRContext* context = loc.getContext();
  OpBuilder builder(context);
  FunctionType type;
  if (passthru_extra_args) {
    type = FunctionType::get(context, types, types);
  } else {
    SmallVector<Type, 4> result_types;
    auto operands = region.front().getTerminator()->getOperandTypes();
    result_types.append(operands.begin(), operands.end());
    type = FunctionType::get(context, types, result_types);
  }

  auto outlined_func = builder.create<func::FuncOp>(loc, name, type);
  outlined_func.getBody().takeBody(region);
  Region& func_region = outlined_func.getBody();

  // Replace all external uses with block args and update uses.
  llvm::SmallVector<Value, 4> new_args;
  new_args.reserve(extern_values.size());
  Block& block = func_region.front();
  for (Value value : extern_values) {
    auto arg = block.addArgument(value.getType(), loc);
    replaceAllUsesInRegionWith(value, arg, func_region);
    new_args.push_back(arg);
  }

  // Replace yield op with return.
  Operation* yield_op = outlined_func.getBody().front().getTerminator();
  OpBuilder b(yield_op);
  llvm::SmallVector<Value, 4> args;
  auto loop_carried_yield_operands =
      yield_op->getOperands().take_front(num_loop_carried);
  args.reserve(loop_carried_yield_operands.size() + new_args.size());
  if (passthru_extra_args) {
    // Add operands of yield to the return, inserting casts if needed.
    for (auto it : llvm::zip_first(loop_carried_yield_operands, types)) {
      auto value = std::get<0>(it);
      auto type = std::get<1>(it);
      if (value.getType() == type) {
        args.push_back(value);
      } else {
        if (IsCompatibleTypeWithTFLCastOp(value.getType()) &&
            IsCompatibleTypeWithTFLCastOp(type)) {
          auto cast = b.create<CastOp>(yield_op->getLoc(), type, value);
          args.push_back(cast);
        } else {
          auto cast = b.create<TF::CastOp>(yield_op->getLoc(), type, value);
          args.push_back(cast);
        }
      }
    }
    args.append(new_args.begin(), new_args.end());
  } else {
    args.append(yield_op->operand_begin(), yield_op->operand_end());
  }
  b.create<func::ReturnOp>(yield_op->getLoc(), args);
  yield_op->erase();
  SymbolTable(region.getParentOfType<ModuleOp>()).insert(outlined_func);
  outlined_func.setPrivate();
  return outlined_func;
}

// Replace region with call to outline function.
void ReplaceRegionWithCall(StringRef name, Region& region,
                           bool passthru_extra_args, int num_loop_carried,
                           const llvm::SetVector<Value>& extern_values,
                           const SmallVectorImpl<Type>& types, Location loc) {
  auto func = CreateOutlineFunc(name, region, passthru_extra_args,
                                num_loop_carried, extern_values, types, loc);
  OpBuilder b(region);
  // The body of the region is empty/has been outlined into the function.
  auto block = b.createBlock(&region);
  SmallVector<Value, 4> new_operands;
  new_operands.reserve(types.size());
  for (Type t : llvm::ArrayRef(types).drop_back(extern_values.size()))
    new_operands.push_back(block->addArgument(t, loc));
  for (Value v : extern_values) new_operands.push_back(v);
  auto call = b.create<func::CallOp>(loc, func, new_operands);
  b.create<YieldOp>(loc, call.getResults());
}
}  // namespace

void WhileOutlinePass::OutlineWhile(WhileOp while_op) {
  OpBuilder builder(&getContext());
  // Collect external values used.
  llvm::SetVector<Value> extern_values;

  // The basic block arguments correspond to values that are loop carried, while
  // all those post are loop independent. Initialize extern_values with while_op
  // not loop carried operands.
  auto num_loop_carried = while_op.getCond().getNumArguments();
  auto not_carried_operands =
      while_op.getOperands().drop_front(num_loop_carried);
  extern_values.insert(not_carried_operands.begin(),
                       not_carried_operands.end());
  auto old_extern_values_size = extern_values.size();

  llvm::SmallVector<Region*, 2> regions{&while_op.getCond(),
                                        &while_op.getBody()};
  for (const auto& it : llvm::enumerate(regions)) {
    llvm::SetVector<Value> region_extern_values;
    getUsedValuesDefinedAbove(*it.value(), region_extern_values);

    // Sink down constants (including quantized constant) into the functions.
    for (auto extern_value : region_extern_values) {
      if (!matchPattern(extern_value, m_Constant()) &&
          !llvm::dyn_cast_or_null<TFL::QConstOp>(
              extern_value.getDefiningOp())) {
        extern_values.insert(extern_value);
        continue;
      }
      // Add constant at start of region.
      auto const_builder =
          OpBuilder(&it.value()->front(), it.value()->front().begin());
      auto const_value = const_builder.clone(*extern_value.getDefiningOp());
      replaceAllUsesInRegionWith(extern_value, const_value->getResult(0),
                                 *it.value());
    }
  }

  bool has_extra_extern_values = old_extern_values_size != extern_values.size();
  // If an extern value is already an operand post the loop carried operands,
  // then it need not be passed in again.
  // Compute all the extra operands that have to be added to the while.
  llvm::SetVector<Value> extra_operands;
  if (has_extra_extern_values) {
    auto new_extern =
        extern_values.getArrayRef().drop_front(old_extern_values_size);
    extra_operands.insert(new_extern.begin(), new_extern.end());
  }

  // Skip if already just calls.
  if (extra_operands.empty() && IsAlreadyOutlined(while_op)) return;

  // Collect new types.
  SmallVector<Type, 4> types;
  types.reserve(extra_operands.size() + while_op.getNumOperands());
  for (Type type : while_op.getCond().getArgumentTypes()) types.push_back(type);
  for (Value operand : extern_values) types.push_back(operand.getType());

  // Create outline function from region. Optional pass extra arguments through
  // to yield.
  ReplaceRegionWithCall(GetName(while_op.getOperation(), "_cond"),
                        while_op.getCond(), false, num_loop_carried,
                        extern_values, types, while_op.getLoc());
  ReplaceRegionWithCall(GetName(while_op.getOperation(), "_body"),
                        while_op.getBody(), true, num_loop_carried,
                        extern_values, types, while_op.getLoc());

  // If there are extern values used then the result type of the while has to
  // change, so replace with new while op.
  if (extra_operands.empty()) return;

  const int operands_size = while_op.getNumOperands() + extra_operands.size();
  SmallVector<Value, 4> operands;
  operands.reserve(operands_size);
  operands.append(while_op.getOperands().begin(), while_op.getOperands().end());
  operands.append(extra_operands.begin(), extra_operands.end());
  SmallVector<Type, 4> new_types;
  new_types.reserve(operands_size);
  new_types.append(while_op.getResultTypes().begin(),
                   while_op.getResultTypes().end());
  for (auto extra_operand : extra_operands)
    new_types.push_back(extra_operand.getType());

  auto new_while_op = OpBuilder(while_op).create<WhileOp>(
      while_op.getLoc(), new_types, operands, while_op->getAttrs());
  new_while_op.getCond().takeBody(while_op.getCond());
  new_while_op.getBody().takeBody(while_op.getBody());
  while_op.replaceAllUsesWith(
      new_while_op.getResults().take_front(while_op.getNumResults()));
  while_op.erase();
}

std::string WhileOutlinePass::GetName(Operation* op, StringRef suffix) {
  return (mapper_.GetUniqueName(op) + suffix).str();
}

void WhileOutlinePass::runOnOperation() {
  getOperation().walk(
      [&](mlir::TFL::WhileOp while_op) { OutlineWhile(while_op); });
}

}  // namespace TFL
}  // namespace mlir
