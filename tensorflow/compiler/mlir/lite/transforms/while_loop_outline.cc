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

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/Support/CommandLine.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"  // TF:llvm-project
#include "mlir/IR/Builders.h"  // TF:llvm-project
#include "mlir/IR/Identifier.h"  // TF:llvm-project
#include "mlir/IR/Location.h"  // TF:llvm-project
#include "mlir/IR/MLIRContext.h"  // TF:llvm-project
#include "mlir/IR/StandardTypes.h"  // TF:llvm-project
#include "mlir/IR/SymbolTable.h"  // TF:llvm-project
#include "mlir/Pass/Pass.h"  // TF:llvm-project
#include "mlir/Transforms/RegionUtils.h"  // TF:llvm-project
#include "tensorflow/compiler/mlir/lite/ir/tfl_ops.h"
#include "tensorflow/compiler/mlir/lite/transforms/passes.h"
#include "tensorflow/compiler/mlir/op_or_arg_name_mapper.h"

namespace mlir {
namespace TFL {
namespace {

// This pass outlines the cond/body region of the TFL WhileOp into functions and
// replaces the regions with calls to these outlined functions.
class WhileOutlinePass : public mlir::ModulePass<WhileOutlinePass> {
 public:
  explicit WhileOutlinePass() {}

 private:
  void runOnModule() override;

  // Outlines the regions of the WhileOp's cond and body and insert function
  // calls instead,
  void OutlineWhile(WhileOp while_op);

  // Get unique name by using the loc to name mapping.
  std::string GetName(Operation* op, StringRef suffix);

  tensorflow::OpOrArgLocNameMapper mapper_;
};
}  // namespace

std::string WhileOutlinePass::GetName(Operation* op, StringRef suffix) {
  return (mapper_.GetUniqueName(op) + suffix).str();
}

// Returns whether the WhileOp is already outlined (e.g., only consists of calls
// to functions).
static bool IsAlreadyOutlinedd(WhileOp while_op) {
  auto just_call = [](Region& region) {
    auto it = region.front().begin();
    if (!isa<CallOp>(*it)) return false;
    ++it;
    if (!isa<YieldOp>(*it)) return false;
    return true;
  };
  return just_call(while_op.body()) && just_call(while_op.cond());
}

void WhileOutlinePass::OutlineWhile(WhileOp while_op) {
  OpBuilder builder(&getContext());
  // Collect external values used.
  llvm::SetVector<Value> extern_values;

  // The basic block arguments correspond to values that are loop carried, while
  // all those post are loop independent. Initialize extern_values with while_op
  // not loop carried operands.
  auto num_loop_carried = while_op.cond().front().getNumArguments();
  auto not_carried_operands =
      while_op.getOperands().drop_front(num_loop_carried);
  extern_values.insert(not_carried_operands.begin(),
                       not_carried_operands.end());
  auto old_extern_values_size = extern_values.size();

  llvm::SmallVector<Region*, 2> regions{&while_op.cond(), &while_op.body()};
  for (auto it : llvm::enumerate(regions)) {
    llvm::SetVector<Value> region_extern_values;
    Value const_none = nullptr;
    getUsedValuesDefinedAbove(*it.value(), region_extern_values);

    // Sink down none type constants into the functions.
    for (auto extern_value : region_extern_values) {
      if (!extern_value.getType().isa<NoneType>()) {
        extern_values.insert(extern_value);
        continue;
      }
      if (!const_none) {
        // Add constant at start of region.
        auto const_builder =
            OpBuilder(&it.value()->front(), it.value()->front().begin());
        const_none = const_builder.create<ConstantOp>(
            while_op.getLoc(), extern_value.getType(),
            const_builder.getUnitAttr());
      }
      replaceAllUsesInRegionWith(extern_value, const_none, *it.value());
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
  if (extra_operands.empty() && IsAlreadyOutlinedd(while_op)) return;

  // Collect new types.
  SmallVector<Type, 4> types;
  types.reserve(extra_operands.size() + while_op.getNumOperands());
  for (BlockArgument ba : while_op.cond().front().getArguments())
    types.push_back(ba.getType());
  for (Value operand : extern_values) types.push_back(operand.getType());

  // Create outline function from region. Optional pass extra arguments through
  // to yield.
  SymbolTable symbol_table(getModule());
  auto create_outline_func = [&](StringRef name, Region& region,
                                 bool passthru_extra_args) {
    FunctionType type;
    if (passthru_extra_args) {
      type = FunctionType::get(types, types, &getContext());
    } else {
      SmallVector<Type, 4> result_types;
      auto operands = region.front().getTerminator()->getOperandTypes();
      result_types.append(operands.begin(), operands.end());
      type = FunctionType::get(types, result_types, &getContext());
    }

    auto outlined_func = builder.create<FuncOp>(while_op.getLoc(), name, type,
                                                ArrayRef<NamedAttribute>{});
    outlined_func.getBody().takeBody(region);
    Region& func_region = outlined_func.getBody();

    // Replace all external uses with block args and update uses.
    llvm::SmallVector<Value, 4> new_args;
    new_args.reserve(extern_values.size());
    Block& block = func_region.front();
    for (Value value : extern_values) {
      auto arg = block.addArgument(value.getType());
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
          auto cast = b.create<CastOp>(yield_op->getLoc(), type, value);
          args.push_back(cast);
        }
      }
      args.append(new_args.begin(), new_args.end());
    } else {
      args.append(yield_op->operand_begin(), yield_op->operand_end());
    }
    b.create<ReturnOp>(yield_op->getLoc(), args);
    yield_op->erase();
    symbol_table.insert(outlined_func);
    outlined_func.setVisibility(FuncOp::Visibility::Private);
    return outlined_func;
  };

  // Replace region with call to outline function.
  auto replace_with_call = [&](StringRef name, Region& region,
                               bool passthru_extra_args) {
    auto func = create_outline_func(name, region, passthru_extra_args);
    OpBuilder b(region);
    // The body of the region is empty/has been outlined into the function.
    auto block = b.createBlock(&region);
    SmallVector<Value, 4> new_operands;
    new_operands.reserve(types.size());
    for (Type t : llvm::makeArrayRef(types).drop_back(extern_values.size()))
      new_operands.push_back(block->addArgument(t));
    for (Value v : extern_values) new_operands.push_back(v);
    auto call = b.create<CallOp>(while_op.getLoc(), func, new_operands);
    b.create<YieldOp>(while_op.getLoc(), call.getResults());
  };

  replace_with_call(GetName(while_op.getOperation(), "_cond"), while_op.cond(),
                    false);
  replace_with_call(GetName(while_op.getOperation(), "_body"), while_op.body(),
                    true);

  // If there are extern values used then the result type of the while has to
  // change, so replace with new while op.
  if (extra_operands.empty()) return;

  Operation* op = while_op.getOperation();
  SmallVector<Value, 4> operands;
  SmallVector<Type, 4> new_types;
  operands.reserve(types.size());
  new_types.reserve(operands.size());
  auto add_operand = [&](Value v) {
    operands.push_back(v);
    new_types.push_back(v.getType());
  };
  for (auto operand : op->getOperands()) add_operand(operand);
  for (auto operand : extra_operands) add_operand(operand);

  Operation* new_op = OpBuilder(op).insert(Operation::create(
      op->getLoc(), op->getName(), new_types, operands, op->getAttrs(),
      /*successors=*/{}, /*numRegions=*/2,
      /*resizableOperandList=*/true));
  for (int i = 0; i < 2; ++i) new_op->getRegion(i).takeBody(op->getRegion(i));
  op->replaceAllUsesWith(new_op->getResults().take_front(op->getNumResults()));
  op->erase();
}

void WhileOutlinePass::runOnModule() {
  getModule().walk(
      [&](mlir::TFL::WhileOp while_op) { OutlineWhile(while_op); });
}

// Creates an instance of the TensorFlow Lite dialect WhileOp outline pass.
std::unique_ptr<OpPassBase<ModuleOp>> CreateWhileOutlinePass() {
  return std::make_unique<WhileOutlinePass>();
}

static PassRegistration<WhileOutlinePass> pass(
    "tfl-while-loop-outline", "Hoist while op regions into functions");

}  // namespace TFL
}  // namespace mlir
