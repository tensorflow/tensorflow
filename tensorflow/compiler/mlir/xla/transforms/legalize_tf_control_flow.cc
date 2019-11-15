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

// This file implements logic for lowering TensorFlow dialect's control flow to
// the XLA dialect.

#include <cstddef>
#include <cstdint>
#include <iterator>
#include <numeric>

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/SmallSet.h"
#include "llvm/ADT/iterator_range.h"
#include "mlir/Dialect/StandardOps/Ops.h"  // TF:local_config_mlir
#include "mlir/IR/Attributes.h"  // TF:local_config_mlir
#include "mlir/IR/BlockAndValueMapping.h"  // TF:local_config_mlir
#include "mlir/IR/MLIRContext.h"  // TF:local_config_mlir
#include "mlir/IR/Module.h"  // TF:local_config_mlir
#include "mlir/IR/Operation.h"  // TF:local_config_mlir
#include "mlir/IR/StandardTypes.h"  // TF:local_config_mlir
#include "mlir/IR/TypeUtilities.h"  // TF:local_config_mlir
#include "mlir/IR/Types.h"  // TF:local_config_mlir
#include "mlir/Pass/Pass.h"  // TF:local_config_mlir
#include "mlir/Pass/PassRegistry.h"  // TF:local_config_mlir
#include "mlir/Transforms/DialectConversion.h"  // TF:local_config_mlir
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_ops.h"
#include "tensorflow/compiler/mlir/xla/ir/hlo_ops.h"
#include "tensorflow/compiler/mlir/xla/transforms/passes.h"
#include "tensorflow/core/util/tensor_format.h"

using mlir::PassRegistration;

namespace mlir {
namespace xla_hlo {
namespace {
class LegalizeTFControlFlow : public ModulePass<LegalizeTFControlFlow> {
 public:
  void runOnModule() override;
};
}  // namespace

std::unique_ptr<mlir::OpPassBase<mlir::ModuleOp>>
createLegalizeTFControlFlowPass() {
  return std::make_unique<LegalizeTFControlFlow>();
}

namespace {

void Detuple(Value* tuple, llvm::iterator_range<ResultIterator> replace,
             OpBuilder* builder) {
  // De-tuple the results of the xla hlo conditional result.
  for (auto result_it : llvm::enumerate(replace)) {
    auto get_tuple_value = builder->create<xla_hlo::GetTupleElementOp>(
        result_it.value()->getLoc(), tuple, result_it.index());
    result_it.value()->replaceAllUsesWith(get_tuple_value);
  }
}

// Imports the source region into the destination region. The XLA conditional
// operation only supports one argument per branch. Therefore any branch that
// requires additional arguments requires their values be tupled together. Then,
// to support multiple returns (as XLA only supports a single return value) the
// results of the conditional are tupled together.
void ImportXlaRegion(Region* src_region, Region* dest_region,
                     bool tuple_return = true) {
  BlockAndValueMapping mapper;
  src_region->cloneInto(dest_region, mapper);

  Block& entry_block = dest_region->front();
  OpBuilder builder(dest_region);
  llvm::SmallVector<Type, 5> arg_types;
  arg_types.reserve(entry_block.getNumArguments());
  for (auto arg : entry_block.getArguments()) {
    arg_types.push_back(arg->getType());
  }
  auto tuple_arg = entry_block.addArgument(builder.getTupleType(arg_types));
  for (int64_t i = 0, s = entry_block.getNumArguments() - 1; i < s; i++) {
    Value* val = entry_block.getArgument(0);
    auto extract =
        builder.create<GetTupleElementOp>(dest_region->getLoc(), tuple_arg, i);
    val->replaceAllUsesWith(extract);
    entry_block.eraseArgument(0);
  }

  dest_region->walk([&](mlir::ReturnOp op) -> void {
    OpBuilder builder(op);
    llvm::SmallVector<Value*, 4> operands(op.operands());
    if (tuple_return) {
      auto tuple = builder.create<xla_hlo::TupleOp>(op.getLoc(), operands);
      operands.assign({tuple.getResult()});
    }

    builder.create<xla_hlo::ReturnOp>(op.getLoc(), operands);
    op.erase();
  });
}

void LowerIf(TF::IfOp op, ModuleOp module) {
  Location loc = op.getLoc();
  OpBuilder builder(op);

  // XLA prefers tuple arguments for control flow due to XLA not supporting
  // multiple return values.
  SmallVector<Value*, 3> inputs(op.input());
  builder.setInsertionPoint(op);
  auto tuple_input = builder.create<xla_hlo::TupleOp>(loc, inputs);

  // Create the new conditional op with tuple inputs.
  SmallVector<Value*, 3> operands(op.getOperands());
  SmallVector<Type, 4> types(op.getResultTypes());
  auto result_type = builder.getTupleType(types);
  auto conditional = builder.create<xla_hlo::ConditionalOp>(
      loc, result_type, op.cond(), tuple_input, tuple_input);

  // Import the regions for both the true and false cases. These regions
  // must be updated to tuple the return results together and use the xla hlo
  // return op.
  BlockAndValueMapping mapper;
  auto then_branch = module.lookupSymbol<mlir::FuncOp>(op.then_branch());
  auto else_branch = module.lookupSymbol<mlir::FuncOp>(op.else_branch());
  ImportXlaRegion(&then_branch.getBody(), &conditional.true_branch());
  ImportXlaRegion(&else_branch.getBody(), &conditional.false_branch());

  // De-tuple the results of the xla hlo conditional result.
  builder.setInsertionPointAfter(op);
  Detuple(conditional.getResult(), op.getResults(), &builder);
  op.erase();
}

void LowerWhile(TF::WhileOp op, ModuleOp module) {
  Location loc = op.getLoc();
  OpBuilder builder(op);

  // XLA prefers tuple arguments for control flow due to XLA not supporting
  // multiple return values.
  SmallVector<Value*, 3> inputs(op.input());
  builder.setInsertionPoint(op);
  Value* tuple_input = builder.create<xla_hlo::TupleOp>(loc, inputs);

  // Create the new while op with tuple inputs.
  SmallVector<Value*, 3> operands(op.getOperands());
  SmallVector<Type, 4> types(op.getResultTypes());
  auto while_op = builder.create<xla_hlo::WhileOp>(
      loc, builder.getTupleType(types), tuple_input);

  // Import the regions for both the cond and body. These regions must be
  // updated to tuple the return results together and use the xla hlo return op.
  auto body_branch = module.lookupSymbol<mlir::FuncOp>(op.body());
  auto cond_branch = module.lookupSymbol<mlir::FuncOp>(op.cond());

  ImportXlaRegion(&body_branch.getBody(), &while_op.body());
  ImportXlaRegion(&cond_branch.getBody(), &while_op.cond(),
                  /*tuple_return=*/false);

  // De-tuple the results of the xla hlo while.
  builder.setInsertionPointAfter(op);
  Detuple(while_op.getResult(), op.getResults(), &builder);
  op.erase();
}
}  // namespace

void LegalizeTFControlFlow::runOnModule() {
  auto module = getModule();

  module.walk([&](TF::WhileOp op) -> void { LowerWhile(op, module); });
  module.walk([&](TF::IfOp op) -> void { LowerIf(op, module); });
}
}  // namespace xla_hlo
}  // namespace mlir

static PassRegistration<mlir::xla_hlo::LegalizeTFControlFlow> cfpass(
    "xla-legalize-tf-control-flow",
    "Legalize TensorFlow control flow to the XLA dialect");
