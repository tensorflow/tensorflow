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
void ImportXlaRegion(Region* src_region, Region* dest_region) {
  BlockAndValueMapping mapper;
  src_region->cloneInto(dest_region, mapper);
  dest_region->walk([&](mlir::ReturnOp op) -> void {
    OpBuilder builder(op);
    llvm::SmallVector<Value*, 4> operands(op.operands());
    auto tuple = builder.create<xla_hlo::TupleOp>(op.getLoc(), operands);
    builder.create<xla_hlo::ReturnOp>(op.getLoc(), tuple.getResult());
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
  auto conditional_result = builder.create<xla_hlo::ConditionalOp>(
      loc, result_type, op.cond(), tuple_input, tuple_input);

  // Import the regions for both the true and false cases. These regions
  // must be updated to tuple the return results together and use the xla hlo
  // return op.
  BlockAndValueMapping mapper;
  auto then_branch = module.lookupSymbol<mlir::FuncOp>(op.then_branch());
  auto else_branch = module.lookupSymbol<mlir::FuncOp>(op.else_branch());
  ImportXlaRegion(&then_branch.getBody(), &conditional_result.true_branch());
  ImportXlaRegion(&else_branch.getBody(), &conditional_result.false_branch());

  // De-tuple the results of the xla hlo conditional result.
  builder.setInsertionPointAfter(op);
  for (auto result_it : llvm::enumerate(op.getResults())) {
    auto get_tuple_value = builder.create<xla_hlo::GetTupleElementOp>(
        loc, conditional_result, result_it.index());
    result_it.value()->replaceAllUsesWith(get_tuple_value);
  }

  op.erase();
}
}  // namespace

void LegalizeTFControlFlow::runOnModule() {
  auto module = getModule();

  TypeConverter type_converter;
  module.walk([&](TF::IfOp op) -> void { LowerIf(op, module); });
}
}  // namespace xla_hlo
}  // namespace mlir

static PassRegistration<mlir::xla_hlo::LegalizeTFControlFlow> cfpass(
    "xla-legalize-tf-control-flow",
    "Legalize TensorFlow control flow to the XLA dialect");
