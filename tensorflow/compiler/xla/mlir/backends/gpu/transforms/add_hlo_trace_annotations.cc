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

#include <memory>
#include <string>
#include <utility>

#include "mlir/Dialect/Func/IR/FuncOps.h"  // from @llvm-project
#include "mlir/IR/MLIRContext.h"  // from @llvm-project
#include "mlir/IR/SymbolTable.h"  // from @llvm-project
#include "tensorflow/compiler/xla/mlir/backends/gpu/transforms/passes.h"
#include "tensorflow/compiler/xla/mlir/runtime/ir/rt_dialect.h"
#include "tensorflow/compiler/xla/translate/mhlo_to_hlo/location_exporter.h"

namespace xla {
namespace gpu {

#define GEN_PASS_DEF_ADDHLOTRACEANNOTATIONSPASS
#include "tensorflow/compiler/xla/mlir/backends/gpu/transforms/passes.h.inc"

using namespace mlir;  // NOLINT

using xla::runtime::HloTraceAttr;

class AddHloTraceAnnotationsPass
    : public impl::AddHloTraceAnnotationsPassBase<AddHloTraceAnnotationsPass> {
  void runOnOperation() override;

  void getDependentDialects(DialectRegistry& registry) const override {
    registry.insert<runtime::RuntimeDialect>();
  }
};

//===----------------------------------------------------------------------===//

void AddHloTraceAnnotationsPass::runOnOperation() {
  MLIRContext* ctx = &getContext();

  ModuleOp module = getOperation();
  SymbolTable sym_table(module);

  // Get a unique mhlo id from the top level module.
  auto uid = module->getAttrOfType<IntegerAttr>("mhlo.unique_id");
  int64_t program_id = uid ? uid.getValue().getZExtValue() : -1;

  // XLA HLO -> MLIR export encodes module name in the location.
  std::string module_name =
      mlir::mhlo::GetDebugNameFromLocation(module->getLoc());

  getOperation().walk([&](func::CallOp call) {
    // Check if the callee is a custom call.
    auto callee = sym_table.lookup<func::FuncOp>(call.getCallee());
    if (!callee->hasAttr("rt.custom_call")) return;

    // HLO operation name is encoded in the operation location.
    std::string hlo_op = mlir::mhlo::GetDebugNameFromLocation(call->getLoc());
    auto annotation = HloTraceAttr::get(ctx, hlo_op, module_name, program_id);
    call->setAttr("rt.trace", annotation);
  });
}

std::unique_ptr<mlir::OperationPass<mlir::ModuleOp>>
createAddHloTraceAnnotationsPass() {
  return std::make_unique<AddHloTraceAnnotationsPass>();
}

}  // namespace gpu
}  // namespace xla
