/* Copyright 2026 The OpenXLA Authors.

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

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Block.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Operation.h"
#include "mlir/Interfaces/FunctionInterfaces.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/LLVM.h"
#include "xla/backends/gpu/codegen/emitters/ir/xla_gpu_ops.h"
#include "xla/backends/gpu/codegen/emitters/transforms/passes.h"
#include "xla/frontend_attributes.h"
#include "triton/Dialect/Triton/IR/Dialect.h"

namespace xla {
namespace gpu {

#define GEN_PASS_DEF_INSERTPDLPASS
#include "xla/backends/gpu/codegen/emitters/transforms/passes.h.inc"

namespace {

mlir::Operation* GetTopLevelOpInFunction(mlir::FunctionOpInterface func,
                                         mlir::Operation* op) {
  mlir::Operation* top_level_op = op;
  while (top_level_op->getParentOp() != func.getOperation()) {
    top_level_op = top_level_op->getParentOp();
  }
  return top_level_op;
}

class InsertPDLPass : public impl::InsertPDLPassBase<InsertPDLPass> {
 public:
  void runOnOperation() override {
    mlir::ModuleOp module = getOperation();
    const bool do_pdl_launch = module->hasAttr(kXlaPdlLaunch);

    module.walk([&](mlir::FunctionOpInterface func) {
      if (func.getFunctionBody().empty()) {
        return;
      }
      if (auto func_op =
              mlir::dyn_cast<mlir::func::FuncOp>(func.getOperation());
          func_op && func_op.isPrivate()) {
        return;
      }

      mlir::Operation* last_dot = nullptr;
      func.walk([&](mlir::Operation* op) {
        if (mlir::isa<mlir::triton::DotOp>(op)) {
          last_dot = op;
        }
      });

      // Insert PDL wait unconditionally at every kernel start.
      mlir::Block& entry_block = func.getFunctionBody().front();
      auto builder_at_begin = mlir::OpBuilder::atBlockBegin(&entry_block);
      xla::gpu::PdlWaitOp::create(builder_at_begin, func.getLoc());

      // Insert PDL launch at the top level after last dot only in annotated
      // kernels.
      if (!do_pdl_launch || last_dot == nullptr) {
        return;
      }
      mlir::Operation* insertion_point =
          GetTopLevelOpInFunction(func, last_dot);
      mlir::OpBuilder builder(insertion_point);
      builder.setInsertionPointAfter(insertion_point);
      xla::gpu::PdlLaunchDependentsOp::create(builder, last_dot->getLoc());
    });
  }
};

}  // namespace
}  // namespace gpu
}  // namespace xla
