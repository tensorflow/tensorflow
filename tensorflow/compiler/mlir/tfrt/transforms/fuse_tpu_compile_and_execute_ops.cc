/* Copyright 2021 The TensorFlow Authors. All Rights Reserved.

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

#include "llvm/ADT/SmallVector.h"
#include "mlir/IR/BuiltinTypes.h"  // from @llvm-project
#include "mlir/Transforms/Passes.h"  // from @llvm-project
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_ops.h"

namespace tensorflow {
namespace {

// This pass rewrites tf._TPUCompileMlirOp and tf.TPUExecuteOp into a single
// tf.TPUCompileMlirAndExecuteOp. Also it removes the unnecessary
// TPUCompileSucceededAssertOp.
class FuseTpuCompileAndExecutePass
    : public mlir::PassWrapper<FuseTpuCompileAndExecutePass,
                               mlir::FunctionPass> {
 public:
  llvm::StringRef getArgument() const final {
    return "tfrt-fuse-tpu-compile-and-execute-ops";
  }
  llvm::StringRef getDescription() const final {
    return "Fuse TPU Ops according to TFRT's requirements.";
  }

  void runOnFunction() override {
    auto func = getFunction();

    // remove TPUCompileSucceededAssertOp
    func.walk([&](mlir::Operation *op) {
      if (llvm::isa<mlir::TF::TPUCompileSucceededAssertOp>(op)) {
        op->erase();
      }
    });

    llvm::SmallVector<mlir::TF::TPUExecuteOp, 4> tpu_execute_ops;
    func.walk([&](mlir::Operation *op) {
      if (auto exec_op = llvm::dyn_cast<mlir::TF::TPUExecuteOp>(op)) {
        tpu_execute_ops.push_back(exec_op);
      }
    });

    mlir::OpBuilder builder(&func.body());
    for (auto exec_op : tpu_execute_ops) {
      auto compile_cache_entry = exec_op.key();
      auto compile_op = ::llvm::dyn_cast<mlir::TF::_TPUCompileMlirOp>(
          compile_cache_entry.getDefiningOp());
      if (!compile_op) {
        exec_op.emitOpError("could not get the _TPUCompileMlirOp");
        signalPassFailure();
        return;
      }

      builder.setInsertionPointAfter(compile_op);
      llvm::SmallVector<mlir::Type, 4> output_types;
      output_types.push_back(mlir::RankedTensorType::get(
          {3}, builder.getType<mlir::TF::StringType>()));
      output_types.insert(output_types.end(), exec_op.getResultTypes().begin(),
                          exec_op.getResultTypes().end());
      auto compile_and_execute_op =
          builder.create<mlir::TF::TPUCompileMlirAndExecuteOp>(
              exec_op.getLoc(), output_types, exec_op.args(),
              compile_op.mlir_module(), compile_op.metadata());

      exec_op.replaceAllUsesWith(compile_and_execute_op.results());
      for (auto program_result : compile_op.program()) {
        program_result.replaceAllUsesWith(
            compile_and_execute_op.rendezvous_key_base());
      }

      assert(exec_op.use_empty());
      exec_op.erase();
      assert(compile_op.use_empty());
      compile_op.erase();
    }
  }
};

}  // namespace

namespace tfrt_compiler {

std::unique_ptr<mlir::OperationPass<mlir::FuncOp>>
CreateFuseTpuCompileAndExecutePass() {
  return std::make_unique<FuseTpuCompileAndExecutePass>();
}

static mlir::PassRegistration<FuseTpuCompileAndExecutePass>
    fuse_tpu_compile_and_execute_ops_pass;

}  // namespace tfrt_compiler

}  // namespace tensorflow
