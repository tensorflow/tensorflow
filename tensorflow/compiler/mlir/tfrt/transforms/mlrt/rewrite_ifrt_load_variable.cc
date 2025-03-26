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

#include "tensorflow/compiler/mlir/tfrt/transforms/mlrt/rewrite_ifrt_load_variable.h"

#include <memory>
#include <vector>

#include "llvm/ADT/APInt.h"
#include "llvm/ADT/StringRef.h"
#include "mlir/IR/Builders.h"  // from @llvm-project
#include "mlir/IR/BuiltinAttributes.h"  // from @llvm-project
#include "mlir/IR/BuiltinOps.h"  // from @llvm-project
#include "mlir/IR/DialectRegistry.h"  // from @llvm-project
#include "mlir/IR/Operation.h"  // from @llvm-project
#include "mlir/IR/Value.h"  // from @llvm-project
#include "mlir/Pass/Pass.h"  // from @llvm-project
#include "mlir/Support/TypeID.h"  // from @llvm-project
#include "mlir/Transforms/DialectConversion.h"  // from @llvm-project
#include "tensorflow/compiler/mlir/tensorflow/ir/host_runtime/tfrt_ops.h"
#include "tensorflow/compiler/mlir/tfrt/ir/mlrt/mlrt_dialect.h"
#include "tensorflow/compiler/mlir/tfrt/ir/mlrt/tf_mlrt_ops.h"

namespace tensorflow {
namespace mlrt_compiler {
namespace {

class RewriteIfrtLoadVariablePass
    : public mlir::PassWrapper<RewriteIfrtLoadVariablePass,
                               mlir::OperationPass<mlir::ModuleOp>> {
 public:
  RewriteIfrtLoadVariablePass() = default;
  RewriteIfrtLoadVariablePass &operator=(const RewriteIfrtLoadVariablePass &) =
      delete;

  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(RewriteIfrtLoadVariablePass)

 private:
  void getDependentDialects(mlir::DialectRegistry &registry) const override {
    registry.insert<tensorflow::tf_mlrt::TensorflowMlrtDialect>();
    registry.insert<mlrt::compiler::MlrtDialect>();
  }

  llvm::StringRef getArgument() const final {
    return "tf-mlrt-rewrite-ifrt-load-variable";
  }

  llvm::StringRef getDescription() const final {
    return "Convert tf.IfrtLoadVariable to tf_mlrt.TFIfrtLoadVariable";
  }

  void runOnOperation() override {
    mlir::ModuleOp module = getOperation();
    mlir::OpBuilder builder(module);

    module->walk([&](mlir::TF::IfrtLoadVariableOp load_variable_op) {
      builder.setInsertionPoint(load_variable_op);

      std::vector<mlir::Type> result_types;
      result_types.push_back(load_variable_op.getArrayKey().getType());
      result_types.push_back(builder.getType<mlrt::compiler::FutureType>());
      auto mlrt_load_variable_op =
          builder.create<tf_mlrt::TFIfrtLoadVariableOp>(
              load_variable_op->getLoc(), result_types,
              load_variable_op->getOperands(), load_variable_op->getAttrs());
      for (auto user : load_variable_op.getTensorFuture().getUsers()) {
        builder.setInsertionPoint(user);
        auto await_op = builder.create<tf_mlrt::TFAwaitOp>(
            user->getLoc(), load_variable_op.getTensorFuture().getType(),
            mlrt_load_variable_op.getTensorFuture());
        user->replaceUsesOfWith(load_variable_op.getTensorFuture(),
                                await_op.getResult());
      }

      for (auto user : load_variable_op.getArrayKey().getUsers()) {
        user->replaceUsesOfWith(load_variable_op.getArrayKey(),
                                mlrt_load_variable_op.getArrayKey());
      }

      load_variable_op->erase();
    });
  }
};

}  // namespace

std::unique_ptr<mlir::OperationPass<mlir::ModuleOp>>
CreateRewriteIfrtLoadVariablePass() {
  return std::make_unique<RewriteIfrtLoadVariablePass>();
}

}  // namespace mlrt_compiler
}  // namespace tensorflow
