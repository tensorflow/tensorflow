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
#include "tensorflow/compiler/mlir/tfrt/transforms/mlrt/ifrt_set_tpu_host_allocator.h"

#include <memory>
#include <vector>

#include "llvm/ADT/APInt.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Casting.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"  // from @llvm-project
#include "mlir/IR/Builders.h"  // from @llvm-project
#include "mlir/IR/BuiltinAttributes.h"  // from @llvm-project
#include "mlir/IR/DialectRegistry.h"  // from @llvm-project
#include "mlir/IR/Operation.h"  // from @llvm-project
#include "mlir/IR/Value.h"  // from @llvm-project
#include "mlir/IR/Visitors.h"  // from @llvm-project
#include "mlir/Pass/Pass.h"  // from @llvm-project
#include "mlir/Support/TypeID.h"  // from @llvm-project
#include "mlir/Transforms/DialectConversion.h"  // from @llvm-project
#include "tensorflow/compiler/mlir/tensorflow/ir/host_runtime/tfrt_ops.h"
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_dialect.h"
#include "tensorflow/compiler/mlir/tfrt/ir/mlrt/mlrt_dialect.h"
#include "tensorflow/compiler/mlir/tfrt/ir/mlrt/tf_mlrt_ops.h"
#include "tensorflow/compiler/mlir/tfrt/transforms/mlrt/mlrt_device_constants.h"

namespace tensorflow {
namespace mlrt_compiler {
namespace {

class IfrtSetTpuHostAllocatorPass
    : public mlir::PassWrapper<IfrtSetTpuHostAllocatorPass,
                               mlir::OperationPass<mlir::func::FuncOp>> {
 public:
  IfrtSetTpuHostAllocatorPass() = default;
  IfrtSetTpuHostAllocatorPass &operator=(const IfrtSetTpuHostAllocatorPass &) =
      delete;

  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(IfrtSetTpuHostAllocatorPass)

 private:
  void getDependentDialects(mlir::DialectRegistry &registry) const override {
    registry.insert<tensorflow::tf_mlrt::TensorflowMlrtDialect>();
    registry.insert<mlrt::compiler::MlrtDialect>();
  }

  llvm::StringRef getArgument() const final {
    return "tf-mlrt-ifrt-set-tpu-host-allocator";
  }

  llvm::StringRef getDescription() const final {
    return "Set input producer to IfrtCall to use Tpu Host Allocator";
  }
  void runOnOperation() override {
    mlir::func::FuncOp func = getOperation();
    mlir::OpBuilder builder(&getContext());

    llvm::SmallDenseSet<mlir::Operation *> producers;

    mlir::WalkResult walk_result = func.walk([&](mlir::TF::IfrtCallOp call) {
      std::vector<int> variable_arg_indices;
      variable_arg_indices.reserve(call.getVariableArgIndices().size());
      for (auto variable_index_attr : call.getVariableArgIndices()) {
        auto variable_index =
            llvm::dyn_cast_or_null<mlir::IntegerAttr>(variable_index_attr);
        if (!variable_index) {
          call->emitError()
              << "Expect variable_arg_indices to be integer, but get "
              << call.getVariableArgIndices();
          return mlir::WalkResult::interrupt();
        }
        variable_arg_indices.push_back(variable_index.getInt());
      }

      int variable_index = 0;
      for (int i = 0; i < call.getOperands().size(); ++i) {
        if (variable_index < variable_arg_indices.size() &&
            i == variable_arg_indices[variable_index]) {
          variable_index++;
          continue;
        }
        producers.insert(call.getOperands()[i].getDefiningOp());
      }
      return mlir::WalkResult::advance();
    });
    if (walk_result.wasInterrupted()) {
      return signalPassFailure();
    }

    for (auto *def : producers) {
      if (def && llvm::isa<mlir::TF::TensorFlowDialect>(def->getDialect())) {
        def->setAttr(kTfMlrtCustomDevice,
                     builder.getStringAttr(kTpuHostDevice));
      }
    }
  }
};

}  // namespace

std::unique_ptr<mlir::OperationPass<mlir::func::FuncOp>>
CreateIfrtSetTpuHostAllocatorPass() {
  return std::make_unique<IfrtSetTpuHostAllocatorPass>();
}

}  // namespace mlrt_compiler
}  // namespace tensorflow
