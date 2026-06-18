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
#include <optional>

#include "llvm/Support/Casting.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"  // from @llvm-project
#include "mlir/IR/BuiltinOps.h"  // from @llvm-project
#include "mlir/IR/Operation.h"  // from @llvm-project
#include "mlir/IR/Visitors.h"  // from @llvm-project
#include "mlir/Pass/Pass.h"  // from @llvm-project
#include "mlir/Support/LogicalResult.h"  // from @llvm-project
#include "xla/xla_data.pb.h"
#include "tensorflow/dtensor/cc/xla_spmd/layout_to_xla_sharding.h"
#include "tensorflow/dtensor/mlir/dtensor_dialect/ir/dialect.h"
#include "tensorflow/dtensor/mlir/ir/tf_dtensor.h"

namespace tensorflow {
namespace dtensor {
namespace {
#define GEN_PASS_DECL_DTENSORSETHLOSHARDINGPASS
#define GEN_PASS_DEF_DTENSORSETHLOSHARDINGPASS
#include "tensorflow/dtensor/mlir/dtensor_passes.h.inc"

mlir::LogicalResult SetHloShardingForInputs(mlir::ModuleOp module,
                                            mlir::OpBuilder builder,
                                            bool check_layout_use_xla_spmd) {
  auto result = module.walk([&](mlir::func::FuncOp func_op)
                                -> mlir::WalkResult {
    for (int arg_index = 0; arg_index < func_op.getNumArguments();
         ++arg_index) {
      mlir::BlockArgument arg = func_op.getArgument(arg_index);
      for (const auto* user : arg.getUsers()) {
        auto layout_op = llvm::dyn_cast<mlir::TF::DTensorLayout>(*user);
        if (!layout_op) {
          continue;
        }

        if (check_layout_use_xla_spmd &&
            !layout_op.getLayout().mesh().use_xla_spmd()) {
          return layout_op.emitOpError(
              "Found a layout operation that is not on XLA SPMD mesh during "
              "XLA SPMD integration.");
        }

        StatusOr<xla::OpSharding> xla_sharding =
            ConvertLayoutToXlaOpSharding(layout_op.getLayout());
        if (!xla_sharding.ok()) {
          return layout_op.emitError(xla_sharding.status().message());
        }
        func_op.setArgAttr(
            arg_index, kXlaShardingAttr,
            builder.getStringAttr(xla_sharding->SerializeAsString()));
        break;
      }
    }
    return mlir::WalkResult::advance();
  });

  return mlir::failure(result.wasInterrupted());
}

mlir::LogicalResult SetHloShardingForOutputs(mlir::ModuleOp module,
                                             mlir::OpBuilder builder) {
  // Set output attributes
  auto result =
      module.walk([&](mlir::func::ReturnOp return_op) -> mlir::WalkResult {
        for (int return_index = 0; return_index < return_op.getNumOperands();
             ++return_index) {
          auto layout_op = llvm::dyn_cast_or_null<mlir::TF::DTensorLayout>(
              return_op->getOperand(return_index).getDefiningOp());
          if (!layout_op) continue;

          StatusOr<xla::OpSharding> xla_sharding =
              ConvertLayoutToXlaOpSharding(layout_op.getLayout());

          if (!xla_sharding.ok()) {
            return module.emitError(xla_sharding.status().message());
          }

          mlir::func::FuncOp func_op =
              layout_op->getParentOfType<mlir::func::FuncOp>();
          if (!func_op) {
            return module.emitError(
                "Error finding surrounding FuncOp during "
                "DTensorXlaSpmdIntegration.");
          }

          func_op.setResultAttr(
              return_index, kXlaShardingAttr,
              builder.getStringAttr(xla_sharding->SerializeAsString()));
        }
        return mlir::WalkResult::advance();
      });
  return mlir::failure(result.wasInterrupted());
}

class DTensorSetHloShardingPass
    : public impl::DTensorSetHloShardingPassBase<DTensorSetHloShardingPass> {
 public:
  using DTensorSetHloShardingPassBase::DTensorSetHloShardingPassBase;

  explicit DTensorSetHloShardingPass(
      std::optional<bool> check_layout_use_xla_spmd) {
    if (check_layout_use_xla_spmd.has_value()) {
      check_layout_use_xla_spmd_ = *check_layout_use_xla_spmd;
    }
  }

  void runOnOperation() override {
    mlir::MLIRContext& context = getContext();
    mlir::OpBuilder builder(&context);
    mlir::ModuleOp module = getOperation();
    if (mlir::failed(SetHloShardingForInputs(
            module, builder, check_layout_use_xla_spmd_.getValue()))) {
      return signalPassFailure();
    }

    if (mlir::failed(SetHloShardingForOutputs(module, builder))) {
      return signalPassFailure();
    }
  }
};

}  // namespace

std::unique_ptr<mlir::OperationPass<mlir::ModuleOp>>
CreateDTensorSetHloShardingPass(std::optional<bool> check_layout_use_xla_spmd) {
  return std::make_unique<DTensorSetHloShardingPass>(check_layout_use_xla_spmd);
}

}  // namespace dtensor
}  // namespace tensorflow
