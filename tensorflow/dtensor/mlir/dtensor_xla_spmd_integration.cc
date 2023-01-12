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

#include "mlir/Dialect/Func/IR/FuncOps.h"  // from @llvm-project
#include "mlir/IR/BuiltinOps.h"  // from @llvm-project
#include "mlir/Pass/Pass.h"  // from @llvm-project
#include "mlir/Support/LogicalResult.h"  // from @llvm-project
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_device.h"
#include "tensorflow/compiler/xla/xla_data.pb.h"
#include "tensorflow/dtensor/cc/xla_spmd/layout_to_xla_sharding.h"
#include "tensorflow/dtensor/mlir/ir/tf_dtensor.h"
#include "tensorflow/dtensor/mlir/op_utils.h"

namespace tensorflow {
namespace dtensor {
namespace {
#define GEN_PASS_DEF_DTENSORXLASPMDINTEGRATION
#include "tensorflow/dtensor/mlir/dtensor_passes.h.inc"

mlir::LogicalResult SetHloShardingForInputsAndOps(mlir::ModuleOp module,
                                                  mlir::OpBuilder builder) {
  module.walk([&](mlir::TF::DTensorLayout layout_op) {
    if (!layout_op.getLayout().mesh().use_xla_spmd()) {
      layout_op.emitOpError(
          "Found a layout operation that is not on XLA SPMD mesh during XLA "
          "SPMD integration.");
    }
    StatusOr<::xla::OpSharding> xla_sharding =
        ConvertLayoutToXlaOpSharding(layout_op.getLayout());

    if (!xla_sharding.ok())
      module.emitError(xla_sharding.status().error_message());

    mlir::Value operand = layout_op.getOperand();

    if (mlir::BlockArgument block_arg =
            operand.dyn_cast_or_null<mlir::BlockArgument>()) {
      mlir::func::FuncOp func_op =
          layout_op->getParentOfType<mlir::func::FuncOp>();
      if (!func_op) {
        module.emitError(
            "Error finding surrounding FuncOp during "
            "DTensorXlaSpmdIntegration.");
      }
      func_op.setArgAttr(
          block_arg.getArgNumber(), kXlaShardingAttr,
          builder.getStringAttr(xla_sharding->SerializeAsString()));
    } else if (mlir::Operation* producing_op = operand.getDefiningOp()) {
      producing_op->setAttr(
          kXlaShardingAttr,
          builder.getStringAttr(xla_sharding->SerializeAsString()));
    }
  });
  return mlir::success();
}

mlir::LogicalResult SetHloShardingForOutputs(mlir::ModuleOp module,
                                             mlir::OpBuilder builder) {
  // Set output attributes
  module.walk([&](mlir::func::ReturnOp return_op) {
    for (auto return_index = 0; return_index < return_op.getNumOperands();
         ++return_index) {
      if (auto layout_op = llvm::dyn_cast_or_null<mlir::TF::DTensorLayout>(
              return_op->getOperand(return_index).getDefiningOp())) {
        StatusOr<::xla::OpSharding> xla_sharding =
            ConvertLayoutToXlaOpSharding(layout_op.getLayout());

        if (!xla_sharding.ok())
          module.emitError(xla_sharding.status().error_message());

        mlir::func::FuncOp func_op =
            layout_op->getParentOfType<mlir::func::FuncOp>();
        if (!func_op) {
          module.emitError(
              "Error finding surrounding FuncOp during "
              "DTensorXlaSpmdIntegration.");
        }

        func_op.setResultAttr(
            return_index, kXlaShardingAttr,
            builder.getStringAttr(xla_sharding->SerializeAsString()));
      }
    }
  });
  return mlir::success();
}

struct DTensorXlaSpmdIntegration
    : public impl::DTensorXlaSpmdIntegrationBase<DTensorXlaSpmdIntegration> {
  void runOnOperation() override {
    mlir::MLIRContext& context = getContext();
    mlir::OpBuilder builder(&context);
    mlir::ModuleOp module = getOperation();

    if (mlir::failed(SetHloShardingForInputsAndOps(module, builder))) {
      return signalPassFailure();
    }

    if (mlir::failed(ReplaceAuxiliaryDTensorLayoutOpsWithIdentity(module))) {
      return signalPassFailure();
    }

    if (mlir::failed(SetHloShardingForOutputs(module, builder))) {
      return signalPassFailure();
    }
    RemoveDTensorLayoutOps(module, /*remove_xla_spmd_layouts=*/true);
  };
};
}  // namespace
std::unique_ptr<mlir::OperationPass<mlir::ModuleOp>>
CreateDTensorXlaSpmdIntegration() {
  return std::make_unique<DTensorXlaSpmdIntegration>();
}
}  // namespace dtensor
}  // namespace tensorflow
