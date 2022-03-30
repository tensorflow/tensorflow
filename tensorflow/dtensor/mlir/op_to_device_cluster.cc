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

#include <string>

#include "llvm/Support/Casting.h"
#include "llvm/Support/FormatVariadic.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"  // from @llvm-project
#include "mlir/IR/Attributes.h"  // from @llvm-project
#include "mlir/IR/Builders.h"  // from @llvm-project
#include "mlir/IR/BuiltinOps.h"  // from @llvm-project
#include "mlir/IR/Operation.h"  // from @llvm-project
#include "mlir/IR/Visitors.h"  // from @llvm-project
#include "mlir/Pass/Pass.h"  // from @llvm-project
#include "mlir/Pass/PassManager.h"  // from @llvm-project
#include "mlir/Support/LogicalResult.h"  // from @llvm-project
#include "mlir/Transforms/Passes.h"  // from @llvm-project
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_device.h"
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_ops.h"
#include "tensorflow/compiler/mlir/tensorflow/transforms/passes.h"
#include "tensorflow/compiler/mlir/tensorflow/utils/error_util.h"
#include "tensorflow/dtensor/cc/constants.h"
#include "tensorflow/dtensor/cc/tensor_layout.h"
#include "tensorflow/dtensor/mlir/dtensor_dialect/ir/dialect.h"
#include "tensorflow/dtensor/mlir/dtensor_mlir_passes.h"
#include "tensorflow/dtensor/mlir/dtensor_mlir_passes_classes.h"
#include "tensorflow/dtensor/mlir/ir/tf_dtensor.h"
#include "tensorflow/dtensor/mlir/layout_parsing.h"

namespace tensorflow {
namespace dtensor {

namespace {

// Extracts mesh config from the Op.
// We currently hard extract mesh information from all the args and assume they
// are the same. This should not be the case when we have multiple functions.
mlir::LogicalResult WrapDeviceCluster(mlir::OpBuilder *builder,
                                      mlir::Operation *op) {
  // Create new tf_device.cluster op wrapping a single operation.
  builder->setInsertionPoint(op);
  auto cluster = builder->create<mlir::tf_device::ClusterOp>(
      op->getLoc(), op->getResultTypes());
  if (auto layout_op = llvm::dyn_cast<mlir::TF::DTensorLayout>(op)) {
    cluster->setAttr(kMeshAttr, builder->getStringAttr(
                                    layout_op.layout().mesh().ToString()));
  } else if (auto copy_to_mesh = llvm::dyn_cast<mlir::TF::CopyToMeshOp>(op)) {
    const std::string layout_string = copy_to_mesh.layout().str();
    auto layout_or = Layout::FromString(layout_string);
    if (!layout_or.ok())
      return op->emitOpError(
          llvm::formatv("Found tf.CopyToMesh Op with unparsable layout : {0}",
                        layout_string));

    cluster->setAttr(kMeshAttr,
                     builder->getStringAttr(layout_or->mesh().ToString()));
  } else {
    // If mesh configuration can be inferred from the op directly, use the mesh
    // information from op attribute directly. If op is not annotated with mesh
    // information, then mesh will be inferred in following
    // DTensorMeshPropagation pass and will be inferred from consumers or
    // operands.
    auto status_or_mesh = ExtractDeviceMeshFromOp(op);

    if (!status_or_mesh.ok())
      return op->emitOpError(
          llvm::formatv("failed to wrap to device cluster. {0}",
                        status_or_mesh.status().error_message()));

    const auto mesh_config = status_or_mesh.ValueOrDie();
    if (mesh_config)
      cluster->setAttr(kMeshAttr,
                       builder->getStringAttr(mesh_config->ToString()));
  }

  op->replaceAllUsesWith(cluster);

  cluster.body().push_back(new mlir::Block);

  builder->setInsertionPointToEnd(&cluster.GetBody());
  builder->create<mlir::tf_device::ReturnOp>(op->getLoc(), op->getResults());

  // Move `op` inside newly created `ClusterOp`.
  op->moveBefore(cluster.GetBody().getTerminator());

  return mlir::success();
}

// MLIR pass that wraps tf_device.cluster op to every TF op.
struct DTensorOpToDeviceClusterPass
    : public DTensorOpToDeviceClusterBase<DTensorOpToDeviceClusterPass> {
  void getDependentDialects(mlir::DialectRegistry &registry) const override {
    registry.insert<mlir::dtensor::DTensorDialect>();
    registry.insert<mlir::tf_device::TensorFlowDeviceDialect>();
  }

  void runOnOperation() override {
    mlir::MLIRContext &context = getContext();
    mlir::OpBuilder op_builder(&context);
    mlir::Dialect *tf =
        getContext().getLoadedDialect<mlir::TF::TensorFlowDialect>();

    auto walk_result = getOperation().walk([&](mlir::Operation *operation) {
      const auto op_dialect = operation->getDialect();
      // Only TF dialects are supported for layout propagation.
      if (op_dialect != tf) return mlir::WalkResult::advance();

      // For control flow operations, tf.yield ops exists and should not be
      // wrapped to tf_device.cluster as the op does not need to be transformed
      // in SPMD expansion and tf.If/tf.While op require all ops to terminate
      // with tf.Yield op. Wrapping yield op in tf_device.cluster invalidates
      // this invariant.
      if (llvm::isa<mlir::TF::YieldOp>(operation))
        return mlir::WalkResult::advance();

      if (mlir::failed(WrapDeviceCluster(&op_builder, operation)))
        return mlir::WalkResult::interrupt();
      return mlir::WalkResult::advance();
    });

    if (walk_result.wasInterrupted()) signalPassFailure();
  }
};

}  // namespace

std::unique_ptr<mlir::OperationPass<mlir::func::FuncOp>>
CreateDTensorOpToDeviceClusterPass() {
  return std::make_unique<DTensorOpToDeviceClusterPass>();
}

}  // namespace dtensor
}  // namespace tensorflow
