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

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/StringRef.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"  // from @llvm-project
#include "mlir/IR/Attributes.h"  // from @llvm-project
#include "mlir/IR/Builders.h"  // from @llvm-project
#include "mlir/IR/Operation.h"  // from @llvm-project
#include "mlir/IR/Visitors.h"  // from @llvm-project
#include "mlir/Transforms/Passes.h"  // from @llvm-project
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_device.h"
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_ops.h"
#include "tensorflow/dtensor/cc/constants.h"
#include "tensorflow/dtensor/cc/tensor_layout.h"
#include "tensorflow/dtensor/mlir/dtensor_mlir_passes.h"
#include "tensorflow/dtensor/mlir/ir/tf_dtensor.h"

namespace tensorflow {
namespace dtensor {

namespace {
#define GEN_PASS_DEF_DTENSORDESIGNATERESOURCEHANDLEMESH
#include "tensorflow/dtensor/mlir/dtensor_passes.h.inc"

mlir::LogicalResult SetMeshForResourceCreatingCluster(
    mlir::tf_device::ClusterOp cluster, mlir::OpBuilder* builder) {
  auto result = cluster.walk([](mlir::Operation* op) {
    if (llvm::isa<mlir::TF::VarHandleOp, mlir::TF::DestroyResourceOp>(op))
      return mlir::WalkResult::interrupt();
    return mlir::WalkResult::advance();
  });

  if (!result.wasInterrupted()) return mlir::success();

  const auto& cluster_ops = cluster.GetBody().without_terminator();

  bool has_single_tf_op =
      llvm::count_if(cluster_ops, [](auto& operation) {
        return !llvm::isa<mlir::TF::DTensorLayout>(&operation);
      }) == 1;

  if (!has_single_tf_op) {
    return cluster.emitOpError(
        "cluster containing tf.VarHandleOp/DestroyResourceOp must contain "
        "single operation and a terminator");
  }

  if (!cluster->hasAttr(kMeshAttr)) {
    cluster->setAttr(kMeshAttr, builder->getStringAttr(Mesh::kEmptyMeshString));
  }
  return mlir::success();
}

struct DTensorDesignateResourceHandleMesh
    : public impl::DTensorDesignateResourceHandleMeshBase<
          DTensorDesignateResourceHandleMesh> {
  void runOnOperation() override {
    mlir::MLIRContext& context = getContext();
    mlir::OpBuilder builder(&context);

    auto walk_result =
        getOperation().walk([&](mlir::tf_device::ClusterOp cluster) {
          if (mlir::failed(
                  SetMeshForResourceCreatingCluster(cluster, &builder)))
            return mlir::WalkResult::interrupt();
          return mlir::WalkResult::advance();
        });

    if (walk_result.wasInterrupted()) return signalPassFailure();
  }
};

}  // namespace

std::unique_ptr<mlir::OperationPass<mlir::func::FuncOp>>
CreateDTensorDesignateResourceHandleMesh() {
  return std::make_unique<DTensorDesignateResourceHandleMesh>();
}

}  // namespace dtensor
}  // namespace tensorflow
