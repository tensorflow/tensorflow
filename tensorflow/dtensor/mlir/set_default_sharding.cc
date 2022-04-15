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

#include "llvm/ADT/StringRef.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"  // from @llvm-project
#include "mlir/IR/Attributes.h"  // from @llvm-project
#include "mlir/IR/Builders.h"  // from @llvm-project
#include "mlir/IR/Operation.h"  // from @llvm-project
#include "mlir/Transforms/Passes.h"  // from @llvm-project
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_device.h"
#include "tensorflow/compiler/xla/client/sharding_builder.h"
#include "tensorflow/dtensor/mlir/dtensor_mlir_passes.h"
#include "tensorflow/dtensor/mlir/dtensor_mlir_passes_classes.h"
#include "tensorflow/dtensor/mlir/ir/tf_dtensor.h"

namespace tensorflow {
namespace dtensor {
namespace {

// Assigns inputs/outputs for TPU computation to logical core 0.
void SetDefaultSharding(mlir::tf_device::ClusterFuncOp cluster,
                        mlir::OpBuilder* builder) {
  const std::string logical_core_0_sharding =
      xla::sharding_builder::AssignDevice(0).SerializeAsString();

  llvm::SmallVector<llvm::StringRef, 4> input_sharding(cluster.getNumOperands(),
                                                       logical_core_0_sharding);
  llvm::SmallVector<llvm::StringRef, 4> output_sharding(
      cluster.getNumResults(), logical_core_0_sharding);

  cluster->setAttr("input_sharding_configuration",
                   builder->getStrArrayAttr(input_sharding));
  cluster->setAttr("output_sharding_configuration",
                   builder->getStrArrayAttr(output_sharding));
}

// MLIR pass that sets xla sharding of TPU computation input/outputs to
// maximally assigned to logical core 0.
struct DTensorSetDefaultSharding
    : public DTensorSetDefaultShardingBase<DTensorSetDefaultSharding> {
  void runOnOperation() override {
    mlir::MLIRContext& context = getContext();
    mlir::OpBuilder builder(&context);

    getOperation().walk([&](mlir::tf_device::ClusterFuncOp cluster_func) {
      // Skip non-tpu device cluster_func.
      auto replicate_attr =
          cluster_func->getAttrOfType<mlir::StringAttr>("_tpu_replicate");
      if (!replicate_attr) return;

      SetDefaultSharding(cluster_func, &builder);
    });
  }
};

}  // namespace

std::unique_ptr<mlir::OperationPass<mlir::func::FuncOp>>
CreateDTensorSetDefaultSharding() {
  return std::make_unique<DTensorSetDefaultSharding>();
}

}  // namespace dtensor
}  // namespace tensorflow
