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

#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"  // from @llvm-project
#include "mlir/IR/Attributes.h"  // from @llvm-project
#include "mlir/IR/Builders.h"  // from @llvm-project
#include "mlir/IR/BuiltinOps.h"  // from @llvm-project
#include "mlir/IR/MLIRContext.h"  // from @llvm-project
#include "mlir/IR/Operation.h"  // from @llvm-project
#include "mlir/Pass/Pass.h"  // from @llvm-project
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_device.h"
#include "xla/hlo/builder/sharding_builder.h"
#include "tensorflow/dtensor/cc/constants.h"

namespace tensorflow {
namespace dtensor {

namespace {
#define GEN_PASS_DEF_DTENSORSETDEFAULTSHARDING
#include "tensorflow/dtensor/mlir/dtensor_passes.h.inc"

// Sets the input/output sharding for the device cluster.
void SetDefaultSharding(mlir::tf_device::ClusterFuncOp cluster,
                        mlir::OpBuilder* builder, bool multi_device_mode) {
  std::string sharding;
  if (multi_device_mode) {
    // Assigns inputs/outputs for TPU computation to "replicated" so XLA
    // does not change their shape or attempt further SPMD expansion.
    sharding = xla::sharding_builder::Replicate().SerializeAsString();
  } else {
    // Assigns inputs/outputs for TPU computation to logical core 0.
    sharding = xla::sharding_builder::AssignDevice(0).SerializeAsString();
  }

  llvm::SmallVector<llvm::StringRef, 4> input_sharding(cluster.getNumOperands(),
                                                       sharding);
  llvm::SmallVector<llvm::StringRef, 4> output_sharding(cluster.getNumResults(),
                                                        sharding);

  cluster->setAttr("input_sharding_configuration",
                   builder->getStrArrayAttr(input_sharding));
  cluster->setAttr("output_sharding_configuration",
                   builder->getStrArrayAttr(output_sharding));
}

// MLIR pass that sets xla sharding of TPU computation input/outputs to
// maximally assigned to logical core 0.
struct DTensorSetDefaultSharding
    : public impl::DTensorSetDefaultShardingBase<DTensorSetDefaultSharding> {
  void runOnOperation() override {
    mlir::MLIRContext& context = getContext();
    mlir::OpBuilder builder(&context);

    getOperation().walk([&](mlir::tf_device::ClusterFuncOp cluster_func) {
      // Skip non-tpu device cluster_func.
      auto replicate_attr =
          cluster_func->getAttrOfType<mlir::StringAttr>("_tpu_replicate");
      if (!replicate_attr) return;

      auto module_op = cluster_func->getParentOfType<mlir::ModuleOp>();
      auto multi_device_attr = module_op->getAttrOfType<mlir::BoolAttr>(
          dtensor::kEnableMultiDeviceMode);
      bool multi_device_mode =
          multi_device_attr && multi_device_attr.getValue();

      SetDefaultSharding(cluster_func, &builder, multi_device_mode);
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
