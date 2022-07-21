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

#include "tensorflow/compiler/mlir/lite/experimental/tac/transforms/device_transform_gpu.h"

#include <memory>
#include <utility>

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/raw_ostream.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"  // from @llvm-project
#include "mlir/IR/Attributes.h"  // from @llvm-project
#include "mlir/IR/Builders.h"  // from @llvm-project
#include "mlir/IR/BuiltinOps.h"  // from @llvm-project
#include "mlir/IR/BuiltinTypes.h"  // from @llvm-project
#include "mlir/IR/Value.h"  // from @llvm-project
#include "mlir/Pass/Pass.h"  // from @llvm-project
#include "mlir/Pass/PassRegistry.h"  // from @llvm-project
#include "mlir/Support/LLVM.h"  // from @llvm-project
#include "mlir/Support/LogicalResult.h"  // from @llvm-project
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"  // from @llvm-project
#include "tensorflow/compiler/mlir/lite/experimental/tac/common/targets.h"
#include "tensorflow/compiler/mlir/lite/experimental/tac/hardwares/gpu_hardware.h"
#include "tensorflow/compiler/mlir/lite/ir/tfl_ops.h"
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_ops.h"

namespace mlir {
namespace TFL {
namespace tac {
namespace {

struct DeviceTransformGPUPass
    : public mlir::PassWrapper<DeviceTransformGPUPass,
                               OperationPass<func::FuncOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(DeviceTransformGPUPass)

  llvm::StringRef getArgument() const final {
    return "tfl-device-transform-gpu";
  }
  llvm::StringRef getDescription() const final {
    return "Suitable transformation for gpu only.";
  }
  void getDependentDialects(DialectRegistry& registry) const override {
    registry.insert<TF::TensorFlowDialect>();
  }
  void runOnOperation() override;
};

void DeviceTransformGPUPass::runOnOperation() {
  auto func = getOperation();
  auto* ctx = &getContext();
  RewritePatternSet patterns = GetHardwareRewritePatternsGPU(ctx);
  (void)applyPatternsAndFoldGreedily(func, std::move(patterns));
}

}  // namespace

RewritePatternSet GetHardwareRewritePatternsGPU(MLIRContext* context) {
  GpuHardware gpu_hardware;
  return gpu_hardware.GetTransformations(context);
}

std::unique_ptr<OperationPass<func::FuncOp>> CreateDeviceTransformGPUPass() {
  return std::make_unique<DeviceTransformGPUPass>();
}

static PassRegistration<DeviceTransformGPUPass> pass;

}  // namespace tac
}  // namespace TFL
}  // namespace mlir
