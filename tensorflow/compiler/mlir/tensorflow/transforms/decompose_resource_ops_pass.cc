/* Copyright 2019 The TensorFlow Authors. All Rights Reserved.

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

#include "mlir/IR/PatternMatch.h"  // TF:local_config_mlir
#include "mlir/Pass/Pass.h"  // TF:local_config_mlir
#include "tensorflow/compiler/mlir/tensorflow/transforms/decompose_resource_ops.h"

namespace mlir {
namespace TFDevice {
namespace {

// A pass that decomposes composite resource operations into primitive ones like
// ReadVariableOp, AssignVariableOp and other computations to facilitate
// transformations like resource op lifting.
//
// For example:
//
// tf.AssignAddVariableOp(%res, %0)
//
// Becomes
//
// %res_val = tf.ReadVariableOp(%res)
// %1 = tf.AddV2(%res_val, %0)
// tf.AssignVariableOp(%res, %1)
struct DecomposeResourceOps : public FunctionPass<DecomposeResourceOps> {
  void runOnFunction() override {
    // Add lowering patterns to the list.
    OwningRewritePatternList patterns;
    mlir::TF::PopulateDecomposeResourceOpsPatterns(&getContext(), &patterns);

    applyPatternsGreedily(getFunction(), patterns);
  }
};

}  // namespace

std::unique_ptr<OpPassBase<FuncOp>> CreateDecomposeResourceOpsPass() {
  return std::make_unique<DecomposeResourceOps>();
}

}  // namespace TFDevice
}  // namespace mlir

static mlir::PassRegistration<mlir::TFDevice::DecomposeResourceOps> pass(
    "tf-device-decompose-resource-ops",
    "Decompose composite resource variable operations into primitive "
    "Read/AssignVariableOp and raw computation");
