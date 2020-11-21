/* Copyright 2020 The TensorFlow Authors. All Rights Reserved.

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

// Converts DeviceIndex to constant device.

#include "mlir/Dialect/StandardOps/IR/Ops.h"  // from @llvm-project
#include "mlir/IR/Attributes.h"  // from @llvm-project
#include "mlir/IR/Builders.h"  // from @llvm-project
#include "mlir/IR/Operation.h"  // from @llvm-project
#include "mlir/IR/PatternMatch.h"  // from @llvm-project
#include "mlir/Pass/Pass.h"  // from @llvm-project
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_ops.h"
#include "tensorflow/compiler/mlir/tensorflow/transforms/passes.h"

namespace mlir {
namespace TF {
namespace {

// Folds the DeviceIndex op to a constant value. The DeviceIndex return the
// index of the device the op should run on. The user can use this to provide
// different op specializations. E.g.,
//
// ```mlir
//  %1 = "tf.DeviceIndex"()
//          {device = "", device_names = ["CPU", "GPU"]} : () -> tensor<i32>
//  %4 = "tf.Case"(%1, %arg0, %arg1)
//          {branches = [@foo, @baz], output_shapes = [#tf.shape<>]} :
//            (tensor<i32>, tensor<f32>, tensor<f32>) -> tensor<f32>
// ```
//
// Shows an example where there are 2 different functions which could be
// executed to produce the same values but with different functions optimized
// for CPU or GPU.
struct DeviceIndexSelector
    : public PassWrapper<DeviceIndexSelector, OperationPass<FuncOp>> {
  void runOnOperation() override;
};

}  // namespace

void DeviceIndexSelector::runOnOperation() {
  FuncOp func = getOperation();
  // Convert all the DeviceIndex ops to constant values.
  func.getBody().walk([](TF::DeviceIndexOp op) {
    // This just selects the default in all cases where DeviceIndex feeds into
    // tf.Case. This could be enhanced to have some sort of policy in the
    // future.
    OpBuilder b(op);
    RankedTensorType type = RankedTensorType::get({}, b.getIntegerType(32));
    int index = op.device_names().size();
    for (auto use : op.getOperation()->getUsers()) {
      // Skip if it doesn't feed into case. Alternatively this could always
      // return the CPU device index if it exists.
      if (!isa<TF::CaseOp>(use)) return;
    }
    DenseElementsAttr attr =
        DenseElementsAttr::get(type, b.getI32IntegerAttr(index));
    auto constant = b.create<ConstantOp>(op.getLoc(), type, attr);
    op.replaceAllUsesWith(constant.getOperation());
    op.erase();
  });
}

// Creates an instance of the TensorFlow DeviceIndex selector pass.
std::unique_ptr<OperationPass<FuncOp>> CreateDeviceIndexSelectorPass() {
  return std::make_unique<DeviceIndexSelector>();
}

static PassRegistration<DeviceIndexSelector> pass(
    "tf-device-index-selector", "Fold tf.DeviceIndex to constant");

}  // namespace TF
}  // namespace mlir
