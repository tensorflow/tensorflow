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

#include <cstddef>
#include <cstdint>
#include <string>
#include <utility>

#include "mlir/IR/Attributes.h"  // from @llvm-project
#include "mlir/IR/Visitors.h"  // from @llvm-project
#include "mlir/Pass/Pass.h"  // from @llvm-project
#include "mlir/Pass/PassManager.h"  // from @llvm-project
#include "mlir/Support/LLVM.h"  // from @llvm-project
#include "mlir/Transforms/Passes.h"  // from @llvm-project
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_ops.h"
#include "tensorflow/compiler/mlir/tensorflow/utils/visitor_util.h"

namespace tensorflow {
namespace {

std::string get_stage_description(const WalkStage &stage) {
  if (stage.IsBeforeAllRegions()) return "before all regions";
  if (stage.IsAfterAllRegions()) return "after all regions";
  return "before region #" + std::to_string(stage.GetNextRegion());
}

// A pass that annotates each operation with an remarks that include a unique
// step ID and a description of the visitor step.
class TestVisitorUtil
    : public mlir::PassWrapper<TestVisitorUtil, mlir::FunctionPass> {
 public:
  void runOnFunction() override {
    mlir::FuncOp func = getOperation();
    int step_id = 0;
    GenericWalk(func, [&](mlir::Operation *op, const WalkStage &stage) {
      op->emitRemark() << step_id++ << ": " << get_stage_description(stage);
    });

    // Exercise static inference of operation type
    GenericWalk(func, [&](mlir::TF::IfRegionOp op, const WalkStage &stage) {
      op.emitRemark() << step_id++ << ": " << get_stage_description(stage);
    });
  }
};

class TestVisitorUtilInterrupt
    : public mlir::PassWrapper<TestVisitorUtilInterrupt, mlir::FunctionPass> {
 public:
  void runOnFunction() override {
    mlir::FuncOp func = getOperation();
    int step_id = 0;

    auto walker = [&](mlir::Operation *op, const WalkStage &stage) {
      if (auto interrupt_before_all =
              op->getAttrOfType<mlir::BoolAttr>("interrupt_before_all"))
        if (interrupt_before_all.getValue() && stage.IsBeforeAllRegions())
          return mlir::WalkResult::interrupt();

      if (auto interrupt_after_all =
              op->getAttrOfType<mlir::BoolAttr>("interrupt_after_all"))
        if (interrupt_after_all.getValue() && stage.IsAfterAllRegions())
          return mlir::WalkResult::interrupt();

      if (auto interrupt_after_region =
              op->getAttrOfType<mlir::IntegerAttr>("interrupt_after_region"))
        if (stage.IsAfterRegion(
                static_cast<int>(interrupt_after_region.getInt())))
          return mlir::WalkResult::interrupt();

      op->emitRemark() << step_id++ << ": " << get_stage_description(stage);
      return mlir::WalkResult::advance();
    };

    // Interrupt the walk based on attributes on the operation.
    auto result = GenericWalk(func, walker);

    if (result.wasInterrupted())
      func.emitRemark() << step_id++ << ": walk was interrupted";

    // Exercise static inference of operation type for interrupting callback.
    result =
        GenericWalk(func, [&](mlir::TF::IfRegionOp op, const WalkStage &stage) {
          return walker(op, stage);
        });

    if (result.wasInterrupted())
      func.emitRemark() << step_id++ << ": walk was interrupted";
  }
};

mlir::PassRegistration<TestVisitorUtil> pass(
    "tf-test-visitor-util",
    "Add remarks that trace order of visiting operations using TF visitor "
    "utilities.");

mlir::PassRegistration<TestVisitorUtilInterrupt> pass_interrupt(
    "tf-test-visitor-util-interrupt",
    "Add remarks that trace order of visiting operations using TF visitor "
    "utilities, interrupt version.");

}  // anonymous namespace
}  // namespace tensorflow
