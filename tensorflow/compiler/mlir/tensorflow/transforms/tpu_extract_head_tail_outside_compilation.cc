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

#include <memory>

#include "mlir/Pass/Pass.h"  // from @llvm-project
#include "mlir/Pass/PassRegistry.h"  // from @llvm-project
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_device.h"
#include "tensorflow/compiler/mlir/tensorflow/transforms/passes.h"

namespace mlir {
namespace TFTPU {

// This pass extracts a CPU computation cluster with `_xla_outside_compilation`
// annotation from the head or tail of a TPU cluster.

namespace {

constexpr char kXlaOutsideCompilationAttr[] = "_xla_outside_compilation";

struct TPUExtractHeadTailOutsideCompilation
    : public PassWrapper<TPUExtractHeadTailOutsideCompilation, FunctionPass> {
  void runOnFunction() override;
};

void TPUExtractHeadTailOutsideCompilation::runOnFunction() {
  getFunction().walk([&](tf_device::LaunchOp launch) {
    Block& launch_block = launch.GetBody();
    for (auto& op : llvm::make_early_inc_range(launch_block.getOperations())) {
      // TODO(b/155115766): Handle outputs that should be inputs to TPU
      // LaunchOp.
      if (auto attr =
              op.getAttrOfType<StringAttr>(kXlaOutsideCompilationAttr)) {
        op.moveBefore(launch);
      } else {
        break;
      }
    }
  });
}

}  // anonymous namespace

std::unique_ptr<OperationPass<FuncOp>>
CreateTPUExtractHeadTailOutsideCompilationPass() {
  return std::make_unique<TPUExtractHeadTailOutsideCompilation>();
}

static PassRegistration<TPUExtractHeadTailOutsideCompilation> pass(
    "tf-tpu-extract-head-tail-outside-compilation",
    "Extracts TPU head or tail outside compilation to separate "
    "parallel_execute.");

}  // namespace TFTPU
}  // namespace mlir
