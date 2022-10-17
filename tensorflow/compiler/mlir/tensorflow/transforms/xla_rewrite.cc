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

// This transformation pass converts stateful partitioned calls with
// _xla_compile_device_type attribute to XLA launch ops.

#include <memory>
#include <string>
#include <vector>

#include "tensorflow/compiler/mlir/tensorflow/ir/tf_device.h"
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_ops.h"
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_types.h"
#include "tensorflow/compiler/mlir/tensorflow/transforms/passes.h"
#include "tensorflow/compiler/mlir/tensorflow/utils/attribute_utils.h"

#define DEBUG_TYPE "tf-xla-rewrite"

namespace mlir {
namespace {

#define GEN_PASS_DEF_XLAREWRITEPASS
#include "tensorflow/compiler/mlir/tensorflow/transforms/tf_device_passes.h.inc"

struct XlaRewritePass : public impl::XlaRewritePassBase<XlaRewritePass> {
  void runOnOperation() override;
};

template <typename OpT,
          typename std::enable_if<llvm::is_one_of<
              OpT, TF::PartitionedCallOp,
              TF::StatefulPartitionedCallOp>::value>::type * = nullptr>
void Rewrite(OpT pcall_op) {
  OpBuilder builder(pcall_op->getContext());
  builder.setInsertionPoint(pcall_op);

  llvm::SmallVector<Value> args, resources;
  for (auto arg : pcall_op.args()) {
    if (getElementTypeOrSelf(arg.getType()).template isa<TF::ResourceType>()) {
      resources.push_back(arg);
    } else {
      args.push_back(arg);
    }
  }

  auto xla_launch_op = builder.create<TF::XlaLaunchOp>(
      pcall_op.getLoc(), pcall_op.getResultTypes(),
      /*constants=*/ValueRange({}), ValueRange(args), ValueRange(resources),
      pcall_op.fAttr());
  CopyDeviceAndUnderscoredAttributes(pcall_op, xla_launch_op);
  pcall_op.replaceAllUsesWith(xla_launch_op.getResults());
  pcall_op.erase();
}

void XlaRewritePass::runOnOperation() {
  func::FuncOp func_op = getOperation();

  func_op.walk([&](mlir::Operation *op) {
    if (!op->hasAttr(tensorflow::kCompileDeviceTypeAttr))
      return WalkResult::advance();
    if (auto pcall_op = dyn_cast<TF::PartitionedCallOp>(op)) {
      Rewrite(pcall_op);
    } else if (auto stateful_pcall_op =
                   dyn_cast<TF::StatefulPartitionedCallOp>(op)) {
      Rewrite(stateful_pcall_op);
    }
    return WalkResult::advance();
  });
}
}  // namespace

namespace TFDevice {
std::unique_ptr<OperationPass<func::FuncOp>> CreateXlaRewritePass() {
  return std::make_unique<XlaRewritePass>();
}

}  // namespace TFDevice
}  // namespace mlir
