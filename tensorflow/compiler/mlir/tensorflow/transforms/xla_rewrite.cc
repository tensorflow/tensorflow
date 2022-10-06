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

void Rewrite(TF::StatefulPartitionedCallOp call_op) {
  OpBuilder builder(call_op->getContext());
  builder.setInsertionPoint(call_op);

  llvm::SmallVector<Value> args, resources;
  for (auto arg : call_op.args()) {
    if (getElementTypeOrSelf(arg.getType()).isa<TF::ResourceType>()) {
      resources.push_back(arg);
    } else {
      args.push_back(arg);
    }
  }

  auto xla_launch_op = builder.create<TF::XlaLaunchOp>(
      call_op.getLoc(), call_op.getResultTypes(), /*constants=*/ValueRange({}),
      ValueRange(args), ValueRange(resources), call_op.fAttr());
  CopyDeviceAndUnderscoredAttributes(call_op, xla_launch_op);
  call_op.replaceAllUsesWith(xla_launch_op.getResults());
  call_op.erase();
}

void XlaRewritePass::runOnOperation() {
  func::FuncOp func_op = getOperation();

  llvm::SmallVector<TF::StatefulPartitionedCallOp, 4> ops;
  func_op.walk([&](TF::StatefulPartitionedCallOp call_op) {
    if (call_op->hasAttr(tensorflow::kCompileDeviceTypeAttr)) {
      ops.push_back(call_op);
    }
  });

  // Rewrite tf.StatefulPartitionedCall ops to tf.XlaLaunch ops.
  for (auto call_op : ops) Rewrite(call_op);
}

}  // namespace

namespace TFDevice {
std::unique_ptr<OperationPass<func::FuncOp>> CreateXlaRewritePass() {
  return std::make_unique<XlaRewritePass>();
}

}  // namespace TFDevice
}  // namespace mlir
