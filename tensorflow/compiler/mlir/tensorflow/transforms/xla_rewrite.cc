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

// This transformation pass converts stateful and stateless partitioned calls
// with _xla_compile_device_type attribute to XLA launch ops.

#include <memory>

#include "mlir/Support/LLVM.h"  // from @llvm-project
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_device.h"
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_ops.h"
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_types.h"
#include "tensorflow/compiler/mlir/tensorflow/transforms/passes.h"
#include "tensorflow/compiler/mlir/tensorflow/utils/attribute_utils.h"
#include "tensorflow/compiler/mlir/tensorflow/utils/call_graph_util.h"

#define DEBUG_TYPE "tf-xla-rewrite"

namespace mlir {
namespace {

#define GEN_PASS_DEF_XLAREWRITEPASS
#include "tensorflow/compiler/mlir/tensorflow/transforms/tf_device_passes.h.inc"

struct XlaRewritePass : public impl::XlaRewritePassBase<XlaRewritePass> {
  void runOnOperation() override;
};

void MoveResourceArgsToEnd(func::FuncOp callee) {
  llvm::DenseMap<BlockArgument, BlockArgument> mapping;
  unsigned num_params = callee.getNumArguments();
  llvm::BitVector removed_params(num_params);
  // Copy the resource-type parameters to the end.
  for (unsigned i = 0; i < num_params; ++i) {
    BlockArgument param = callee.getArgument(i);
    if (mlir::isa<TF::ResourceType>(getElementTypeOrSelf(param.getType()))) {
      removed_params.set(i);
      callee.getBody().addArgument(param.getType(), param.getLoc());
      param.replaceAllUsesWith(callee.getArguments().back());
      removed_params.push_back(false);
    }
  }
  // Remove old resource-type parameters.
  callee.getBody().front().eraseArguments(removed_params);
  // Update function type.
  callee.setFunctionType(FunctionType::get(callee.getContext(),
                                           callee.getBody().getArgumentTypes(),
                                           callee.getResultTypes()));
}

void RewriteCall(tf_device::ClusterFuncOp cluster_func_op, SymbolTable &symtab,
                 OpBuilder &builder) {
  llvm::SmallVector<Value> non_resource_args, resource_args;
  bool has_resources = false, in_order = true;
  for (const Value &arg : cluster_func_op.getOperands()) {
    if (!mlir::isa<TF::ResourceType>(getElementTypeOrSelf(arg.getType()))) {
      non_resource_args.push_back(arg);
      if (has_resources) in_order = false;
    } else {
      resource_args.push_back(arg);
      has_resources = true;
    }
  }

  if (!in_order) {
    // Functions do not get reused in practice, so skip the check for if the
    // callee has been updated.
    StringAttr callee_sym = cluster_func_op.getFuncAttr().getAttr();
    MoveResourceArgsToEnd(symtab.lookup<func::FuncOp>(callee_sym));
  }
  builder.setInsertionPoint(cluster_func_op);
  auto xla_launch_op = builder.create<TF::XlaLaunchOp>(
      cluster_func_op.getLoc(), cluster_func_op.getResultTypes(),
      /*constants=*/ValueRange({}), ValueRange(non_resource_args),
      ValueRange(resource_args), cluster_func_op.getFuncAttr());

  CopyDeviceAndUnderscoredAttributes(cluster_func_op, xla_launch_op);
  cluster_func_op.replaceAllUsesWith(xla_launch_op.getResults());
  cluster_func_op.erase();
}

void XlaRewritePass::runOnOperation() {
  ModuleOp module = getOperation();
  SymbolTable symtab(module);
  OpBuilder builder(&getContext());
  module.walk([&](tf_device::ClusterFuncOp cluster_func_op) {
    RewriteCall(cluster_func_op, symtab, builder);
  });
}

}  // namespace

namespace TFDevice {
std::unique_ptr<OperationPass<ModuleOp>> CreateXlaRewritePass() {
  return std::make_unique<XlaRewritePass>();
}

}  // namespace TFDevice
}  // namespace mlir
