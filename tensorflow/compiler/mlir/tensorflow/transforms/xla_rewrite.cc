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

// This transformation pass converts stateful and stateless paritioned calls
// with _xla_compile_device_type attribute to XLA launch ops.

#include <stack>

#include "tensorflow/compiler/mlir/tensorflow/ir/tf_device.h"
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_ops.h"
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_types.h"
#include "tensorflow/compiler/mlir/tensorflow/transforms/call_graph_util.h"
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

void MoveResourceArgsToEnd(func::FuncOp callee) {
  llvm::DenseMap<BlockArgument, BlockArgument> mapping;
  unsigned num_params = callee.getNumArguments();
  llvm::BitVector removed_params(num_params);
  // Copy the resource-type parameters to the end.
  for (unsigned i = 0; i < num_params; ++i) {
    BlockArgument param = callee.getArgument(i);
    if (getElementTypeOrSelf(param.getType())
            .template isa<TF::ResourceType>()) {
      removed_params.set(i);
      callee.getBody().addArgument(param.getType(), param.getLoc());
      param.replaceAllUsesWith(callee.getArguments().back());
      removed_params.push_back(false);
    }
  }
  // Remove old reousrce-type parameters.
  callee.getBody().front().eraseArguments(removed_params);
  // Update function type.
  callee.setFunctionType(FunctionType::get(callee.getContext(),
                                           callee.getBody().getArgumentTypes(),
                                           callee.getResultTypes()));
}

template <typename OpT,
          typename std::enable_if<llvm::is_one_of<
              OpT, TF::PartitionedCallOp,
              TF::StatefulPartitionedCallOp>::value>::type * = nullptr>
void RewriteCall(OpT call_op, SymbolTable &symtab) {
  llvm::SmallVector<Value> non_resource_args, resource_args;
  bool has_resources = false, in_order = true;
  for (const Value &arg : call_op.getArgs()) {
    if (!getElementTypeOrSelf(arg.getType()).template isa<TF::ResourceType>()) {
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
    StringAttr callee_sym =
        cast<SymbolRefAttr>(call_op.getFAttr()).getRootReference();
    MoveResourceArgsToEnd(symtab.lookup<func::FuncOp>(callee_sym));
  }
  OpBuilder builder(call_op->getContext());
  builder.setInsertionPoint(call_op);
  auto xla_launch_op = builder.create<TF::XlaLaunchOp>(
      call_op.getLoc(), call_op.getResultTypes(),
      /*constants=*/ValueRange({}), ValueRange(non_resource_args),
      ValueRange(resource_args), call_op.getFAttr());

  CopyDeviceAndUnderscoredAttributes(call_op, xla_launch_op);
  call_op.replaceAllUsesWith(xla_launch_op.getResults());
  call_op.erase();
}

void XlaRewritePass::runOnOperation() {
  ModuleOp module = getOperation();
  SymbolTable symtab(module);
  module.walk([&](tf_device::ClusterOp cluster_op) {
    cluster_op.getBody().walk([&](mlir::Operation *op) {
      if (auto call_op = llvm::dyn_cast<TF::StatefulPartitionedCallOp>(op)) {
        RewriteCall(call_op, symtab);
      } else if (auto call_op = llvm::dyn_cast<TF::PartitionedCallOp>(op)) {
        RewriteCall(call_op, symtab);
      }
    });
  });

  // Verify that there are no nested XLA launch ops.
  module.walk([&](TF::XlaLaunchOp xla_launch_op) {
    llvm::SmallVector<mlir::Operation *> nested_launch_ops;
    func::FuncOp root = symtab.lookup<func::FuncOp>(
        xla_launch_op.getFunctionAttr().getRootReference());
    if (failed(GetOutermostOpsOfType<TF::XlaLaunchOp>(root, symtab,
                                                      nested_launch_ops)))
      return signalPassFailure();
    if (!nested_launch_ops.empty()) {
      xla_launch_op.emitError() << "Nested XLA launch ops detected";
      return signalPassFailure();
    }
  });
}

}  // namespace

namespace TFDevice {
std::unique_ptr<OperationPass<ModuleOp>> CreateXlaRewritePass() {
  return std::make_unique<XlaRewritePass>();
}

}  // namespace TFDevice
}  // namespace mlir
