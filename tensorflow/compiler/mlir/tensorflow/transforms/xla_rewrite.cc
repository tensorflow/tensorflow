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

// This transformation pass converts outermost stateful and stateless
// partitioned calls with _xla_compile_device_type attribute to XLA launch ops.

#include <memory>
#include <stack>
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

inline constexpr absl::string_view kEntryFunction = "main";

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
void RewriteCall(OpT pcall_op, SymbolTable &symtab) {
  llvm::SmallVector<Value> non_resource_args, resource_args;
  bool has_resources = false, in_order = true;
  for (const Value &arg : pcall_op.getArgs()) {
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
        cast<SymbolRefAttr>(pcall_op.getFAttr()).getRootReference();
    MoveResourceArgsToEnd(cast<func::FuncOp>(symtab.lookup(callee_sym)));
  }
  OpBuilder builder(pcall_op->getContext());
  builder.setInsertionPoint(pcall_op);
  auto xla_launch_op = builder.create<TF::XlaLaunchOp>(
      pcall_op.getLoc(), pcall_op.getResultTypes(),
      /*constants=*/ValueRange({}), ValueRange(non_resource_args),
      ValueRange(resource_args), pcall_op.getFAttr());

  CopyDeviceAndUnderscoredAttributes(pcall_op, xla_launch_op);
  pcall_op.replaceAllUsesWith(xla_launch_op.getResults());
  pcall_op.erase();
}

// Rewrite outermost tf.StatefulPartitionedCallOp or tf.PartitionedCallOp with
// _xla_compile_device_type attribute.
LogicalResult RewriteOutermostCallOps(func::FuncOp func, SymbolTable &symtab) {
  std::stack<SymbolUserOpInterface> worklist;
  func->walk([&](SymbolUserOpInterface op) { worklist.push(op); });
  while (!worklist.empty()) {
    auto op = worklist.top();
    worklist.pop();
    if (auto stateful_pcall_op =
            llvm::dyn_cast<TF::StatefulPartitionedCallOp>(op.getOperation())) {
      if (op->hasAttr(tensorflow::kCompileDeviceTypeAttr)) {
        RewriteCall(stateful_pcall_op, symtab);
        continue;
      }
    }

    if (auto pcall_op =
            llvm::dyn_cast<TF::PartitionedCallOp>(op.getOperation())) {
      if (op->hasAttr(tensorflow::kCompileDeviceTypeAttr)) {
        RewriteCall(pcall_op, symtab);
        continue;
      }
    }

    for (auto attr : op->getAttrs()) {
      auto sym = attr.getValue().dyn_cast<SymbolRefAttr>();
      if (!sym) continue;
      auto func = symtab.lookup<func::FuncOp>(sym.getRootReference());
      if (!func) {
        func.emitError() << "Cannot find function " << sym.getRootReference();
        return failure();
      }
      func->walk([&](SymbolUserOpInterface op) { worklist.push(op); });
    }
  }
  return success();
}

void XlaRewritePass::runOnOperation() {
  ModuleOp module = getOperation();
  SymbolTable symtab(module);
  func::FuncOp entry_func = symtab.lookup<func::FuncOp>(kEntryFunction);
  if (!entry_func) {
    // This is not expected to happen in practice.
    module.emitError() << "entry function " << kEntryFunction
                       << " must be present";
    return signalPassFailure();
  }
  if (failed(RewriteOutermostCallOps(entry_func, symtab)))
    return signalPassFailure();
}
}  // namespace

namespace TFDevice {
std::unique_ptr<OperationPass<ModuleOp>> CreateXlaRewritePass() {
  return std::make_unique<XlaRewritePass>();
}

}  // namespace TFDevice
}  // namespace mlir
