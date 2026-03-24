/* Copyright 2024 The OpenXLA Authors.

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

#include <utility>

#include "llvm/ADT/STLExtras.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Value.h"
#include "mlir/IR/Visitors.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "xla/python/ifrt/ir/ifrt_dialect.h"
#include "xla/python/ifrt/ir/ifrt_ops.h"
#include "xla/python/ifrt/ir/transforms/passes.h"
#include "xla/python/ifrt/ir/transforms/utils.h"

namespace xla {
namespace ifrt {

#define GEN_PASS_DEF_IFRTRESHARDTOCOPYARRAYSPASS
#include "xla/python/ifrt/ir/transforms/passes.h.inc"

namespace {

class ReshardToCopyArraysOpPattern : public mlir::OpRewritePattern<ReshardOp> {
 public:
  using mlir::OpRewritePattern<ReshardOp>::OpRewritePattern;

  mlir::LogicalResult matchAndRewrite(
      ReshardOp op, mlir::PatternRewriter& rewriter) const override {
    // Indices of the input arrays that are just copied.
    llvm::SmallVector<int> copy_indices;
    // Indices of the input arrays that are resharded.
    llvm::SmallVector<int> reshard_indices;
    llvm::SmallVector<mlir::Value> reshard_inputs_left;
    llvm::SmallVector<mlir::Type> reshard_outputs_left;
    for (const auto& [idx, pair] :
         llvm::enumerate(llvm::zip(op.getInputs(), op.getOutputs()))) {
      auto in_array_type =
          mlir::cast<IfrtArrayType>(std::get<0>(pair).getType());
      if (in_array_type == nullptr) {
        op.emitOpError() << "requires all inputs to be `IfrtArrayType`. Input #"
                         << idx << ": " << std::get<0>(pair).getType();
        return mlir::failure();
      }
      auto out_array_type =
          mlir::cast<IfrtArrayType>(std::get<1>(pair).getType());
      if (out_array_type == nullptr) {
        op.emitOpError()
            << "requires all outputs to be `IfrtArrayType`. Output #" << idx
            << ": " << std::get<1>(pair).getType();
        return mlir::failure();
      }
      if (IsReshard(in_array_type, out_array_type)) {
        reshard_indices.push_back(idx);
        reshard_inputs_left.push_back(op.getInputs()[idx]);
        reshard_outputs_left.push_back(op.getOutputs()[idx].getType());
      } else {
        copy_indices.push_back(idx);
      }
    }

    if (reshard_indices.size() == op.getInputs().size()) {
      // All arrays are resharded. No need to modify the ifrt.Reshard op.
      return mlir::failure();
    }

    if (!op.getControlOutput().getUses().empty()) {
      // If the control output dependency of the ifrt.Reshard op is used then it
      // is unclear what to do with the newly added ifrt.CopyArrays ops. The
      // conservative approach would be to add these as control dependencies to
      // all the ops that have a control dependency on the ifrt.Reshard op.
      // However, we could also add them just to the ops that have a control
      // dependency on the ifrt.Reshard op and use the same devices. For now,
      // we will just throw an error as the ifrt.Reshard control dependencies
      // are not used at the moment.
      op.emitOpError() << " cannot extract `ifrt.CopyArrays` from "
                          "`ifrt.Reshard` with control dependency output";
      return mlir::failure();
    }

    llvm::SmallVector<mlir::Value> outputs;
    outputs.resize(op.getOutputs().size());
    // If an ifrt.Reshard is still left, then we replace the usage of the
    // current ifrt.Reshard op's control output with its control output.
    // Otherwise, we replace it with the control output of the last
    // ifrt.CopyArrays op.
    mlir::Value control_output;

    // Replace the ifrt.Reshard with a pruned version that only takes the arrays
    // that are resharded.
    if (!reshard_inputs_left.empty()) {
      ReshardOp reshard_op =
          ReshardOp::create(rewriter, op.getLoc(),
                            /*outputs=*/reshard_outputs_left,
                            /*control_output=*/op.getControlOutput().getType(),
                            /*inputs=*/reshard_inputs_left,
                            /*donated=*/op.getDonated(),
                            /*control_inputs=*/op.getControlInputs());
      for (const auto& [idx, output] :
           llvm::zip(reshard_indices, reshard_op.getOutputs())) {
        outputs[idx] = output;
      }
      control_output = reshard_op.getControlOutput();
    }

    // Add a CopyArrays op for each copied array. All new CopyArrays inherit
    // *all* the input control dependencies of the Reshard op. They could
    // receive a subset of the control dependencies (e.g., dependencies
    // generated by ops running use the same devices as the ones the arrays are
    // coppied to), but that is not supported yet.
    for (auto array_idx : copy_indices) {
      CopyArraysOp copy_arrays_op = CopyArraysOp::create(
          rewriter, op.getLoc(),
          /*outputs=*/{op.getOutputs()[array_idx].getType()},
          /*control_output=*/op.getControlOutput().getType(),
          /*inputs=*/{op.getInputs()[array_idx]},
          /*donated=*/op.getDonated(),
          /*control_inputs=*/op.getControlInputs());
      outputs[array_idx] = copy_arrays_op.getOutputs().front();
      if (reshard_inputs_left.empty()) {
        // No reshard op left, so the control output is replaced with the
        // control output of the last inserted CopyArrays op.
        control_output = copy_arrays_op.getControlOutput();
      }
    }
    outputs.push_back(control_output);
    rewriter.replaceOp(op, outputs);
    return mlir::success();
  }
};

class IfrtReshardToCopyArraysPass
    : public impl::IfrtReshardToCopyArraysPassBase<
          IfrtReshardToCopyArraysPass> {
 public:
  void runOnOperation() override {
    mlir::RewritePatternSet patterns(&getContext());
    patterns.add<ReshardToCopyArraysOpPattern>(&getContext());
    mlir::ModuleOp module_op = getOperation();
    if (mlir::failed(
            mlir::applyPatternsGreedily(module_op, std::move(patterns)))) {
      signalPassFailure();
    }
  }
};

}  // namespace
}  // namespace ifrt
}  // namespace xla
