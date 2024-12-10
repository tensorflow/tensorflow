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

#include <memory>
#include <utility>

#include "llvm/ADT/STLExtras.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Value.h"
#include "mlir/IR/Visitors.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "xla/python/ifrt/ir/ifrt_dialect.h"
#include "xla/python/ifrt/ir/ifrt_ops.h"
#include "xla/python/ifrt/ir/transforms/passes.h"
#include "xla/python/ifrt/ir/transforms/utils.h"

namespace xla {
namespace ifrt {

namespace {

#define GEN_PASS_DEF_IFRTRESHARDTOCOPYARRAYSPASS
#include "xla/python/ifrt/ir/transforms/passes.h.inc"

class ReshardToCopyArraysOpPattern
    : public mlir::OpRewritePattern<xla::ifrt::ReshardOp> {
 public:
  using mlir::OpRewritePattern<xla::ifrt::ReshardOp>::OpRewritePattern;

  mlir::LogicalResult matchAndRewrite(
      xla::ifrt::ReshardOp op, mlir::PatternRewriter& rewriter) const override {
    // Map from devices attribute to indices of the input arrays that are just
    // copied to those devices.
    llvm::DenseMap<xla::ifrt::IfrtDevicesAttr, llvm::SmallVector<int>>
        copy_indices;
    // Indices of the input arrays that are resharded.
    llvm::SmallVector<int> reshard_indices;
    for (const auto& [idx, pair] :
         llvm::enumerate(llvm::zip(op.getInputs(), op.getOutputs()))) {
      auto in_array_type =
          mlir::cast<xla::ifrt::IfrtArrayType>(std::get<0>(pair).getType());
      if (in_array_type == nullptr) {
        op.emitOpError() << "requires all inputs to be `IfrtArrayType`. Input #"
                         << idx << ": " << std::get<0>(pair).getType();
        return mlir::failure();
      }
      auto out_array_type =
          mlir::cast<xla::ifrt::IfrtArrayType>(std::get<1>(pair).getType());
      if (out_array_type == nullptr) {
        op.emitOpError()
            << "requires all outputs to be `IfrtArrayType`. Output #" << idx
            << ": " << std::get<1>(pair).getType();
        return mlir::failure();
      }
      if (!IsReshard(in_array_type, out_array_type)) {
        copy_indices[out_array_type.getDevicesAttr()].push_back(idx);
      } else {
        reshard_indices.push_back(idx);
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
    llvm::SmallVector<mlir::Value> reshard_input_values;
    llvm::SmallVector<mlir::Type> reshard_output_types;
    for (int idx : reshard_indices) {
      outputs[idx] = op.getOutputs()[idx];
      reshard_input_values.push_back(op.getInputs()[idx]);
      reshard_output_types.push_back(op.getOutputs()[idx].getType());
    }
    if (!reshard_input_values.empty()) {
      auto reshard_op = rewriter.create<xla::ifrt::ReshardOp>(
          op.getLoc(),
          /*outputs=*/reshard_output_types,
          /*control_output=*/op.getControlOutput().getType(),
          /*inputs=*/reshard_input_values,
          /*donated=*/op.getDonated(),
          /*control_inputs=*/op.getControlInputs());
      for (const auto& [idx, output] :
           llvm::zip(reshard_indices, reshard_op.getOutputs())) {
        outputs[idx] = output;
      }
      control_output = reshard_op.getControlOutput();
    }

    // Add an ifrt.CopyArrays op for each set of arrays that are copied to a
    // set of devices. The new ifrt.CopyArrays ops will inherit *all* the input
    // control dependencies of the ifrt.Reshard op. They could receive a subset
    // of the control dependencies (e.g., dependencies generated by ops running
    // use the same devices as the ones the arrays are coppied to), but that is
    // not supported yet.
    for (const auto& [devices_attr, indices] : copy_indices) {
      llvm::SmallVector<mlir::Value> copy_input_values;
      llvm::SmallVector<mlir::Type> copy_output_types;
      for (int idx : indices) {
        copy_input_values.push_back(op.getInputs()[idx]);
        copy_output_types.push_back(op.getOutputs()[idx].getType());
      }
      auto copy_arrays_op = rewriter.create<xla::ifrt::CopyArraysOp>(
          op.getLoc(),
          /*outputs=*/copy_output_types,
          /*control_output=*/op.getControlOutput().getType(),
          /*inputs=*/copy_input_values,
          /*donated=*/op.getDonated(),
          /*control_inputs=*/op.getControlInputs());
      for (const auto& [idx, output] :
           llvm::zip(indices, copy_arrays_op.getOutputs())) {
        outputs[idx] = output;
      }
      if (reshard_input_values.empty()) {
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
    if (mlir::failed(mlir::applyPatternsAndFoldGreedily(module_op,
                                                        std::move(patterns)))) {
      signalPassFailure();
    }
  }
};

}  // namespace

std::unique_ptr<mlir::OperationPass<mlir::ModuleOp>>
CreateIfrtReshardToCopyArraysPass() {
  return std::make_unique<IfrtReshardToCopyArraysPass>();
}

}  // namespace ifrt
}  // namespace xla
