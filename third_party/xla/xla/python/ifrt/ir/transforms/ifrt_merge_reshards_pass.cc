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
#include <tuple>
#include <vector>

#include "absl/log/check.h"
#include "llvm/ADT/STLExtras.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Types.h"
#include "mlir/IR/Value.h"
#include "mlir/IR/ValueRange.h"
#include "mlir/IR/Visitors.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/LLVM.h"
#include "xla/python/ifrt/ir/constants.h"
#include "xla/python/ifrt/ir/ifrt_dialect.h"
#include "xla/python/ifrt/ir/ifrt_ops.h"
#include "xla/python/ifrt/ir/transforms/passes.h"

namespace xla {
namespace ifrt {

namespace {

#define GEN_PASS_DEF_IFRTMERGERESHARDSPASS
#include "xla/python/ifrt/ir/transforms/passes.h.inc"

class IfrtMergeReshardsPass
    : public impl::IfrtMergeReshardsPassBase<IfrtMergeReshardsPass> {
 public:
  void runOnOperation() override;
};

// Merges reshards on `source_values` which flow into the same
// destination. We merge only if the reshard:
// - has only one input and output. I.e. it isn't already merged.
// - has no input control dependencies.
// - has the same `donation` setting.
//
// `source_values` are expected to be of type IfrtArrayType on the same devices,
// and be OpResults from the same ops, or BlockArgs in the same block.
//
// We defer erasing the op until the end of the pass, to avoid invalidating the
// iterator.
void MergeReshardsIgnoringControlDependencies(
    mlir::ValueRange source_values, std::vector<mlir::Operation*>& ops_to_erase,
    mlir::RewriterBase& rewriter) {
  // We group reshards by {first_user, devices, donated, src_memory_kind,
  // dst_memory_kind}. We need to group by:
  // - first_user because we need to pick some destination.
  // - devices as well, because certain users can have multiple devices, e.g.
  // func.return.
  // - donated because the donation is all-or-nothing.
  // - src and dst memory kind because we can't merge reshards that change
  // memory kind.
  llvm::DenseMap<std::tuple<mlir::Operation*, IfrtDevicesAttr, mlir::Attribute,
                            mlir::StringAttr, mlir::StringAttr>,
                 llvm::SmallVector<ReshardOp>>
      user_device_donate_tuple_to_reshards;

  // Group reshards by their first user.
  for (mlir::Value value : source_values) {
    CHECK(mlir::isa<IfrtArrayType>(value.getType()));

    for (mlir::Operation* user : value.getUsers()) {
      auto reshard_op = mlir::dyn_cast<ReshardOp>(user);
      if (!reshard_op || reshard_op.getOutputs().size() != 1 ||
          reshard_op->use_empty() || !reshard_op.getControlInputs().empty()) {
        continue;
      }

      // This could potentially be very expensive as `isBeforeInBlock` is
      // average O(1) but worst case O(n).
      mlir::Operation* first_reshard_user = *llvm::min_element(
          reshard_op->getUsers(), [](mlir::Operation* a, mlir::Operation* b) {
            return a->isBeforeInBlock(b);
          });

      auto output_type =
          mlir::cast<IfrtArrayType>(reshard_op.getOutputs().front().getType());
      user_device_donate_tuple_to_reshards
          [{first_reshard_user, output_type.getDevicesAttr(),
            // We can't hash by the bool itself, and `donated` is a optional
            // attr, so false can be represented by nullptr or BoolAttr(false).
            // So we explicitly convert to BoolAttr.
            rewriter.getBoolAttr(reshard_op.getDonated()),
            mlir::cast<IfrtArrayType>(reshard_op.getInputs().front().getType())
                .getMemoryKindAttr(),
            output_type.getMemoryKindAttr()}]
              .push_back(reshard_op);
    }
  }

  // Rewrite each group of reshards.
  for (auto& [_, reshards] : user_device_donate_tuple_to_reshards) {
    // Create a new reshard op that takes all the inputs of the reshards.
    llvm::SmallVector<mlir::Value> inputs;
    llvm::SmallVector<mlir::Type> output_types;
    llvm::SmallVector<mlir::Location> locs;
    inputs.reserve(reshards.size());
    output_types.reserve(reshards.size());
    locs.reserve(reshards.size());

    for (ReshardOp reshard : reshards) {
      CHECK_EQ(reshard.getInputs().size(), 1);
      CHECK_EQ(reshard.getOutputs().size(), 1);
      inputs.push_back(reshard.getInputs()[0]);
      output_types.push_back(reshard.getOutputs()[0].getType());
      locs.push_back(reshard.getLoc());
    }

    // Insert the new reshard op just before one of the reshards, to
    // minimize reordering reshards.
    rewriter.setInsertionPoint(reshards.front());
    auto merged_reshard =
        rewriter.create<ReshardOp>(rewriter.getFusedLoc(locs),
                                   /*outputs=*/output_types,
                                   /*control_output=*/
                                   IfrtControlType::get(rewriter.getContext()),
                                   /*inputs=*/inputs,
                                   /*donated=*/reshards.front().getDonated(),
                                   /*control_inputs=*/mlir::ValueRange());

    // Replace the original reshards with the new merged reshard.
    for (auto [index, reshard] : llvm::enumerate(reshards)) {
      rewriter.replaceAllUsesWith(reshard.getOutputs()[0],
                                  merged_reshard.getOutputs()[index]);
      rewriter.replaceAllUsesWith(reshard.getControlOutput(),
                                  merged_reshard.getControlOutput());
      ops_to_erase.push_back(reshard);
    }
  }
}

template <typename T>
bool MergeReshardsIgnoringControlDependencies(
    mlir::Operation* op, std::vector<mlir::Operation*>& ops_to_erase,
    mlir::RewriterBase& rewriter) {
  if (auto casted = mlir::dyn_cast<T>(op)) {
    MergeReshardsIgnoringControlDependencies(casted.getOutputs(), ops_to_erase,
                                             rewriter);
    return true;
  }
  return false;
}

template <typename First, typename Second, typename... Rest>
bool MergeReshardsIgnoringControlDependencies(
    mlir::Operation* op, std::vector<mlir::Operation*>& ops_to_erase,
    mlir::RewriterBase& rewriter) {
  return MergeReshardsIgnoringControlDependencies<First>(op, ops_to_erase,
                                                         rewriter) ||
         MergeReshardsIgnoringControlDependencies<Second, Rest...>(
             op, ops_to_erase, rewriter);
}

void IfrtMergeReshardsPass::runOnOperation() {
  mlir::func::FuncOp func_op = getOperation();
  mlir::IRRewriter rewriter(func_op->getContext());
  std::vector<mlir::Operation*> ops_to_erase;

  // We only need to run this pass on IFRT functions.
  if (!func_op->hasAttr(kIfrtFunctionAttrName) &&
      !func_op->hasAttr(kIfrtReshardFunctionAttrName)) {
    return;
  }

  // Handle func block args.
  {
    llvm::DenseMap<IfrtDevicesAttr, llvm::SmallVector<mlir::Value>>
        devices_to_args;
    for (mlir::Value arg : func_op.getArguments()) {
      if (auto array_type = mlir::dyn_cast<IfrtArrayType>(arg.getType())) {
        devices_to_args[array_type.getDevicesAttr()].push_back(arg);
      }
    }

    for (auto& [_, args] : devices_to_args) {
      MergeReshardsIgnoringControlDependencies(args, ops_to_erase, rewriter);
    }
  }

  // Handle ops in the IFRT function body.
  func_op.getFunctionBody().walk([&](mlir::Operation* op) {
    MergeReshardsIgnoringControlDependencies<
        CallOp, CallLoadedExecutableOp, ReshardOp, CopyArraysOp, RemapArraysOp>(
        op, ops_to_erase, rewriter);
  });

  for (mlir::Operation* op : ops_to_erase) {
    rewriter.eraseOp(op);
  }
}

}  // namespace

std::unique_ptr<mlir::OperationPass<mlir::func::FuncOp>>
CreateIfrtMergeReshardsPass() {
  return std::make_unique<IfrtMergeReshardsPass>();
}

}  // namespace ifrt
}  // namespace xla
