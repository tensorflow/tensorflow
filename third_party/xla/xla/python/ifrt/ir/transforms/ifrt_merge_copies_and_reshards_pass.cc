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

#include <queue>

#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/strings/str_cat.h"
#include "llvm/ADT/Hashing.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/Casting.h"
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
#include "mlir/Support/LLVM.h"
#include "xla/python/ifrt/ir/constants.h"
#include "xla/python/ifrt/ir/ifrt_dialect.h"
#include "xla/python/ifrt/ir/ifrt_ops.h"
#include "xla/python/ifrt/ir/transforms/passes.h"

namespace xla {
namespace ifrt {

#define GEN_PASS_DEF_IFRTMERGECOPIESANDRESHARDSPASS
#include "xla/python/ifrt/ir/transforms/passes.h.inc"

namespace {

using mlir::StringAttr;
using mlir::func::FuncOp;

class IfrtMergeCopiesAndReshardsPass
    : public impl::IfrtMergeCopiesAndReshardsPassBase<
          IfrtMergeCopiesAndReshardsPass> {
 public:
  void runOnOperation() override;
};

// We need to group CopyArrays ops by:
//
// - src and dst devices because they require src and dst devices to match.
// - donated because the donation is all-or-nothing.
// - src and dst memory kind because we can't CopyArrays ops that change memory
//   kind.
// - src and dst layout mode because we can't merge CopyArrays ops that change
//   layout mode.
llvm::hash_code GetMergeKey(CopyArraysOp op) {
  // TODO(icgog): A CopyArrays with device memory kind will not be merged with a
  // CopyArrays with default/no memory kind because we don't have device info to
  // canonicalize the memory kind. Fix this limitation.
  llvm::hash_code hash = llvm::hash_value("CopyArraysOp");
  // CopyArrayOps do not support inputs with different src devices nor outputs
  // with different dst devices. Thus, it is safe to only include the first
  // src and dst devices.
  auto input_type = mlir::cast<IfrtArrayType>(op.getInputs().front().getType());
  hash = llvm::hash_combine(hash, input_type.getDevicesAttr());
  auto output_type =
      mlir::cast<IfrtArrayType>(op.getOutputs().front().getType());
  hash = llvm::hash_combine(hash, output_type.getDevicesAttr());
  // We can't hash by the bool itself, and `donated` is a optional attr, so
  // false can be represented by nullptr or BoolAttr(false). So we
  // explicitly convert to BoolAttr.
  hash = llvm::hash_combine(
      hash, mlir::BoolAttr::get(op.getContext(), op.getDonated()));
  hash = llvm::hash_combine(
      hash,
      StringAttr::get(op.getContext(), absl::StrCat(input_type.MemoryKind())));
  hash = llvm::hash_combine(
      hash, StringAttr::get(op.getContext(),
                            absl::StrCat(input_type.LayoutMode().ToString())));
  hash = llvm::hash_combine(
      hash,
      StringAttr::get(op.getContext(), absl::StrCat(output_type.MemoryKind())));
  hash = llvm::hash_combine(
      hash, StringAttr::get(op.getContext(),
                            absl::StrCat(output_type.LayoutMode().ToString())));
  return hash;
}

// We need to group Reshard ops by:
//
// - src and dst devices to avoid merging all Reshard ops into a single one.
// - donated because the donation is all-or-nothing.
llvm::hash_code GetMergeKey(ReshardOp op) {
  llvm::hash_code hash = llvm::hash_value("ReshardOp");
  // Only ReshardOp with one input and output are merged to other ops so its
  // safe to only take into account the first input and output types.
  auto input_type = mlir::cast<IfrtArrayType>(op.getInputs().front().getType());
  hash = llvm::hash_combine(hash, input_type.getDevicesAttr());
  auto output_type =
      mlir::cast<IfrtArrayType>(op.getOutputs().front().getType());
  hash = llvm::hash_combine(hash, output_type.getDevicesAttr());
  // We can't hash by the bool itself, and `donated` is a optional attr, so
  // false can be represented by nullptr or BoolAttr(false). So we explicitly
  // convert to BoolAttr.
  hash = llvm::hash_combine(
      hash, mlir::BoolAttr::get(op.getContext(), op.getDonated()));
  return hash;
}

void RewriteCopyArraysGroup(mlir::IRRewriter& rewriter,
                            llvm::SmallVector<mlir::Operation*>& to_merge) {
  // Create a new op that takes all the inputs of the group ops.
  llvm::SmallVector<mlir::Value> inputs;
  llvm::SmallVector<mlir::Type> output_types;
  llvm::SmallVector<mlir::Location> locs;
  inputs.reserve(to_merge.size());
  output_types.reserve(to_merge.size());
  locs.reserve(to_merge.size());

  bool donated = false;
  for (mlir::Operation* group_op : to_merge) {
    auto copy_arrays_op = mlir::cast<CopyArraysOp>(group_op);
    inputs.append(copy_arrays_op.getInputs().begin(),
                  copy_arrays_op.getInputs().end());
    for (mlir::Value output : copy_arrays_op.getOutputs()) {
      output_types.push_back(output.getType());
    }
    donated = copy_arrays_op.getDonated();
    locs.push_back(group_op->getLoc());
  }

  // Insert the new op just after the last group op in the block order to
  // minimize reordering ops while not violating dominance order after the
  // merge.
  rewriter.setInsertionPointAfter(to_merge.back());
  CopyArraysOp merged_op =
      CopyArraysOp::create(rewriter, rewriter.getFusedLoc(locs),
                           /*outputs=*/output_types,
                           /*control_output=*/
                           IfrtControlType::get(rewriter.getContext()),
                           /*inputs=*/inputs,
                           /*donated=*/donated,
                           /*control_inputs=*/mlir::ValueRange());

  // Replace the original group with the new merged CopyArrays.
  int merged_output_index = 0;
  for (auto group_op : to_merge) {
    auto copy_arrays_op = mlir::cast<CopyArraysOp>(group_op);
    for (auto output : copy_arrays_op.getOutputs()) {
      rewriter.replaceAllUsesWith(output,
                                  merged_op.getOutputs()[merged_output_index]);
      merged_output_index++;
    }
    rewriter.replaceAllUsesWith(copy_arrays_op.getControlOutput(),
                                merged_op.getControlOutput());
    rewriter.eraseOp(group_op);
  }
}

void RewriteReshardGroup(mlir::IRRewriter& rewriter,
                         llvm::SmallVector<mlir::Operation*>& to_merge) {
  // Create a new op that takes all the inputs of the group ops.
  llvm::SmallVector<mlir::Value> inputs;
  llvm::SmallVector<mlir::Type> output_types;
  llvm::SmallVector<mlir::Location> locs;
  inputs.reserve(to_merge.size());
  output_types.reserve(to_merge.size());
  locs.reserve(to_merge.size());

  bool donated = false;
  for (mlir::Operation* group_op : to_merge) {
    auto reshard_op = mlir::cast<ReshardOp>(group_op);
    inputs.append(reshard_op.getInputs().begin(), reshard_op.getInputs().end());
    for (mlir::Value output : reshard_op.getOutputs()) {
      output_types.push_back(output.getType());
    }
    donated = reshard_op.getDonated();
    locs.push_back(group_op->getLoc());
  }

  // Insert the new op just after the last group op in the block order to
  // minimize reordering ops while not violating dominance order after the
  // merge.
  rewriter.setInsertionPointAfter(to_merge.back());
  ReshardOp merged_op =
      ReshardOp::create(rewriter, rewriter.getFusedLoc(locs),
                        /*outputs=*/output_types,
                        /*control_output=*/
                        IfrtControlType::get(rewriter.getContext()),
                        /*inputs=*/inputs,
                        /*donated=*/donated,
                        /*control_inputs=*/mlir::ValueRange());

  // Replace the original group with the new merged Reshard.
  int merged_output_index = 0;
  for (auto group_op : to_merge) {
    auto reshard_op = mlir::cast<ReshardOp>(group_op);
    for (auto output : reshard_op.getOutputs()) {
      rewriter.replaceAllUsesWith(output,
                                  merged_op.getOutputs()[merged_output_index]);
      merged_output_index++;
    }
    rewriter.replaceAllUsesWith(reshard_op.getControlOutput(),
                                merged_op.getControlOutput());
    rewriter.eraseOp(group_op);
  }
}

// Merges CopyArraysOp and ReshardOp in `func_op`. Only ops that do not have
// control dependencies are merged.
//
// TODO(b/492972846): Take into account control dependencies.
bool MergeCopyAndReshardsIgnoringControlDependencies(FuncOp func_op) {
  mlir::IRRewriter rewriter(func_op->getContext());

  // CopyArraysOp and Reshard ops grouped by merging keys. Values are sorted in
  // program order.
  llvm::DenseMap<llvm::hash_code, std::queue<mlir::Operation*>> merge_groups;

  func_op.walk([&](mlir::Operation* op) {
    if (auto copy_arrays_op = mlir::dyn_cast<CopyArraysOp>(op);
        copy_arrays_op != nullptr && !copy_arrays_op->use_empty() &&
        copy_arrays_op.getControlInputs().empty()) {
      merge_groups[GetMergeKey(copy_arrays_op)].push(op);
    } else if (auto reshard_op = mlir::dyn_cast<ReshardOp>(op);
               reshard_op != nullptr && reshard_op.getOutputs().size() == 1 &&
               !reshard_op->use_empty() &&
               reshard_op.getControlInputs().empty()) {
      merge_groups[GetMergeKey(reshard_op)].push(op);
    }
  });

  // Rewrite each group of reshards/copy_arrays while respecting the dominance
  // order.
  bool rewritten = false;
  for (auto& [_, merge_group] : merge_groups) {
    if (merge_group.size() <= 1) {
      continue;
    }

    while (!merge_group.empty()) {
      llvm::SmallVector<mlir::Operation*> to_merge;

      mlir::Operation* op = merge_group.front();
      merge_group.pop();
      to_merge.push_back(op);

      // This could potentially be very expensive as `isBeforeInBlock` is
      // average O(1) but worst case O(n).
      mlir::Operation* first_group_user = *llvm::min_element(
          op->getUsers(), [](mlir::Operation* a, mlir::Operation* b) {
            return a->isBeforeInBlock(b);
          });

      // Only the ops that are before the first user of `op` can be merged with
      // `op` because otherwise it would violate dominance order. `merge_group`
      // is sorted in program order.
      while (!merge_group.empty() &&
             merge_group.front()->isBeforeInBlock(first_group_user)) {
        mlir::Operation* new_group_op = merge_group.front();
        // Update the first user of the group if the new group member has a user
        // that is before the current first user.
        for (mlir::Operation* user : new_group_op->getUsers()) {
          if (user->isBeforeInBlock(first_group_user)) {
            first_group_user = user;
          }
        }
        to_merge.push_back(new_group_op);
        merge_group.pop();
      }

      if (to_merge.size() <= 1) {
        continue;
      }

      if (llvm::isa_and_nonnull<CopyArraysOp>(to_merge.front())) {
        RewriteCopyArraysGroup(rewriter, to_merge);
      } else if (llvm::isa_and_nonnull<ReshardOp>(to_merge.front())) {
        RewriteReshardGroup(rewriter, to_merge);
      } else {
        LOG(FATAL) << "Unsupported op type: "
                   << to_merge.front()->getName().getStringRef().str()
                   << " for merging.";
      }

      rewritten = true;
    }
  }
  return rewritten;
}

void IfrtMergeCopiesAndReshardsPass::runOnOperation() {
  FuncOp func_op = getOperation();
  // We only need to run this pass on IFRT functions.
  if (!func_op->hasAttr(kIfrtFunctionAttrName) &&
      !func_op->hasAttr(kIfrtReshardFunctionAttrName)) {
    return;
  }

  // Run transformation until fixpoint since merging reshard ops may create more
  // opportunities for further merges. This loop runs at most O(num_reshard_ops)
  // times even in the worst case.
  while (true) {
    if (!MergeCopyAndReshardsIgnoringControlDependencies(func_op)) {
      break;
    }
  }
}

}  // namespace
}  // namespace ifrt
}  // namespace xla
