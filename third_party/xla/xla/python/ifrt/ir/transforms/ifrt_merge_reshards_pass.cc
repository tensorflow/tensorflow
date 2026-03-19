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
#include <tuple>

#include "absl/log/check.h"
#include "absl/strings/str_cat.h"
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
#include "mlir/Support/LLVM.h"
#include "xla/python/ifrt/ir/constants.h"
#include "xla/python/ifrt/ir/ifrt_dialect.h"
#include "xla/python/ifrt/ir/ifrt_ops.h"
#include "xla/python/ifrt/ir/transforms/passes.h"

namespace xla {
namespace ifrt {

#define GEN_PASS_DEF_IFRTMERGERESHARDSPASS
#include "xla/python/ifrt/ir/transforms/passes.h.inc"

namespace {

class IfrtMergeReshardsPass
    : public impl::IfrtMergeReshardsPassBase<IfrtMergeReshardsPass> {
 public:
  void runOnOperation() override;
};

// We need to group reshard ops by:
//
// - src and dst devices because some reshards will be lowered to copies and
//   IFRT CopyArrays requires src and dst devices to match.
// - donated because the donation is all-or-nothing.
// - src and dst memory kind because we can't merge reshards that change
//   memory kind.
using ReshardKey = std::tuple<             //
    /*input_devices=*/IfrtDevicesAttr,     //
    /*output_devices=*/IfrtDevicesAttr,    //
    /*donated=*/mlir::Attribute,           //
    /*src_memory_kind=*/mlir::StringAttr,  //
    /*dst_memory_kind=*/mlir::StringAttr,  //
    /*src_layout_mode=*/mlir::StringAttr,  //
    /*dst_layout_mode=*/mlir::StringAttr>;

ReshardKey GetReshardKey(ReshardOp op) {
  // Only ReshardOp with one input and output are merged to other ops so its
  // safe to only take into account the first input and output types.
  auto input_type = mlir::cast<IfrtArrayType>(op.getInputs().front().getType());
  auto output_type =
      mlir::cast<IfrtArrayType>(op.getOutputs().front().getType());
  // TODO(icgog): A reshard with device memory kind will not be merged with a
  // reshard with default/no memory kind because we don't have device info to
  // canonicalize the memory kind. Fix this limitation.
  return ReshardKey{
      input_type.getDevicesAttr(),
      output_type.getDevicesAttr(),
      // We can't hash by the bool itself, and `donated` is a optional attr, so
      // false can be represented by nullptr or BoolAttr(false). So we
      // explicitly convert to BoolAttr.
      mlir::BoolAttr::get(op.getContext(), op.getDonated()),
      mlir::StringAttr::get(op.getContext(),
                            absl::StrCat(input_type.MemoryKind())),
      mlir::StringAttr::get(op.getContext(),
                            absl::StrCat(output_type.MemoryKind())),
      mlir::StringAttr::get(op.getContext(),
                            input_type.LayoutMode().ToString()),
      mlir::StringAttr::get(op.getContext(),
                            output_type.LayoutMode().ToString()),
  };
}

// Merges reshards in `func_op`. We merge only if the reshard:
//
// - has only one input and output. I.e. it isn't already merged.
// - has no input control dependencies.
// - has the same source and destination devices.
// - has the same `donation` setting.
//
// TODO(b/492972846): Take into account control dependencies.
bool MergeReshardsIgnoringControlDependencies(mlir::func::FuncOp func_op) {
  mlir::IRRewriter rewriter(func_op->getContext());

  // Reshard ops grouped by keys. Values are sorted in program order.
  llvm::DenseMap<ReshardKey, std::queue<ReshardOp>> reshard_groups;

  func_op.walk([&](ReshardOp reshard_op) {
    if (reshard_op.getOutputs().size() == 1 && !reshard_op->use_empty() &&
        reshard_op.getControlInputs().empty()) {
      reshard_groups[GetReshardKey(reshard_op)].push(reshard_op);
    }
  });

  // Rewrite each group of reshards while respecting the dominance order.
  bool rewritten = false;
  for (auto& [_, reshards] : reshard_groups) {
    if (reshards.size() <= 1) {
      continue;
    }

    while (!reshards.empty()) {
      llvm::SmallVector<ReshardOp> group;

      ReshardOp op = reshards.front();
      reshards.pop();
      group.push_back(op);

      // This could potentially be very expensive as `isBeforeInBlock` is
      // average O(1) but worst case O(n).
      mlir::Operation* first_reshard_user = *llvm::min_element(
          op->getUsers(), [](mlir::Operation* a, mlir::Operation* b) {
            return a->isBeforeInBlock(b);
          });

      // Only the reshard ops that are before the first user of `op` can be
      // merged with `op` because otherwise it would violate dominance order.
      // `reshards` is sorted in program order.
      while (!reshards.empty() &&
             reshards.front()->isBeforeInBlock(first_reshard_user)) {
        xla::ifrt::ReshardOp& new_group_reshard = reshards.front();
        // Update the first user of the group if the new reshard group member
        // has a user that is before the current first user.
        for (mlir::Operation* user : new_group_reshard->getUsers()) {
          if (user->isBeforeInBlock(first_reshard_user)) {
            first_reshard_user = user;
          }
        }
        group.push_back(new_group_reshard);
        reshards.pop();
      }
      if (group.size() <= 1) {
        continue;
      }

      // Create a new reshard op that takes all the inputs of the reshards.
      llvm::SmallVector<mlir::Value> inputs;
      llvm::SmallVector<mlir::Type> output_types;
      llvm::SmallVector<mlir::Location> locs;
      inputs.reserve(group.size());
      output_types.reserve(group.size());
      locs.reserve(group.size());

      for (ReshardOp reshard : group) {
        CHECK_EQ(reshard.getInputs().size(), 1);
        CHECK_EQ(reshard.getOutputs().size(), 1);
        inputs.push_back(reshard.getInputs()[0]);
        output_types.push_back(reshard.getOutputs()[0].getType());
        locs.push_back(reshard.getLoc());
      }

      // Insert the new reshard op just after the last reshard in the block
      // order in order to minimize reordering reshards while not violating
      // dominance order after the merge.
      rewriter.setInsertionPointAfter(group.back());
      auto merged_reshard =
          ReshardOp::create(rewriter, rewriter.getFusedLoc(locs),
                            /*outputs=*/output_types,
                            /*control_output=*/
                            IfrtControlType::get(rewriter.getContext()),
                            /*inputs=*/inputs,
                            /*donated=*/group.front().getDonated(),
                            /*control_inputs=*/mlir::ValueRange());

      // Replace the original group with the new merged reshard.
      for (auto [index, reshard] : llvm::enumerate(group)) {
        rewriter.replaceAllUsesWith(reshard.getOutputs()[0],
                                    merged_reshard.getOutputs()[index]);
        rewriter.replaceAllUsesWith(reshard.getControlOutput(),
                                    merged_reshard.getControlOutput());
        rewriter.eraseOp(reshard);
      }
      rewritten = true;
    }
  }
  return rewritten;
}

void IfrtMergeReshardsPass::runOnOperation() {
  mlir::func::FuncOp func_op = getOperation();
  // We only need to run this pass on IFRT functions.
  if (!func_op->hasAttr(kIfrtFunctionAttrName) &&
      !func_op->hasAttr(kIfrtReshardFunctionAttrName)) {
    return;
  }

  // Run transformation until fixpoint since merging reshard ops may create more
  // opportunities for further merges. This loop runs at most O(num_reshard_ops)
  // times even in the worst case.
  while (true) {
    if (!MergeReshardsIgnoringControlDependencies(func_op)) {
      break;
    }
  }
}

}  // namespace
}  // namespace ifrt
}  // namespace xla
