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

#include <tuple>

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

// Merges reshards in `func_op`. We merge only if the reshard:
//
// - has only one input and output. I.e. it isn't already merged.
// - has no input control dependencies.
// - has the same source and destination devices.
// - has the same `donation` setting.
bool MergeReshardsIgnoringControlDependencies(mlir::func::FuncOp func_op) {
  mlir::IRRewriter rewriter(func_op->getContext());

  // We group reshards by {first_user, devices, donated, src_memory_kind,
  // dst_memory_kind}. We need to group by:
  //
  // - first_user because we need to pick some destination. Also, reshard ops
  //   with the same first user are always safe to merge without violating
  //   dominance order because if one of reshard X's arguments is transitively
  //   produced by reshard Y, reshard Y's first user will be either reshard X or
  //   any operation before reshard X.
  //
  // - src and dst devices because some reshards will be lowered to copies and
  //   IFRT CopyArrays requires src and dst devices to match.
  //
  // - donated because the donation is all-or-nothing.
  //
  // - src and dst memory kind because we can't merge reshards that change
  //   memory kind.
  using Key = std::tuple<mlir::Operation*, IfrtDevicesAttr, IfrtDevicesAttr,
                         mlir::Attribute, mlir::StringAttr, mlir::StringAttr>;
  llvm::DenseMap<Key, llvm::SmallVector<ReshardOp>> reshard_groups;

  func_op.walk([&](ReshardOp reshard_op) {
    if (reshard_op.getOutputs().size() != 1 || reshard_op->use_empty() ||
        !reshard_op.getControlInputs().empty()) {
      return;
    }

    // This could potentially be very expensive as `isBeforeInBlock` is
    // average O(1) but worst case O(n).
    mlir::Operation* first_reshard_user = *llvm::min_element(
        reshard_op->getUsers(), [](mlir::Operation* a, mlir::Operation* b) {
          return a->isBeforeInBlock(b);
        });

    auto input_type =
        mlir::cast<IfrtArrayType>(reshard_op.getInputs().front().getType());
    auto output_type =
        mlir::cast<IfrtArrayType>(reshard_op.getOutputs().front().getType());

    const Key key = std::make_tuple(
        first_reshard_user, input_type.getDevicesAttr(),
        output_type.getDevicesAttr(),
        // We can't hash by the bool itself, and `donated` is a optional
        // attr, so false can be represented by nullptr or BoolAttr(false).
        // So we explicitly convert to BoolAttr.
        rewriter.getBoolAttr(reshard_op.getDonated()),
        input_type.getMemoryKindAttr(), output_type.getMemoryKindAttr());
    reshard_groups[key].push_back(reshard_op);
  });

  // Rewrite each group of reshards.
  bool rewritten = false;
  for (auto& [_, reshards] : reshard_groups) {
    if (reshards.size() <= 1) {
      continue;
    }

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

    // Insert the new reshard op just before the last reshard in the block order
    // in order to minimize reordering reshards while not violating dominance
    // order after the merge.
    rewriter.setInsertionPoint(reshards.back());
    auto merged_reshard =
        ReshardOp::create(rewriter, rewriter.getFusedLoc(locs),
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
      rewriter.eraseOp(reshard);
    }

    rewritten = true;
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
