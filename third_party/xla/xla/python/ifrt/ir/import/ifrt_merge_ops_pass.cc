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

#include <cstdint>
#include <queue>
#include <string>

#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/strings/str_cat.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/Hashing.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/TypeSwitch.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/TypeRange.h"
#include "mlir/IR/Types.h"
#include "mlir/IR/Value.h"
#include "mlir/IR/ValueRange.h"
#include "mlir/IR/Visitors.h"
#include "mlir/Support/LLVM.h"
#include "xla/python/ifrt/ir/ifrt_dialect.h"
#include "xla/python/ifrt/ir/ifrt_interfaces.h"
#include "xla/python/ifrt/ir/ifrt_ops.h"
#include "xla/python/ifrt/ir/import/passes.h"

namespace xla {
namespace ifrt {

#define GEN_PASS_DEF_IFRTMERGEOPSPASS
#include "xla/python/ifrt/ir/import/passes.h.inc"

namespace {

using mlir::StringAttr;
using mlir::func::FuncOp;

class IfrtMergeOpsPass : public impl::IfrtMergeOpsPassBase<IfrtMergeOpsPass> {
 public:
  void runOnOperation() override;
};

// Returns true if `op` has non-empty inputs/outputs, all inputs share the same
// devices, and all outputs share the same devices.
bool HasUniformDevices(IfrtPairwiseArrayOpInterface op) {
  if (op.getInputs().empty() || op.getOutputs().empty()) {
    return false;
  }
  IfrtDevicesAttr src_devices =
      GetArrayType(op.getInputs().front()).getDevicesAttr();
  IfrtDevicesAttr dst_devices =
      GetArrayType(op.getOutputs().front()).getDevicesAttr();
  return llvm::all_of(op.getInputs().drop_front(),
                      [&](mlir::Value input) {
                        return GetArrayType(input).getDevicesAttr() ==
                               src_devices;
                      }) &&
         llvm::all_of(op.getOutputs().drop_front(), [&](mlir::Value output) {
           return GetArrayType(output).getDevicesAttr() == dst_devices;
         });
}

// Returns true if `op` has non-empty inputs/outputs, all inputs share the same
// layout mode, and all outputs share the same layout mode.
bool HasUniformLayoutModes(CopyArraysOp op) {
  if (op.getInputs().empty() || op.getOutputs().empty()) {
    return false;
  }
  std::string src_layout =
      GetArrayType(op.getInputs().front()).LayoutMode().ToString();
  std::string dst_layout =
      GetArrayType(op.getOutputs().front()).LayoutMode().ToString();
  return llvm::all_of(op.getInputs().drop_front(),
                      [&](mlir::Value input) {
                        return GetArrayType(input).LayoutMode().ToString() ==
                               src_layout;
                      }) &&
         llvm::all_of(op.getOutputs().drop_front(), [&](mlir::Value output) {
           return GetArrayType(output).LayoutMode().ToString() == dst_layout;
         });
}

// Returns true if `candidate` donates its inputs and shares at least one input
// value with an op already in `to_merge`.
bool SharesDonatedInput(llvm::ArrayRef<IfrtPairwiseArrayOpInterface> to_merge,
                        IfrtPairwiseArrayOpInterface candidate) {
  if (!candidate.getDonated()) {
    return false;
  }
  llvm::SmallDenseSet<mlir::Value, 8> donated_inputs;
  for (IfrtPairwiseArrayOpInterface group_op : to_merge) {
    donated_inputs.insert(group_op.getInputs().begin(),
                          group_op.getInputs().end());
  }
  return llvm::any_of(candidate.getInputs(), [&](mlir::Value input) {
    return donated_inputs.contains(input);
  });
}

// Computes the base merge key for `IfrtPairwiseArrayOpInterface` ops from the
// op name, source/destination devices, donation attribute, and discardable
// attributes.
llvm::hash_code GetBaseMergeKey(IfrtPairwiseArrayOpInterface op) {
  llvm::hash_code hash = llvm::hash_value(op->getName().getStringRef());
  IfrtArrayType input_type = GetArrayType(op.getInputs().front());
  hash = llvm::hash_combine(hash, input_type.getDevicesAttr());
  IfrtArrayType output_type = GetArrayType(op.getOutputs().front());
  hash = llvm::hash_combine(hash, output_type.getDevicesAttr());
  // We can't hash by the bool itself, and `donated` is an optional attr, so
  // false can be represented by nullptr or BoolAttr(false). So we explicitly
  // convert to BoolAttr.
  hash = llvm::hash_combine(
      hash, mlir::BoolAttr::get(op.getContext(), op.getDonated()),
      op->getDiscardableAttrDictionary());
  return hash;
}

// Groups `CopyArraysOp`s by:
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
  // TODO(icgog): Do not include devices in the key after `CopyArrays` supports
  // MPMD.
  // CopyArrayOps do not support inputs with different src devices nor outputs
  // with different dst devices. Thus, it is safe to only include the first
  // src and dst devices.
  llvm::hash_code hash = GetBaseMergeKey(op);
  IfrtArrayType input_type = GetArrayType(op.getInputs().front());
  IfrtArrayType output_type = GetArrayType(op.getOutputs().front());
  hash = llvm::hash_combine(
      hash, mlir::BoolAttr::get(op.getContext(), op.getReuse()));
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

// Groups `ReshardOp`s by:
// - src and dst devices to avoid merging all `Reshard` ops into a single one.
// - donated because the donation is all-or-nothing.
llvm::hash_code GetMergeKey(ReshardOp op) { return GetBaseMergeKey(op); }

// Groups `BitcastArraysOp`s by:
// - src and dst devices to avoid merging `BitcastArraysOp`s across meshes.
// - `donated` because the donation is all-or-nothing.
llvm::hash_code GetMergeKey(BitcastArraysOp op) { return GetBaseMergeKey(op); }

template <typename OpT>
IfrtPairwiseArrayOpInterface CreateMergedPairwiseOp(
    mlir::IRRewriter& rewriter, OpT first_op, mlir::Location loc,
    mlir::TypeRange output_types, mlir::ValueRange inputs) {
  llvm::SmallVector<mlir::Type> result_types(output_types.begin(),
                                             output_types.end());
  result_types.push_back(IfrtControlType::get(rewriter.getContext()));
  typename OpT::Properties props = first_op.getProperties();
  props.operandSegmentSizes = {static_cast<int32_t>(inputs.size()), 0};
  OpT merged = OpT::create(rewriter, loc, result_types, inputs, props,
                           first_op->getDiscardableAttrDictionary().getValue());
  return merged;
}

void RewriteMergedGroup(mlir::IRRewriter& rewriter,
                        llvm::ArrayRef<IfrtPairwiseArrayOpInterface> to_merge) {
  CHECK(!to_merge.empty());
  // Create a new op that takes all the inputs of the group ops.
  llvm::SmallVector<mlir::Value> inputs;
  llvm::SmallVector<mlir::Type> output_types;
  llvm::SmallVector<mlir::Location> locs;
  inputs.reserve(to_merge.size());
  output_types.reserve(to_merge.size());
  locs.reserve(to_merge.size());

  IfrtPairwiseArrayOpInterface first_group_op = to_merge.front();
  const bool donated = first_group_op.getDonated();
  for (IfrtPairwiseArrayOpInterface group_op : to_merge) {
    CHECK_EQ(first_group_op->getName(), group_op->getName());
    CHECK_EQ(donated, group_op.getDonated());
    inputs.append(group_op.getInputs().begin(), group_op.getInputs().end());
    for (mlir::Value output : group_op.getOutputs()) {
      output_types.push_back(output.getType());
    }
    locs.push_back(group_op->getLoc());
  }

  // Insert the new op just after the last group op in the block order to
  // minimize reordering ops while not violating dominance order after the
  // merge.
  rewriter.setInsertionPointAfter(to_merge.back());
  mlir::Location fused_loc = rewriter.getFusedLoc(locs);
  IfrtPairwiseArrayOpInterface merged_op =
      llvm::TypeSwitch<mlir::Operation*, IfrtPairwiseArrayOpInterface>(
          first_group_op.getOperation())
          .Case<CopyArraysOp, ReshardOp, BitcastArraysOp>([&](auto first_op) {
            return CreateMergedPairwiseOp(rewriter, first_op, fused_loc,
                                          output_types, inputs);
          })
          .Default([&](mlir::Operation* op) -> IfrtPairwiseArrayOpInterface {
            LOG(FATAL) << "Unsupported op type: "
                       << op->getName().getStringRef().str() << " for merging.";
          });

  // Replace the original group with the new merged op.
  int merged_output_index = 0;
  for (IfrtPairwiseArrayOpInterface group_op : to_merge) {
    for (mlir::Value output : group_op.getOutputs()) {
      rewriter.replaceAllUsesWith(
          output, merged_op.getOutputs()[merged_output_index++]);
    }
    rewriter.replaceAllUsesWith(group_op.getControlOutput(),
                                merged_op.getControlOutput());
    rewriter.eraseOp(group_op);
  }
}

// Merges `CopyArraysOp`, `ReshardOp`, and `BitcastArraysOp` in `func_op`. Only
// ops that do not have control dependencies are merged.
//
// TODO(b/492972846): Take into account control dependencies.
bool MergeOpsIgnoringControlDependencies(FuncOp func_op) {
  mlir::IRRewriter rewriter(func_op->getContext());

  // Ops grouped by merging keys. Values are sorted in program order.
  llvm::DenseMap<llvm::hash_code, std::queue<IfrtPairwiseArrayOpInterface>>
      merge_groups;

  func_op.walk([&](IfrtPairwiseArrayOpInterface array_op) {
    if (array_op->use_empty() || !array_op.getControlInputs().empty() ||
        array_op.getInputs().empty() || array_op.getOutputs().empty()) {
      return;
    }
    if (CopyArraysOp copy_arrays_op =
            mlir::dyn_cast<CopyArraysOp>(array_op.getOperation());
        copy_arrays_op != nullptr && HasUniformDevices(copy_arrays_op) &&
        HasUniformLayoutModes(copy_arrays_op)) {
      merge_groups[GetMergeKey(copy_arrays_op)].push(array_op);
    } else if (ReshardOp reshard_op =
                   mlir::dyn_cast<ReshardOp>(array_op.getOperation());
               reshard_op != nullptr && HasUniformDevices(reshard_op)) {
      merge_groups[GetMergeKey(reshard_op)].push(array_op);
    } else if (BitcastArraysOp bitcast_op =
                   mlir::dyn_cast<BitcastArraysOp>(array_op.getOperation());
               bitcast_op != nullptr && HasUniformDevices(bitcast_op)) {
      merge_groups[GetMergeKey(bitcast_op)].push(array_op);
    }
  });

  // Rewrite each group of ops while respecting the dominance order.
  bool rewritten = false;
  for (auto& [_, merge_group] : merge_groups) {
    if (merge_group.size() <= 1) {
      continue;
    }

    while (!merge_group.empty()) {
      llvm::SmallVector<IfrtPairwiseArrayOpInterface> to_merge;

      IfrtPairwiseArrayOpInterface op = merge_group.front();
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
        IfrtPairwiseArrayOpInterface new_group_op = merge_group.front();
        if (new_group_op->getName() != op->getName() ||
            SharesDonatedInput(to_merge, new_group_op)) {
          break;
        }
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

      RewriteMergedGroup(rewriter, to_merge);

      rewritten = true;
    }
  }
  return rewritten;
}

void IfrtMergeOpsPass::runOnOperation() {
  FuncOp func_op = getOperation();
  // We only need to run this pass on IFRT functions.
  if (!IsIfrtFunction(func_op)) {
    return;
  }

  // Run transformation until fixpoint since merging ops may create more
  // opportunities for further merges. This loop runs at most O(num_ops) times
  // even in the worst case.
  while (true) {
    if (!MergeOpsIgnoringControlDependencies(func_op)) {
      break;
    }
  }
}

}  // namespace
}  // namespace ifrt
}  // namespace xla
