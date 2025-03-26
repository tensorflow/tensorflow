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

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <memory>
#include <optional>
#include <set>
#include <string>
#include <vector>

#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/FormatVariadic.h"
#include "mlir/Analysis/TopologicalSortUtils.h"  // from @llvm-project
#include "mlir/Dialect/Func/IR/FuncOps.h"  // from @llvm-project
#include "mlir/IR/Block.h"  // from @llvm-project
#include "mlir/IR/Builders.h"  // from @llvm-project
#include "mlir/IR/BuiltinAttributes.h"  // from @llvm-project
#include "mlir/IR/BuiltinOps.h"  // from @llvm-project
#include "mlir/IR/BuiltinTypes.h"  // from @llvm-project
#include "mlir/IR/Location.h"  // from @llvm-project
#include "mlir/IR/Matchers.h"  // from @llvm-project
#include "mlir/IR/Types.h"  // from @llvm-project
#include "mlir/IR/Value.h"  // from @llvm-project
#include "mlir/IR/Visitors.h"  // from @llvm-project
#include "mlir/Pass/Pass.h"  // from @llvm-project
#include "mlir/Support/LLVM.h"  // from @llvm-project
#include "mlir/Support/LogicalResult.h"  // from @llvm-project
#include "tensorflow/compiler/mlir/tensorflow/analysis/side_effect_analysis.h"
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_device.h"
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_ops.h"
#include "tensorflow/compiler/mlir/tensorflow/transforms/collection_ops_util.h"
#include "tensorflow/compiler/mlir/utils/name_utils.h"
#include "tensorflow/core/platform/status.h"
#include "tensorflow/core/platform/strcat.h"
#include "tensorflow/core/platform/types.h"
#include "tensorflow/dtensor/cc/constants.h"
#include "tensorflow/dtensor/cc/dstatus.h"
#include "tensorflow/dtensor/mlir/dtensor_location.h"
#include "tensorflow/dtensor/mlir/ir/tf_dtensor.h"
#include "tensorflow/dtensor/mlir/layout_parsing.h"

namespace tensorflow {
namespace dtensor {

namespace {
#define GEN_PASS_DEF_DTENSORALLREDUCECOMBINEOPTIMIZATION
#include "tensorflow/dtensor/mlir/dtensor_passes.h.inc"

namespace ops_util = ::mlir::TF::collection_ops_util;

// Pad the merged tensor shape to multiples of 1024B, so delinearization
// skipping optimization in XLA can get activated.
constexpr int32 kAllReducePadding = 1024;

// Returns true if `successor` depends on `predecessor`.
// TODO(jiawenhao): Repeatedly computing dependency sets for a large cluster can
// get expensive when the number of all-reduces is high. Consider building a
// cluster-scope op dependency graph ahead of time to amortize the cost.
bool DependsOn(mlir::Operation* successor, mlir::Operation* predecessor,
               const mlir::TF::detail::SideEffectAnalysisInfo& info) {
  llvm::SmallVector<mlir::Operation*, 4> to_visit;
  llvm::SmallPtrSet<mlir::Operation*, 4> visited;
  to_visit.push_back(predecessor);
  while (!to_visit.empty()) {
    mlir::Operation* producer = to_visit.pop_back_val();
    if (visited.contains(producer)) continue;
    visited.insert(producer);
    if (successor == producer) return true;
    for (mlir::Operation* user : producer->getUsers()) {
      if (visited.contains(user)) continue;
      to_visit.push_back(user);
    }
    // Include indirectly dependent ops from side effects
    for (mlir::Operation* user : info.DirectControlSuccessors(producer)) {
      if (visited.contains(user)) continue;
      to_visit.push_back(user);
    }
  }
  return false;
}

// Merge all-reduces in the group into one all-reduce.
//
// Requirements:
//   - The group should have at least two all-reduces.
//   - They should be located next to each other in the parent block.
//   - They should all have the same element type.
//   - They should all have the same group assignment.
//
// The merged all-reduce operates on a 1D tensor, whose size is the sum of all
// merged all-reduce tensors padded to 1024B. (The padding is necessary for the
// XLA delinearization skipping logic.) Each to-be-merged all-reduce flattens
// its input tensor and writes the resulting 1D tensor into the corresponding
// offset in the merged 1D tensor. After the merged all-reduce is done, the
// reverse happens: results are sliced out and reshaped to the original shape.
mlir::LogicalResult MergeAllReduceGroup(
    std::vector<mlir::TF::DTensorAllReduceOp>& all_reduce_group) {
  // Create the initial all-zero merged tensor.
  // The merged tensor's size is the sum of all individual all-reduces' sizes.
  int num_all_reduces = all_reduce_group.size();
  DCHECK(num_all_reduces > 1)
      << "All reduce group size expected to be greater than 1.";
  int total_num_elements = 0;
  std::vector<llvm::ArrayRef<int64_t>> all_reduce_shapes;
  all_reduce_shapes.reserve(num_all_reduces);
  for (mlir::TF::DTensorAllReduceOp& all_reduce : all_reduce_group) {
    auto all_reduce_ranked_type =
        mlir::dyn_cast<mlir::RankedTensorType>(all_reduce.getType());
    if (!all_reduce_ranked_type || !all_reduce_ranked_type.hasStaticShape()) {
      return all_reduce.emitOpError(llvm::formatv(
          "requires static shape for DTensorAllReduceOp, but got : {0}",
          all_reduce_ranked_type));
    }
    int num_elements = all_reduce_ranked_type.getNumElements();
    total_num_elements += num_elements;
    all_reduce_shapes.push_back(all_reduce_ranked_type.getShape());
  }

  // Pad the merged tensor shape to multiples of 1024B, so delinearization
  // skipping optimization in XLA can get activated.
  if (total_num_elements % kAllReducePadding != 0) {
    total_num_elements =
        total_num_elements / kAllReducePadding * kAllReducePadding +
        kAllReducePadding;
  }

  // Fill the merged tensor with 0 initially.
  mlir::OpBuilder builder(all_reduce_group[0]);
  mlir::Location loc = all_reduce_group[0].getLoc();
  mlir::Type elem_type = all_reduce_group[0].getType().getElementType();
  auto zero_scalar = ops_util::CreateScalarConst(0, builder, loc);
  auto zero_scalar_elem_type = builder.create<mlir::TF::CastOp>(
      loc, mlir::RankedTensorType::get({}, elem_type), zero_scalar);
  auto merged = builder.create<mlir::TF::FillOp>(
      loc, ops_util::GetR1Const({total_num_elements}, builder, loc),
      zero_scalar_elem_type);

  // Store every all-reduce's input at an offset location in the merged tensor,
  // as a 1D tensor.
  int offset_num_elements = 0;
  std::vector<mlir::Type> flattened_types;
  flattened_types.reserve(num_all_reduces);
  mlir::Value updated;
  for (int i = 0; i < all_reduce_group.size(); ++i) {
    mlir::TF::DTensorAllReduceOp& all_reduce = all_reduce_group[i];
    mlir::Location loc = all_reduce.getLoc();
    auto all_reduce_ranked_type =
        mlir::dyn_cast<mlir::RankedTensorType>(all_reduce.getType());
    if (!all_reduce_ranked_type || !all_reduce_ranked_type.hasStaticShape()) {
      return all_reduce.emitOpError(llvm::formatv(
          "requires static shape for DTensorAllReduceOp, but got : {0}",
          all_reduce_ranked_type));
    }

    int num_elements = all_reduce_ranked_type.getNumElements();
    auto flattened = builder.create<mlir::TF::ReshapeOp>(
        DT_LOC2(loc, "CombinedReduceFlatten"), all_reduce.getInput(),
        ops_util::GetR1Const({num_elements}, builder, loc));
    flattened_types.push_back(flattened.getType());
    auto indices = ops_util::GetR1Const({offset_num_elements}, builder, loc);

    if (all_reduce.getDeviceType().contains("TPU")) {
      updated = builder.create<mlir::TF::XlaDynamicUpdateSliceOp>(
          DT_LOC2(loc, "CombinedReduceUpdateSlice"), merged.getType(),
          /*input=*/i == 0 ? merged.getResult() : updated,
          /*update=*/flattened, indices);
    } else {
      auto end = ops_util::GetR1Const({offset_num_elements + num_elements},
                                      builder, loc);
      auto strides = ops_util::GetR1Const({1}, builder, loc);
      updated = builder.create<mlir::TF::TensorStridedSliceUpdateOp>(
          DT_LOC2(loc, "CombinedReduceUpdateSlice"), merged.getType(),
          /*input=*/i == 0 ? merged.getResult() : updated, indices, end,
          strides,
          /*value=*/flattened);
    }
    offset_num_elements += num_elements;
  }

  // All-reduce the updated merged tensor.
  auto merged_all_reduce = builder.create<mlir::TF::DTensorAllReduceOp>(
      all_reduce_group[0].getLoc(), updated.getType(), updated,
      all_reduce_group[0].getGroupAssignment(),
      all_reduce_group[0].getReduceOp(), all_reduce_group[0].getDeviceType());
  SetSingleLayoutOnOp(
      merged_all_reduce,
      ExtractSingleLayoutFromOp(all_reduce_group[0]).value().value());

  // Slice out the original all-reduces, and reshape back to the original shape.
  offset_num_elements = 0;
  std::vector<mlir::TF::ReshapeOp> replacements;
  replacements.reserve(num_all_reduces);
  for (int i = 0; i < all_reduce_group.size(); ++i) {
    mlir::TF::DTensorAllReduceOp& all_reduce = all_reduce_group[i];
    mlir::Location loc = all_reduce.getLoc();
    auto all_reduce_ranked_type =
        mlir::dyn_cast<mlir::RankedTensorType>(all_reduce.getType());
    if (!all_reduce_ranked_type || !all_reduce_ranked_type.hasStaticShape()) {
      return all_reduce.emitOpError(llvm::formatv(
          "requires static shape for DTensorAllReduceOp, but got : {0}",
          all_reduce_ranked_type));
    }
    int num_elements = all_reduce_ranked_type.getNumElements();
    auto slice = builder.create<mlir::TF::SliceOp>(
        DT_LOC2(loc, "PostCombinedReduceSlice"), flattened_types[i],
        /*input=*/merged_all_reduce,
        /*begin=*/ops_util::GetR1Const({offset_num_elements}, builder, loc),
        /*size=*/ops_util::GetR1Const({num_elements}, builder, loc));
    auto replacement = builder.create<mlir::TF::ReshapeOp>(
        DT_LOC2(loc, "PostCombinedReduceReshape"), slice.getResult(),
        ops_util::GetR1Const(all_reduce_shapes[i], builder, loc));
    replacements.push_back(replacement);
    offset_num_elements += num_elements;
  }

  // Replace usages and clean up.
  for (int i = 0; i < all_reduce_group.size(); ++i) {
    mlir::TF::DTensorAllReduceOp& all_reduce = all_reduce_group[i];
    mlir::TF::ReshapeOp& replacement = replacements[i];
    all_reduce.replaceAllUsesWith(replacement.getResult());
    all_reduce.erase();
  }
  return mlir::success();
}

// Dump the dependencies between AllReduce ops as a DOT graph.
std::string DrawAllReduceDependencies(
    std::vector<mlir::TF::DTensorAllReduceOp> all_reduces,
    const mlir::TF::detail::SideEffectAnalysisInfo& info) {
  std::vector<std::vector<int>> dependents(all_reduces.size(),
                                           std::vector<int>());
  for (int j = 0; j < all_reduces.size(); ++j) {
    mlir::TF::DTensorAllReduceOp later = all_reduces[j];
    for (int i = 0; i < j; ++i) {
      mlir::TF::DTensorAllReduceOp earlier = all_reduces[i];
      DCHECK(!DependsOn(earlier, later, info));
      if (earlier->getBlock() != later->getBlock() ||
          DependsOn(later, earlier, info)) {
        dependents[i].push_back(j);
      }
    }
  }
  std::string output = "digraph all_reduces {\n";
  for (int i = 0; i < dependents.size(); i++) {
    strings::StrAppend(&output, i);
    strings::StrAppend(&output, "\n");
  }
  for (int i = 0; i < dependents.size(); i++) {
    for (int j : dependents[i]) {
      strings::StrAppend(&output, i, " -> ", j, "\n");
    }
  }
  output += "}";
  return output;
}

// Combine cross-slice DTensorAllReduce ops of the same element type and group
// assignment into as few groups as possible. Only independent ops can be
// combined together.
//
// For example, this program:
//
// clang-format off
// NOLINTBEGIN(whitespace/line_length)
// %0 = "tf_device.cluster"() ({
//   %1 = "tf.Const"() {value = dense<0.0> : tensor<4x4xf32>} : () -> tensor<4x4xf32>
//   %2 = "tf.Const"() {value = dense<[[0, 1], [2, 3]]> : tensor<2x2xi32>} : () -> tensor<2x2xi32>
//   %3 = "tf.DTensorAllReduce"(%1, %2) {reduce_op = "Add"} : (tensor<4x4xf32>, tensor<2x2xi32>) -> tensor<4x4xf32>
//   %4 = "tf.Const"() {value = dense<0.0> : tensor<4x4xf32>} : () -> tensor<4x4xf32>
//   %5 = "tf.Const"() {value = dense<[[0, 1], [2, 3]]> : tensor<2x2xi32>} : () -> tensor<2x2xi32>
//   %6 = "tf.DTensorAllReduce"(%4, %5) {reduce_op = "Add"} : (tensor<4x4xf32>, tensor<2x2xi32>) -> tensor<4x4xf32>
//   %7 = "tf.Add"(%3, %6) : (tensor<4x4xf32>, tensor<4x4xf32>) -> tensor<4x4xf32>
//   "tf_device.return"(%7) : (tensor<4x4xf32>) -> ()
// }) : () -> tensor<4x4xf32>
// NOLINTEND
// clang-format on
//
// will become this:
//
// clang-format off
// NOLINTBEGIN(whitespace/line_length)
// %0 = "tf_device.cluster"() ( {
//   %cst = "tf.Const"() {value = dense<0.000000e+00> : tensor<4x4xf32>} : () -> tensor<4x4xf32>
//   %cst_0 = "tf.Const"() {value = dense<[[0, 1], [2, 3]]> : tensor<2x2xi32>} : () -> tensor<2x2xi32>
//   %cst_1 = "tf.Const"() {value = dense<0.000000e+00> : tensor<4x4xf32>} : () -> tensor<4x4xf32>
//   %cst_2 = "tf.Const"() {value = dense<[[0, 1], [2, 3]]> : tensor<2x2xi32>} : () -> tensor<2x2xi32>
//   %cst_3 = "tf.Const"() {value = dense<0> : tensor<i32>} : () -> tensor<i32>
//   %1 = "tf.Cast"(%cst_3) {Truncate = false} : (tensor<i32>) -> tensor<f32>
//   %cst_4 = "tf.Const"() {value = dense<1024> : tensor<1xi32>} : () -> tensor<1xi32>
//   %2 = "tf.Fill"(%cst_4, %1) : (tensor<1xi32>, tensor<f32>) -> tensor<1024xf32>
//   %cst_5 = "tf.Const"() {value = dense<16> : tensor<1xi32>} : () -> tensor<1xi32>
//   %3 = "tf.Reshape"(%cst, %cst_5) : (tensor<4x4xf32>, tensor<1xi32>) -> tensor<16xf32>
//   %cst_6 = "tf.Const"() {value = dense<0> : tensor<1xi32>} : () -> tensor<1xi32>
//   %4 = "tf.XlaDynamicUpdateSlice"(%2, %3, %cst_6) : (tensor<1024xf32>, tensor<16xf32>, tensor<1xi32>) -> tensor<1024xf32>
//   %cst_7 = "tf.Const"() {value = dense<16> : tensor<1xi32>} : () -> tensor<1xi32>
//   %5 = "tf.Reshape"(%cst_1, %cst_7) : (tensor<4x4xf32>, tensor<1xi32>) -> tensor<16xf32>
//   %cst_8 = "tf.Const"() {value = dense<16> : tensor<1xi32>} : () -> tensor<1xi32>
//   %6 = "tf.XlaDynamicUpdateSlice"(%4, %5, %cst_8) : (tensor<1024xf32>, tensor<16xf32>, tensor<1xi32>) -> tensor<1024xf32>
//   %7 = "tf.DTensorAllReduce"(%6, %cst_0) {reduce_op = "Add"} : (tensor<1024xf32>, tensor<2x2xi32>) -> tensor<1024xf32>
//   %cst_9 = "tf.Const"() {value = dense<0> : tensor<1xi32>} : () -> tensor<1xi32>
//   %cst_10 = "tf.Const"() {value = dense<16> : tensor<1xi32>} : () -> tensor<1xi32>
//   %8 = "tf.Slice"(%7, %cst_9, %cst_10) : (tensor<1024xf32>, tensor<1xi32>, tensor<1xi32>) -> tensor<16xf32>
//   %cst_11 = "tf.Const"() {value = dense<4> : tensor<2xi32>} : () -> tensor<2xi32>
//   %9 = "tf.Reshape"(%8, %cst_11) : (tensor<16xf32>, tensor<2xi32>) -> tensor<4x4xf32>
//   %cst_12 = "tf.Const"() {value = dense<16> : tensor<1xi32>} : () -> tensor<1xi32>
//   %cst_13 = "tf.Const"() {value = dense<16> : tensor<1xi32>} : () -> tensor<1xi32>
//   %10 = "tf.Slice"(%7, %cst_12, %cst_13) : (tensor<1024xf32>, tensor<1xi32>, tensor<1xi32>) -> tensor<16xf32>
//   %cst_14 = "tf.Const"() {value = dense<4> : tensor<2xi32>} : () -> tensor<2xi32>
//   %11 = "tf.Reshape"(%10, %cst_14) : (tensor<16xf32>, tensor<2xi32>) -> tensor<4x4xf32>
//   %12 = "tf.Add"(%9, %11) : (tensor<4x4xf32>, tensor<4x4xf32>) -> tensor<4x4xf32>
//   tf_device.return %12 : tensor<4x4xf32>
// }) : () -> tensor<4x4xf32>
// NOLINTEND
// clang-format on
mlir::LogicalResult CombineAllReduceOps(
    mlir::tf_device::ClusterOp cluster,
    std::vector<mlir::TF::DTensorAllReduceOp>& all_reduces) {
  // A single op has nothing to combine with.
  int num_all_reduces = all_reduces.size();
  if (num_all_reduces <= 1) return mlir::success();

  // Move all-reduces in the same group together and combine them.
  auto& all_reduce_group = all_reduces;
  mlir::TF::DTensorAllReduceOp final_all_reduce =
      all_reduce_group[num_all_reduces - 1];

  for (int i = num_all_reduces - 2; i >= 0; --i) {
    mlir::TF::DTensorAllReduceOp all_reduce = all_reduce_group[i];
    all_reduce->moveBefore(final_all_reduce);
  }
  auto merge_result = MergeAllReduceGroup(all_reduce_group);
  if (merge_result.failed()) return merge_result;

  return mlir::success();
}

// Returns true if both group assignments are constant and equal.
bool same_group_assignments(mlir::Value group_assignment_a,
                            mlir::Value group_assignment_b) {
  if (group_assignment_a == group_assignment_b) {
    return true;
  }
  mlir::DenseIntElementsAttr attr_a;
  if (!matchPattern(group_assignment_a, m_Constant(&attr_a))) {
    return false;
  }
  mlir::DenseIntElementsAttr attr_b;
  if (!matchPattern(group_assignment_b, m_Constant(&attr_b))) {
    return false;
  }
  if (attr_a.getType().getShape() != attr_b.getType().getShape()) {
    return false;
  }
  // Group assignment should not be empty.
  DCHECK(!attr_a.empty() && !attr_b.empty());

  return std::equal(attr_a.begin(), attr_a.end(), attr_b.begin(), attr_b.end());
}

std::vector<std::vector<mlir::TF::DTensorAllReduceOp>>
createIndependentReduceOpsGroups(
    const std::vector<mlir::TF::DTensorAllReduceOp>& ordered_all_reduces,
    const mlir::TF::detail::SideEffectAnalysisInfo& info) {
  // Build a reverse adjacency matrix from node to its dependents.
  std::vector<std::vector<int>> dependents(ordered_all_reduces.size(),
                                           std::vector<int>());
  auto num_all_reduces = ordered_all_reduces.size();
  for (int i = 0; i < num_all_reduces - 1; ++i) {
    mlir::TF::DTensorAllReduceOp requirement = ordered_all_reduces[i];
    for (int j = i + 1; j < num_all_reduces; ++j) {
      mlir::TF::DTensorAllReduceOp dependent = ordered_all_reduces[j];
      DCHECK(!DependsOn(requirement, dependent,
                        info));  // guaranteed by program order
      // In this example, all three DTensorAllReduce ops are independent
      // from each other according to MLIR value use-def chains considered
      // by DependsOn. However, moving all three to after the WhileRegion
      // and combine them would break the program.
      //
      // %3 = tf.DTensorAllReduce(%1, %2)
      // %4 = tf.WhileRegion(%1) ({
      // ^bb0(%arg):
      //   %5 = tf.TooBool(%arg)
      //   tf.Yield(%5)
      // }, {
      //   %6 = tf.DTensorAllReduce(%1, %2)
      //   tf.Yield(%5)
      // })
      // %7 = tf.DTensorAllReduce(%1, %2)
      //
      // Therefore, in addition to DependsOn, we also check if two
      // DTensorAllReduceOps belong to different blocks. If they do, since
      // they exist in the same ClusterOp, one or both of them must be
      // inside a control flow region block. We treat them as if there is
      // a dependency between them.
      //
      // In the example above, the second DTensorAllReduceOp would "depend
      // on" the first one, and the third on the second. This effectively
      // prevents any two DTensorAllReduce from merging together.
      if (requirement->getBlock() != dependent->getBlock() ||
          DependsOn(dependent, requirement, info)) {
        dependents[i].push_back(j);
      }
    }
  }

  // Traverse the adjacency matrix layer by layer from last op to find
  // combination groups.
  std::vector<std::vector<mlir::TF::DTensorAllReduceOp>> all_reduce_groups;
  std::set<int> fulfilled;
  while (fulfilled.size() < ordered_all_reduces.size()) {
    std::vector<mlir::TF::DTensorAllReduceOp> group;
    std::vector<int64_t> group_ids;
    for (int i = dependents.size() - 1; i >= 0; i--) {
      if (fulfilled.count(i) > 0) continue;  // Already added op
      bool all_deps_added = true;
      for (int j = dependents[i].size() - 1; j >= 0; j--) {
        if (fulfilled.count(dependents[i][j]) == 0) {
          all_deps_added = false;
          break;
        }
      }
      if (all_deps_added) {
        // Node with no dependents/already captured dependents degrees.
        group_ids.push_back(i);
      }
    }

    std::sort(group_ids.begin(), group_ids.end(),
              [](const int64_t lhs, const int64_t rhs) { return lhs < rhs; });

    for (auto x : group_ids) {
      group.push_back(ordered_all_reduces[x]);
      fulfilled.insert(x);
    }
    all_reduce_groups.push_back(group);
  }

  // Export the all reduces as a DOT graph.
  VLOG(4) << "Visualizing AllReduce dependencies:\n"
          << DrawAllReduceDependencies(ordered_all_reduces, info);
  return all_reduce_groups;
}

std::vector<std::vector<mlir::TF::DTensorAllReduceOp>>
createSubgroupsByElemType(
    std::vector<std::vector<mlir::TF::DTensorAllReduceOp>> all_reduce_groups) {
  std::vector<std::vector<mlir::TF::DTensorAllReduceOp>> all_reduce_new_groups;
  // Combine all-reduces of the same element type.
  for (const auto& all_reduce_group : all_reduce_groups) {
    llvm::DenseMap<mlir::Type, std::vector<mlir::TF::DTensorAllReduceOp>>
        all_reduces_by_elem_type;
    for (auto all_reduce : all_reduce_group) {
      mlir::Type elem_type = all_reduce.getType().getElementType();
      all_reduces_by_elem_type[elem_type].push_back(all_reduce);
    }

    for (const auto& elem_type_pair : all_reduces_by_elem_type) {
      all_reduce_new_groups.push_back(elem_type_pair.second);
    }
  }
  VLOG(4) << "current number of groups: " << all_reduce_new_groups.size()
          << " after grouping by element type.";
  return all_reduce_new_groups;
}

std::vector<std::vector<mlir::TF::DTensorAllReduceOp>>
createSubgroupsByReductionAttr(
    std::vector<std::vector<mlir::TF::DTensorAllReduceOp>> all_reduce_groups) {
  std::vector<std::vector<mlir::TF::DTensorAllReduceOp>> all_reduce_new_groups;
  // Combine all-reduces of the same reduction attribute.
  for (const auto& all_reduce_group : all_reduce_groups) {
    llvm::DenseMap<llvm::StringRef, std::vector<mlir::TF::DTensorAllReduceOp>>
        all_reduces_by_attr_reduce_op;
    for (mlir::TF::DTensorAllReduceOp all_reduce : all_reduce_group) {
      llvm::StringRef attr_reduce_op = all_reduce.getReduceOp();
      all_reduces_by_attr_reduce_op[attr_reduce_op].push_back(all_reduce);
    }
    for (const auto& all_reduces_for_reduce_op_attr :
         all_reduces_by_attr_reduce_op) {
      all_reduce_new_groups.push_back(all_reduces_for_reduce_op_attr.second);
    }
  }
  VLOG(4) << "current number of groups: " << all_reduce_new_groups.size()
          << " after grouping by reduction attribute.";
  return all_reduce_new_groups;
}

std::vector<std::vector<mlir::TF::DTensorAllReduceOp>>
createSubgroupsByGroupAssignment(
    std::vector<std::vector<mlir::TF::DTensorAllReduceOp>> all_reduce_groups) {
  std::vector<std::vector<mlir::TF::DTensorAllReduceOp>> all_reduce_new_groups;
  // Combine all-reduces of the group assignment.
  for (const auto& all_reduce_group : all_reduce_groups) {
    std::vector<mlir::Value> group_assignments;
    llvm::DenseMap<mlir::Value, std::vector<mlir::TF::DTensorAllReduceOp>>
        all_reduces_by_group_assignment;
    for (mlir::TF::DTensorAllReduceOp all_reduce : all_reduce_group) {
      mlir::Value group_assignment = all_reduce.getGroupAssignment();
      bool seen = false;
      for (mlir::Value seen_group_assignment : group_assignments) {
        if (same_group_assignments(group_assignment, seen_group_assignment)) {
          group_assignment = seen_group_assignment;
          seen = true;
          break;
        }
      }
      if (!seen) group_assignments.push_back(group_assignment);
      all_reduces_by_group_assignment[group_assignment].push_back(all_reduce);
    }
    for (const auto& all_reduce_group_to_merge :
         all_reduces_by_group_assignment) {
      all_reduce_new_groups.push_back(all_reduce_group_to_merge.second);
    }
  }
  VLOG(4) << "current number of groups: " << all_reduce_new_groups.size()
          << " after grouping by group assignment.";
  return all_reduce_new_groups;
}

// Experimental extended grouping logics to avoid aggressive grouping.
// This function performs the same grouping method as tf.distribute, which group
// all reduce ops by user defined group size (number of ops) in the input order.
// Note that group_size will be in range of [0, INT_MAX]. It is advised to pick
// a value based on knowledge of the total number of AllReduces. When group_size
// is too big, the function will act as aggressive grouping. When group_size is
// too small, the function will act as having no extended grouping.
std::vector<std::vector<mlir::TF::DTensorAllReduceOp>>
createSubgroupsByExtendedNumOps(
    std::vector<std::vector<mlir::TF::DTensorAllReduceOp>> all_reduce_groups,
    int group_size) {
  VLOG(4) << "max number of ops in a all-reduce group: " << group_size;
  // Disable extended grouping if group size is set to zero
  if (group_size <= 0) return all_reduce_groups;
  std::vector<std::vector<mlir::TF::DTensorAllReduceOp>> all_reduce_new_groups;
  // Further break down the current all_reduced_groups by extended group size
  for (const auto& all_reduce_group : all_reduce_groups) {
    if (all_reduce_group.size() <= group_size) {
      all_reduce_new_groups.push_back(all_reduce_group);
      continue;
    }
    // Safe to "assume" num_groups would be greater or equal to two, because the
    // above condition check guarantees case of zero or one would be skipped
    int num_groups = (all_reduce_group.size() + group_size - 1) / group_size;
    VLOG(4) << all_reduce_group.size() << " all_reduce ops in the current group"
            << ", able to split into " << num_groups << " groups\n";
    for (int i = 0; i < num_groups - 1; i++) {
      all_reduce_new_groups.push_back(std::vector<mlir::TF::DTensorAllReduceOp>(
          all_reduce_group.begin() + i * group_size,
          all_reduce_group.begin() + (i + 1) * group_size));
    }
    // Handle the last sub-group
    all_reduce_new_groups.push_back(std::vector<mlir::TF::DTensorAllReduceOp>(
        all_reduce_group.begin() + (num_groups - 1) * group_size,
        all_reduce_group.end()));
  }
  VLOG(4) << "current number of groups: " << all_reduce_new_groups.size()
          << " after grouping by extended num ops size.";
  return all_reduce_new_groups;
}

// Experimental grouping logics to optimize from aggressive grouping.
// This function first sort by topological level, then create AllReduce sub-
// groups by accessing each topological distance from its previous AllReduce.
// Note that topo_dist will be in range of [0, INT_MAX]. It is advised to select
// a value based on knowledge of the compute graph, such as the minimum distance
// between two model layers. When topo_dist is too big, the function will act
// as aggressive grouping. When topo_dist is too small, the function will act as
// having no extended grouping.
StatusOr<std::vector<std::vector<mlir::TF::DTensorAllReduceOp>>>
createSubgroupsByTopoDist(
    std::vector<std::vector<mlir::TF::DTensorAllReduceOp>> all_reduce_groups,
    llvm::DenseMap<mlir::TF::DTensorAllReduceOp, int> all_reduce_topo,
    int topo_dist) {
  // Disable extended grouping if topological distance is set to zero or less
  if (topo_dist <= 0) return all_reduce_groups;
  std::vector<std::vector<mlir::TF::DTensorAllReduceOp>> all_reduce_new_groups;

  // Further break down the current all_reduced_groups by topological distance
  // between two ops
  for (auto& all_reduce_group : all_reduce_groups) {
    std::vector<mlir::TF::DTensorAllReduceOp> new_group;
    absl::Status status = absl::OkStatus();

    // Sort AllReduces by topological level as the input order may not reflect
    // their dependencies on the operands in the compute graph.
    std::sort(all_reduce_group.begin(), all_reduce_group.end(),
              [&all_reduce_topo, &status](mlir::TF::DTensorAllReduceOp& lhs,
                                          mlir::TF::DTensorAllReduceOp& rhs) {
                if ((all_reduce_topo.find(lhs) == all_reduce_topo.end()) ||
                    (all_reduce_topo.find(rhs) == all_reduce_topo.end())) {
                  status = absl::InternalError(
                      "Error: encounter AllReduce op with no topological level"
                      " assignment.");
                  return false;
                }
                return all_reduce_topo[lhs] < all_reduce_topo[rhs];
              });
    // Unable to sort AllReduces based on topological level due to error. Return
    // directly as we are not able to group based on incorrect/partial topology.
    if (!status.ok()) return status;

    // Form AllReduce groups based on the topological distance between ops
    DCHECK(!all_reduce_group.empty());
    int prev_topo_level = all_reduce_topo[all_reduce_group[0]];
    for (const auto& all_reduce : all_reduce_group) {
      DCHECK(all_reduce_topo.find(all_reduce) != all_reduce_topo.end());
      int cur_topo_level = all_reduce_topo[all_reduce];
      if (abs(cur_topo_level - prev_topo_level) <= topo_dist) {
        new_group.push_back(all_reduce);
      } else {
        // Start a new group
        all_reduce_new_groups.push_back(
            std::vector<mlir::TF::DTensorAllReduceOp>(new_group.begin(),
                                                      new_group.end()));
        new_group.clear();
        new_group.push_back(all_reduce);
      }
      prev_topo_level = cur_topo_level;
    }
    all_reduce_new_groups.push_back(new_group);
  }
  VLOG(4) << "current number of groups: " << all_reduce_new_groups.size()
          << " after grouping by topological distance.";
  return all_reduce_new_groups;
}

// Compute the topological level for each AllReduce op in a cluster. The level
// is defined as 1 + max operands' depth in the compute graph. If an op do not
// depend on any input/operand, then it is level 0.
llvm::DenseMap<mlir::TF::DTensorAllReduceOp, int> computeAllReduceTopoLevel(
    mlir::tf_device::ClusterOp cluster) {
  llvm::DenseMap<mlir::Operation*, int> op_topo_level;
  llvm::DenseMap<mlir::TF::DTensorAllReduceOp, int> all_reduce_topo;

  // Compute topological level for each op.
  cluster.getBody().walk([&](mlir::Operation* op) {
    int max_depth = 0;
    for (mlir::Value operand : op->getOperands()) {
      if (mlir::Operation* operand_op = operand.getDefiningOp()) {
        if (op_topo_level.find(operand_op) != op_topo_level.end()) {
          max_depth = fmax(max_depth, op_topo_level[operand_op]);
        }
      }
    }
    op_topo_level[op] = max_depth + 1;

    // Save the AllReduce topological level
    mlir::TF::DTensorAllReduceOp all_reduce =
        llvm::dyn_cast<mlir::TF::DTensorAllReduceOp>(op);
    if (all_reduce && !all_reduce.getDeviceType().contains("TPU")) {
      all_reduce_topo[all_reduce] = op_topo_level[op];
    }
  });

  return all_reduce_topo;
}

struct DTensorAllReduceCombineOptimization
    : public impl::DTensorAllReduceCombineOptimizationBase<
          DTensorAllReduceCombineOptimization> {
  void runOnOperation() override {
    mlir::func::FuncOp function = getOperation();
    auto module = function->getParentOfType<mlir::ModuleOp>();
    function.walk([&](mlir::tf_device::ClusterOp cluster) {
      std::vector<mlir::TF::DTensorAllReduceOp> ordered_all_reduces;
      std::vector<mlir::Block*> ordered_blocks;
      llvm::DenseSet<mlir::Block*> blocks;
      cluster.GetBody().walk([&](mlir::TF::DTensorAllReduceOp all_reduce) {
        if (!all_reduce.getDeviceType().contains("TPU")) {
          // Only combine all reduces for GPU and CPU
          mlir::RankedTensorType all_reduce_ranked_type =
              mlir::dyn_cast<mlir::RankedTensorType>(all_reduce.getType());

          if (all_reduce_ranked_type &&
              all_reduce_ranked_type.hasStaticShape()) {
            // Static known shape is required to merge all reduces. If shape is
            // not known skip merging.
            ordered_all_reduces.push_back(all_reduce);

            blocks.insert(all_reduce->getBlock());
          }
        }
      });

      if (ordered_all_reduces.size() > 1) {
        VLOG(2) << ordered_all_reduces.size()
                << " all-reduce ops eligible for combine optimization.";
        // Build side effect analysis to identify indirect dependencies between
        // all eligible all_reduce operations
        mlir::TF::SideEffectAnalysis side_effect_analysis(module);
        const mlir::TF::detail::SideEffectAnalysisInfo& info =
            side_effect_analysis.GetAnalysisForFunc(function);
        // Create dependency graph for all eligible all_reduce operations,
        // so that independent ops can be merged
        auto all_reduce_groups =
            createIndependentReduceOpsGroups(ordered_all_reduces, info);

        all_reduce_groups = createSubgroupsByElemType(all_reduce_groups);
        all_reduce_groups = createSubgroupsByReductionAttr(all_reduce_groups);
        all_reduce_groups = createSubgroupsByGroupAssignment(all_reduce_groups);

        // Experimental extended grouping: topological distance
        if (module->hasAttrOfType<mlir::IntegerAttr>(
                kAllReduceTopologicalDistance)) {
          llvm::DenseMap<mlir::TF::DTensorAllReduceOp, int> all_reduce_topo =
              computeAllReduceTopoLevel(cluster);

          StatusOr<std::vector<std::vector<mlir::TF::DTensorAllReduceOp>>>
              group = createSubgroupsByTopoDist(
                  all_reduce_groups, all_reduce_topo,
                  module
                      ->getAttrOfType<mlir::IntegerAttr>(
                          kAllReduceTopologicalDistance)
                      .getInt());
          if (!group.ok()) {
            // This is a non-fatal error since topological level distance is one
            // of the optimizations in this combiner pass. Output an error and
            // continue with the rest of the grouping optimization.
            LOG(WARNING) << "Failed to create subgroups using topological "
                         << "level distance: " << group.status();
          } else {
            all_reduce_groups = group.value();
          }
        }

        // Experimental extended grouping: fixed number of AllReduce ops
        if (module->hasAttrOfType<mlir::IntegerAttr>(kAllReduceNumOpsInGroup)) {
          all_reduce_groups = createSubgroupsByExtendedNumOps(
              all_reduce_groups,
              module->getAttrOfType<mlir::IntegerAttr>(kAllReduceNumOpsInGroup)
                  .getInt());
        }

        // Maintain relative order of ALLReduces within the block.
        std::sort(all_reduce_groups.begin(), all_reduce_groups.end(),
                  [](std::vector<mlir::TF::DTensorAllReduceOp> lhs,
                     std::vector<mlir::TF::DTensorAllReduceOp> rhs) {
                    // Prefer groups that are not empty.
                    if (lhs.empty() && !rhs.empty()) return false;
                    if (!lhs.empty() && rhs.empty()) return true;

                    // Then prefer groups that are in earlier-in-memory blocks,
                    // this part just needs to be consistent for strict weak
                    // ordering purposes.
                    if (lhs[0]->getBlock() != rhs[0]->getBlock()) {
                      return lhs[0]->getBlock() < rhs[0]->getBlock();
                    }

                    // Within the block, use the group's actual sorting.
                    return lhs[0]->isBeforeInBlock(rhs[0]);
                  });

        VLOG(2) << ordered_all_reduces.size() << " all-reduce ops in "
                << all_reduce_groups.size() << " groups";

        for (auto& reduce_group : all_reduce_groups) {
          if (reduce_group.size() > 1) {
            VLOG(4) << "Combining following reduce ops into one: ------------";
            for (auto reduce_op : reduce_group) {
              VLOG(4) << mlir::GetNameFromLoc(reduce_op.getLoc());
            }
            VLOG(4) << "-----------------------------------------------------";
          }
          if (mlir::failed(CombineAllReduceOps(cluster, reduce_group))) {
            return signalPassFailure();
          }
        }

        for (auto* b : blocks) {
          mlir::sortTopologically(b);
        }
      }
    });
  }
};

}  // namespace

std::unique_ptr<mlir::OperationPass<mlir::func::FuncOp>>
CreateDTensorAllReduceCombineOptimization() {
  return std::make_unique<DTensorAllReduceCombineOptimization>();
}

}  // namespace dtensor
}  // namespace tensorflow
