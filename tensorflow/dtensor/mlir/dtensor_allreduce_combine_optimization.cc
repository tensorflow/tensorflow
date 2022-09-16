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
#include <memory>
#include <string>
#include <vector>

#include "llvm/ADT/DenseMap.h"
#include "llvm/Support/FormatVariadic.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"  // from @llvm-project
#include "mlir/IR/BuiltinOps.h"  // from @llvm-project
#include "mlir/IR/Types.h"  // from @llvm-project
#include "mlir/IR/Visitors.h"  // from @llvm-project
#include "mlir/Pass/Pass.h"  // from @llvm-project
#include "mlir/Support/LogicalResult.h"  // from @llvm-project
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_device.h"
#include "tensorflow/compiler/mlir/tensorflow/transforms/collection_ops_util.h"
#include "tensorflow/core/platform/str_util.h"
#include "tensorflow/dtensor/cc/constants.h"
#include "tensorflow/dtensor/cc/dtensor_utils.h"
#include "tensorflow/dtensor/mlir/dtensor_mlir_passes.h"
#include "tensorflow/dtensor/mlir/dtensor_mlir_passes_classes.h"
#include "tensorflow/dtensor/mlir/group_assignment.h"
#include "tensorflow/dtensor/mlir/ir/tf_dtensor.h"
#include "tensorflow/dtensor/mlir/layout_parsing.h"

namespace tensorflow {
namespace dtensor {

namespace {

namespace ops_util = ::mlir::TF::collection_ops_util;

// Pad the merged tensor shape to multiples of 1024B, so delinearization
// skipping optimization in XLA can get activated.
constexpr int32 kAllReducePadding = 1024;

// Returns true if `successor` depends on `predecessor`.
// TODO(jiawenhao): Repeatedly computing dependency sets for a large cluster can
// get expensive when the number of all-reduces is high. Consider building a
// cluster-scope op dependency graph ahead of time to amortize the cost.
bool DependsOn(mlir::Operation* successor, mlir::Operation* predecessor) {
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
  }
  return false;
}

// Moves all usages of `a` (direct and transitive) to right after `b` in
// `cluster`, preserving the original order of moved ops.
// `a` and `b` must be in `cluster`. `a` must appear before `b` originally.
// `a` itself is not moved.
//
// For example, this program:
//
// tf_device.cluster() ({
//   %a = tf.A()
//   %1 = tf.C(%a)
//   %2 = tf.D(%a)
//   %3 = tf.E(%1, %2)
//   %b = tf.B()
//   %4 = tf.F(%3)
//   %5 = tf.G(%b)
//   tf_device.return()
// })
//
// will become this:
//
// tf_device.cluster() ({
//   %a = tf.A()
//   %b = tf.B()
//   %1 = tf.C(%a)
//   %2 = tf.D(%a)
//   %3 = tf.E(%1, %2)
//   %4 = tf.F(%3)
//   %5 = tf.G(%b)
//   tf_device.return()
// })
void MoveUsagesAfter(mlir::tf_device::ClusterOp cluster, mlir::Operation* a,
                     mlir::Operation* b) {
  llvm::SmallVector<mlir::Operation*, 4> to_visit;
  llvm::SmallPtrSet<mlir::Operation*, 4> visited;
  to_visit.push_back(a);
  while (!to_visit.empty()) {
    mlir::Operation* producer = to_visit.pop_back_val();
    if (visited.contains(producer)) continue;
    visited.insert(producer);
    for (mlir::Operation* user : producer->getUsers()) {
      if (visited.contains(user)) continue;
      to_visit.push_back(user);
    }
  }

  llvm::SmallVector<mlir::Operation*, 4> to_move;
  cluster.GetBody().walk([&](mlir::Operation* op) {
    if (op != a && visited.contains(op) && op->isBeforeInBlock(b)) {
      to_move.push_back(op);
    }
  });

  mlir::Operation* last = b;
  for (mlir::Operation* op : to_move) {
    if (mlir::dyn_cast<mlir::TF::YieldOp>(op)) {
      LOG(FATAL) << "Should never move YieldOp";  // Crash OK
    }
    op->moveAfter(last);
    last = op;
  }
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
        all_reduce.getType().dyn_cast<mlir::RankedTensorType>();
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
  mlir::TF::XlaDynamicUpdateSliceOp updated;
  for (int i = 0; i < all_reduce_group.size(); ++i) {
    mlir::TF::DTensorAllReduceOp& all_reduce = all_reduce_group[i];
    mlir::Location loc = all_reduce.getLoc();
    auto all_reduce_ranked_type =
        all_reduce.getType().dyn_cast<mlir::RankedTensorType>();
    if (!all_reduce_ranked_type || !all_reduce_ranked_type.hasStaticShape()) {
      return all_reduce.emitOpError(llvm::formatv(
          "requires static shape for DTensorAllReduceOp, but got : {0}",
          all_reduce_ranked_type));
    }

    int num_elements = all_reduce_ranked_type.getNumElements();
    auto flattened = builder.create<mlir::TF::ReshapeOp>(
        loc, all_reduce.input(),
        ops_util::GetR1Const({num_elements}, builder, loc));
    flattened_types.push_back(flattened.getType());
    auto indices = ops_util::GetR1Const({offset_num_elements}, builder, loc);
    updated = builder.create<mlir::TF::XlaDynamicUpdateSliceOp>(
        loc, merged.getType(),
        /*input=*/i == 0 ? merged.getResult() : updated.getResult(),
        /*update=*/flattened, indices);
    offset_num_elements += num_elements;
  }

  // All-reduce the updated merged tensor.
  auto merged_all_reduce = builder.create<mlir::TF::DTensorAllReduceOp>(
      all_reduce_group[0].getLoc(), updated.getType(), updated,
      all_reduce_group[0].group_assignment(), all_reduce_group[0].reduce_op(),
      all_reduce_group[0].device_type());
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
        all_reduce.getType().dyn_cast<mlir::RankedTensorType>();
    if (!all_reduce_ranked_type || !all_reduce_ranked_type.hasStaticShape()) {
      return all_reduce.emitOpError(llvm::formatv(
          "requires static shape for DTensorAllReduceOp, but got : {0}",
          all_reduce_ranked_type));
    }
    int num_elements = all_reduce_ranked_type.getNumElements();
    auto slice = builder.create<mlir::TF::SliceOp>(
        loc, flattened_types[i], /*input=*/merged_all_reduce,
        /*begin=*/ops_util::GetR1Const({offset_num_elements}, builder, loc),
        /*size=*/ops_util::GetR1Const({num_elements}, builder, loc));
    auto replacement = builder.create<mlir::TF::ReshapeOp>(
        loc, slice.getResult(),
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
    std::vector<mlir::TF::DTensorAllReduceOp> all_reduces) {
  std::vector<std::vector<int>> dependents(all_reduces.size(),
                                           std::vector<int>());
  for (int j = 0; j < all_reduces.size(); ++j) {
    mlir::TF::DTensorAllReduceOp later = all_reduces[j];
    for (int i = 0; i < j; ++i) {
      mlir::TF::DTensorAllReduceOp earlier = all_reduces[i];
      DCHECK(!DependsOn(earlier, later));
      if (earlier->getBlock() != later->getBlock() ||
          DependsOn(later, earlier)) {
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
mlir::LogicalResult CombineAllReduceOpsOfSameTypeAndGroupAssignment(
    mlir::tf_device::ClusterOp cluster,
    const std::vector<mlir::TF::DTensorAllReduceOp>& all_reduces) {
  // Drop within-slice all-reduces.
  std::vector<mlir::TF::DTensorAllReduceOp> cross_slice_all_reduces;
  for (mlir::TF::DTensorAllReduceOp all_reduce : all_reduces) {
    mlir::DenseIntElementsAttr group_assignment_attr;
    if (!matchPattern(all_reduce.group_assignment(),
                      m_Constant(&group_assignment_attr))) {
      return all_reduce.emitOpError("group_assignment should be a constant");
    }
    // LINT.IfChange
    int num_slices = NumClients();
    int slice_size = kTpuDonutSize;
    if (group_assignment_attr.getNumElements() < kTpuDonutSize) {
      DCHECK_EQ(num_slices, 1) << "Num slices expected to be equal to 1.";
      slice_size = group_assignment_attr.getNumElements();
    }
    StatusOr<GroupAssignment> group_assignment = GroupAssignment::FromMLIR(
        group_assignment_attr,
        GroupAssignment::ReplicaToDeviceMap::DefaultReplicaToDeviceMap(
            num_slices, slice_size));
    // LINT.ThenChange(//tensorflow/dtensor/mlir/utils/collective_lowering.cc)
    if (!group_assignment.ok()) {
      return all_reduce.emitOpError(
          llvm::formatv("Failed to create a GroupAssignment due to {0}",
                        group_assignment.status().error_message()));
    }
    // Unit tests have only one slice. Always combine all all-reduces in them.
    if (group_assignment->num_slices() == 1 ||
        !group_assignment->IsWithinSlices()) {
      cross_slice_all_reduces.push_back(all_reduce);
    }
  }

  // A single op has nothing to combine with.
  int num_all_reduces = cross_slice_all_reduces.size();
  if (num_all_reduces <= 1) return mlir::success();

  // Export the all reduces as a DOT graph.
  VLOG(4) << "Visualizing AllReduce dependencies:\n"
          << DrawAllReduceDependencies(cross_slice_all_reduces);

  // Build a reverse adjacency matrix from dependents to requirements.
  std::vector<std::vector<int>> requirements(num_all_reduces,
                                             std::vector<int>());
  for (int i = 0; i < num_all_reduces - 1; ++i) {
    mlir::TF::DTensorAllReduceOp requirement = cross_slice_all_reduces[i];
    for (int j = i + 1; j < num_all_reduces; ++j) {
      mlir::TF::DTensorAllReduceOp dependent = cross_slice_all_reduces[j];
      DCHECK(
          !DependsOn(requirement, dependent));  // guaranteed by program order
      // In this example, all three DTensorAllReduce ops are independent from
      // each other according to MLIR value use-def chains considered by
      // DependsOn. However, moving all three to after the WhileRegion and
      // combine them would break the program.
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
      // DTensorAllReduceOps belong to different blocks. If they do, since they
      // exist in the same ClusterOp, one or both of them must be inside a
      // control flow region block. We treat them as if there is a dependency
      // between them.
      //
      // In the example above, the second DTensorAllReduceOp would "depend on"
      // the first one, and the third on the second. This effectively prevents
      // any two DTensorAllReduce from merging together.
      if (requirement->getBlock() != dependent->getBlock() ||
          DependsOn(dependent, requirement)) {
        requirements[j].push_back(i);
      }
    }
  }

  // Traverse the adjacency matrix layer by layer to find combination groups.
  std::vector<std::vector<mlir::TF::DTensorAllReduceOp>> all_reduce_groups;
  std::set<int> fulfilled;
  while (fulfilled.size() < cross_slice_all_reduces.size()) {
    std::vector<int> fulfilled_this_layer;
    for (int j = 0; j < requirements.size(); ++j) {
      if (fulfilled.count(j) > 0) continue;
      bool requirements_met = true;
      for (int i : requirements[j]) {
        if (fulfilled.count(i) == 0) {
          requirements_met = false;
          break;
        }
      }
      if (requirements_met) {
        fulfilled_this_layer.push_back(j);
      }
    }
    VLOG(4) << "Fulfilled: " << str_util::Join(fulfilled_this_layer, ", ");
    all_reduce_groups.push_back({});
    for (int i : fulfilled_this_layer) {
      fulfilled.insert(i);
      all_reduce_groups.back().push_back(cross_slice_all_reduces[i]);
    }
  }
  VLOG(2) << num_all_reduces << " all-reduce ops in "
          << all_reduce_groups.size() << " groups";

  // Move all-reduces in the same group together and combine them.
  for (auto& all_reduce_group : all_reduce_groups) {
    int num_all_reduces = all_reduce_group.size();
    if (num_all_reduces <= 1) continue;
    mlir::TF::DTensorAllReduceOp final_all_reduce =
        all_reduce_group[num_all_reduces - 1];
    for (int i = num_all_reduces - 2; i >= 0; --i) {
      mlir::TF::DTensorAllReduceOp all_reduce = all_reduce_group[i];
      MoveUsagesAfter(cluster, all_reduce, final_all_reduce);
    }
    for (int i = 0; i < num_all_reduces - 1; ++i) {
      mlir::TF::DTensorAllReduceOp all_reduce = all_reduce_group[i];
      all_reduce->moveBefore(final_all_reduce);
    }
    auto merge_result = MergeAllReduceGroup(all_reduce_group);
    if (merge_result.failed()) return merge_result;
  }

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
  return std::equal(attr_a.begin(), attr_a.end(), attr_b.begin(), attr_b.end());
}

// Combines DTensorAllReduce ops of the same element type into as few groups as
// possible. Only ops with the same group assignment can be combined together.
mlir::LogicalResult CombineAllReduceOpsOfSameType(
    mlir::tf_device::ClusterOp cluster,
    const std::vector<mlir::TF::DTensorAllReduceOp>& all_reduces) {
  // Maintain a list of seen group assignments, sorted by first appearance.
  // Also find and store all-reduces by group assignment. Use the first
  // mlir::Value that contains a certain group assignment to represent all the
  // same group assignments.
  std::vector<mlir::Value> group_assignments;
  llvm::DenseMap<mlir::Value, std::vector<mlir::TF::DTensorAllReduceOp>>
      all_reduces_by_group_assignment;
  for (mlir::TF::DTensorAllReduceOp all_reduce : all_reduces) {
    mlir::Value group_assignment = all_reduce.group_assignment();
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

  // Combine all-reduces of the same group assignment in first-appearance order.
  for (mlir::Value group_assignment : group_assignments) {
    mlir::LogicalResult result =
        CombineAllReduceOpsOfSameTypeAndGroupAssignment(
            cluster, all_reduces_by_group_assignment[group_assignment]);
    if (mlir::failed(result)) return result;
  }

  return mlir::success();
}

struct DTensorAllReduceCombineOptimization
    : public DTensorAllReduceCombineOptimizationBase<
          DTensorAllReduceCombineOptimization> {
  void runOnOperation() override {
    mlir::func::FuncOp function = getOperation();
    function.walk([&](mlir::tf_device::ClusterOp cluster) {
      // Maintain a list of seen element types, sorted by first appearance.
      // Also find and store all-reduces by element type.
      std::vector<mlir::Type> elem_types;
      llvm::DenseMap<mlir::Type, std::vector<mlir::TF::DTensorAllReduceOp>>
          all_reduces_by_elem_type;
      cluster.GetBody().walk([&](mlir::TF::DTensorAllReduceOp all_reduce) {
        mlir::Type elem_type = all_reduce.getType().getElementType();
        if (std::find(elem_types.begin(), elem_types.end(), elem_type) ==
            elem_types.end()) {
          elem_types.push_back(elem_type);
        }
        all_reduces_by_elem_type[elem_type].push_back(all_reduce);
      });

      // Combine all-reduces of the same element type in first-appearance order.
      for (mlir::Type elem_type : elem_types) {
        if (mlir::failed(CombineAllReduceOpsOfSameType(
                cluster, all_reduces_by_elem_type[elem_type]))) {
          return signalPassFailure();
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
