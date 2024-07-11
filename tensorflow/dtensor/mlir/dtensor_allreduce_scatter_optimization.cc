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

#include "absl/container/flat_hash_set.h"
#include "absl/log/log.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"  // from @llvm-project
#include "mlir/IR/Builders.h"  // from @llvm-project
#include "mlir/IR/BuiltinAttributes.h"  // from @llvm-project
#include "mlir/IR/Matchers.h"  // from @llvm-project
#include "mlir/IR/Operation.h"  // from @llvm-project
#include "mlir/IR/UseDefLists.h"  // from @llvm-project
#include "mlir/IR/Value.h"  // from @llvm-project
#include "mlir/IR/Visitors.h"  // from @llvm-project
#include "mlir/Pass/Pass.h"  // from @llvm-project
#include "mlir/Support/LLVM.h"  // from @llvm-project
#include "mlir/Support/LogicalResult.h"  // from @llvm-project
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_ops.h"
#include "tensorflow/core/platform/types.h"
#include "tensorflow/dtensor/cc/tensor_layout.h"
#include "tensorflow/dtensor/mlir/collectives_common.h"
#include "tensorflow/dtensor/mlir/ir/tf_dtensor.h"
#include "tensorflow/dtensor/mlir/layout_parsing.h"

namespace tensorflow {
namespace dtensor {

namespace {
#define GEN_PASS_DEF_DTENSORALLREDUCESCATTEROPTIMIZATION
#include "tensorflow/dtensor/mlir/dtensor_passes.h.inc"

// Returns true if both group assignments are constant and equal.
bool same_group_assignments(mlir::DenseIntElementsAttr attr_a,
                            mlir::DenseIntElementsAttr attr_b) {
  if (attr_a.getType().getShape() != attr_b.getType().getShape()) {
    return false;
  }
  return std::equal(attr_a.begin(), attr_a.end(), attr_b.begin(), attr_b.end());
}

mlir::DenseIntElementsAttr GetScatterGroupAssignment(
    mlir::TF::DTensorAllScatterOp all_scatter, int scatter_dim) {
  const Layout original_layout = all_scatter.getInputLayout();
  const Layout desired_layout = all_scatter.getOutputLayout();
  absl::flat_hash_set<std::string> scattered_dims;
  scattered_dims.insert(desired_layout.sharding_spec(scatter_dim));

  auto partitions =
      GetAllReducePartitionsFromReducedDims(original_layout, scattered_dims)
          .value();
  const int32 num_partitions = partitions.size();

  // Construct a flattened list of scatter partitions.
  std::vector<int32> partitions_flat;
  for (auto& p : partitions) {
    partitions_flat.insert(partitions_flat.end(), p.second.begin(),
                           p.second.end());
  }

  int32 partition_size = partitions.begin()->second.size();
  mlir::OpBuilder builder(all_scatter);
  auto group_shaped_type = mlir::RankedTensorType::get(
      {num_partitions, partition_size},
      mlir::IntegerType::get(builder.getContext(), 32));

  return mlir::DenseIntElementsAttr::get(group_shaped_type, partitions_flat);
}

mlir::LogicalResult ApplyOptimization(mlir::func::FuncOp function) {
  std::vector<mlir::Operation*> ops_to_delete;
  function.walk([&](mlir::TF::DTensorAllReduceOp all_reduce) {
    if (all_reduce->hasOneUse()) {
      if (auto all_scatter = mlir::dyn_cast<mlir::TF::DTensorAllScatterOp>(
              *all_reduce->getUsers().begin())) {
        VLOG(2) << "Found potential AllReduce+AllScatter to fuse.";
        if (VLOG_IS_ON(2)) all_reduce.dump();
        if (VLOG_IS_ON(2)) all_scatter.dump();

        const Layout original_layout = all_scatter.getInputLayout();
        const Layout desired_layout = all_scatter.getOutputLayout();

        // Find all potential scatter dimensions.
        std::vector<int> scatter_dims;
        for (int i = 0; i < original_layout.rank(); ++i) {
          if (original_layout.sharding_spec(i) !=
              desired_layout.sharding_spec(i)) {
            scatter_dims.push_back(i);
          }
        }

        if (scatter_dims.empty()) return mlir::WalkResult::advance();
        if (scatter_dims.size() > 1) {
          VLOG(2) << "Multiple dimensions are scatter.  This is unsupported "
                     "for AllReduce+Scatter fusion.";
          return mlir::WalkResult::advance();
        }

        int scatter_dim = scatter_dims[0];
        VLOG(2) << "Scatter_dim: " << scatter_dim;

        // Check that the all-reduce and all-scatter group assignments are the
        // same.
        mlir::DenseIntElementsAttr all_reduce_group_assignment_attr;
        if (!matchPattern(all_reduce.getGroupAssignment(),
                          m_Constant(&all_reduce_group_assignment_attr))) {
          all_reduce.emitOpError("group_assignment should be a constant");
          return mlir::WalkResult::interrupt();
        }

        mlir::DenseIntElementsAttr all_scatter_group_assignment_attr =
            GetScatterGroupAssignment(all_scatter, scatter_dim);

        VLOG(2) << "All scatter group assignment: ";
        if (VLOG_IS_ON(2)) all_scatter_group_assignment_attr.dump();

        bool same_group =
            same_group_assignments(all_reduce_group_assignment_attr,
                                   all_scatter_group_assignment_attr);

        if (!same_group) return mlir::WalkResult::advance();
        VLOG(2) << "Fuse reduce scatter with scatter_dim: " << scatter_dim;

        mlir::OpBuilder builder(all_reduce);
        auto scatter_dim_const_op = builder.create<mlir::TF::ConstOp>(
            all_reduce.getLoc(),
            mlir::DenseIntElementsAttr::get(
                mlir::RankedTensorType::get({}, builder.getI32Type()),
                {scatter_dim}));

        auto reduce_scatter = builder.create<mlir::TF::DTensorReduceScatterOp>(
            all_reduce.getLoc(), all_scatter->getResultTypes(),
            all_reduce.getOperand(0), all_reduce.getGroupAssignment(),
            scatter_dim_const_op, all_reduce.getReduceOp(),
            all_reduce.getDeviceType());
        SetSingleLayoutOnOp(reduce_scatter, desired_layout);

        all_scatter->replaceAllUsesWith(reduce_scatter);

        ops_to_delete.push_back(all_scatter);
        ops_to_delete.push_back(all_reduce);
      }
    }
    return mlir::WalkResult::advance();
  });

  for (mlir::Operation* op : ops_to_delete) {
    op->erase();
  }
  return mlir::success();
}

// MLIR pass that combines AllReduce and AllScatter to ReduceScatter.
struct DTensorAllReduceScatterOptimization
    : public impl::DTensorAllReduceScatterOptimizationBase<
          DTensorAllReduceScatterOptimization> {
  void runOnOperation() override {
    mlir::func::FuncOp function = getOperation();

    if (mlir::failed(ApplyOptimization(function))) return signalPassFailure();
  }
};

}  // namespace

std::unique_ptr<mlir::OperationPass<mlir::func::FuncOp>>
CreateDTensorAllReduceScatterOptimization() {
  return std::make_unique<DTensorAllReduceScatterOptimization>();
}

}  // namespace dtensor
}  // namespace tensorflow
