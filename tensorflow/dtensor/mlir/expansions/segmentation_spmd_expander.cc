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

#include "tensorflow/dtensor/mlir/expansions/segmentation_spmd_expander.h"

#include "llvm/Support/FormatVariadic.h"
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_ops_n_z.h"
#include "tensorflow/dtensor/cc/dstatus.h"
#include "tensorflow/dtensor/cc/tensor_layout.h"
#include "tensorflow/dtensor/mlir/collectives.h"
#include "tensorflow/dtensor/mlir/layout_parsing.h"
#include "tensorflow/dtensor/mlir/shape_utils.h"
#include "tensorflow/dtensor/mlir/spmd_expander_common.h"
#include "tensorflow/dtensor/mlir/value_utils.h"

namespace tensorflow {
namespace dtensor {

// We always forward replicated layout to operands/output of
// UnsortedSegmentedSum op as SPMD logic sharded UnsortedSegmentedSum op is not
// implemented yet.
// TODO(b/171079751): Implement layout propagation for non-trivial layouts
StatusOr<llvm::DenseMap<int, Layout>>
UnsortedSegmentSumSPMDExpander::ComputeLayoutForward(
    mlir::Operation* op, const llvm::DenseMap<int, Layout>& input_layouts) {
  TF_ASSIGN_OR_RETURN(const auto mesh, ExtractDeviceMeshEnclosingCluster(op));
  auto unsorted_segmented_sum = llvm::cast<mlir::TF::UnsortedSegmentSumOp>(op);
  return llvm::DenseMap<int, Layout>(
      {{0, Layout::ReplicatedOnMesh(
               mesh, /*rank=*/ValueRank(unsorted_segmented_sum.output()))}});
}

StatusOr<llvm::DenseMap<int, Layout>>
UnsortedSegmentSumSPMDExpander::ComputeLayoutBackward(
    mlir::Operation* op, const llvm::DenseMap<int, Layout>& output_layouts) {
  TF_ASSIGN_OR_RETURN(const auto mesh, ExtractDeviceMeshEnclosingCluster(op));
  auto unsorted_segmented_sum = llvm::cast<mlir::TF::UnsortedSegmentSumOp>(op);
  return llvm::DenseMap<int, Layout>(
      {{0, Layout::ReplicatedOnMesh(
               mesh, /*rank=*/ValueRank(unsorted_segmented_sum.data()))},
       {1,
        Layout::ReplicatedOnMesh(mesh, /*rank=*/ValueRank(op->getOperand(1)))},
       {2, Layout::ReplicatedOnMesh(mesh, /*rank=*/0)}});
}

StatusOr<mlir::Operation*> UnsortedSegmentSumSPMDExpander::ExpandOp(
    mlir::Operation* op) {
  // The algorithm is simple
  //
  // 1. Up to rank of the segment_ids, if the data or ids are sharded, perform
  //    all-concat, respectively.
  // 2. We do not care the sharding dims of data[rank(ids):] and just leave as
  //    is
  // 3. output.layout[0] is expected to be replicated due to the steps above.
  //    otherwise, perform a slicing.
  // 4. output.layout[1:] is expected to be same as data.layout[rank(ids):] as
  //    untouched. otherwise, perform all-concat or slicing.
  //
  // For item 3 and 4, we perform a single all-concat Op.
  //
  // Alternative to the steps 1 and 2 above could be
  //   a. all-concat data
  //   b. local unsorted seg sum followed by a all reduce with some masks.
  //
  // Alternative to the step 4 above is merging it with step 1 (upon the dim is
  // compatible).

  auto sum_op = mlir::cast<mlir::TF::UnsortedSegmentSumOp>(op);
  auto data = sum_op.data();
  auto segment_ids = sum_op.segment_ids();

  TF_ASSIGN_OR_RETURN(auto data_layout, ExtractLayoutFromOperand(data));
  TF_ASSIGN_OR_RETURN(auto segment_ids_layout,
                      ExtractLayoutFromOperand(segment_ids));

  const auto data_rank = ValueRank(data);
  const auto segment_ids_rank = ValueRank(segment_ids);

  // Prepares the resulting output layout. Fills the default unsharded dim for
  // the first axis (dim size is num_segments).
  LayoutProto result_output_layout;
  *result_output_layout.mutable_mesh_config() = data_layout->mesh().ToProto();
  result_output_layout.add_sharding_specs()->set_sharding_spec(
      Layout::kUnshardedDim);

  // Prepares the replicated target data output (up to segment_ids_rank).
  LayoutProto tgt_data_layout;
  *tgt_data_layout.mutable_mesh_config() = data_layout->mesh().ToProto();

  bool need_data_all_concat = false;
  for (int i = 0; i < data_rank; i++) {
    if (i < segment_ids_rank) {
      tgt_data_layout.add_sharding_specs()->set_sharding_spec(
          Layout::kUnshardedDim);
      if (data_layout->sharding_spec(i) != Layout::kUnshardedDim) {
        need_data_all_concat = true;
      }
    } else {
      tgt_data_layout.add_sharding_specs()->set_sharding_spec(
          data_layout->sharding_spec(i));
      result_output_layout.add_sharding_specs()->set_sharding_spec(
          data_layout->sharding_spec(i));
    }
  }

  mlir::OpBuilder builder(op);
  if (need_data_all_concat) {
    TF_ASSIGN_OR_RETURN(
        auto data_concat,
        EmitAllGather(builder, data, *data_layout,
                      Layout::FromProto(tgt_data_layout).ValueOrDie()));
    data = data_concat;
  }

  // Ensure segment IDs are fully replicated.
  if (!segment_ids_layout->IsFullyReplicated()) {
    TF_ASSIGN_OR_RETURN(
        auto segment_ids_concat,
        EmitAllGather(builder, segment_ids, *segment_ids_layout,
                      Layout::ReplicatedOnMesh(segment_ids_layout->mesh(),
                                               segment_ids_layout->rank())));
    segment_ids = segment_ids_concat;
  }

  auto new_sum_op = builder.create<mlir::TF::UnsortedSegmentSumOp>(
      op->getLoc(), sum_op.output().getType(), data, segment_ids,
      sum_op.num_segments());

  InferSPMDExpandedLocalShape(new_sum_op);

  // Transform the result to the expected output_layout, if necessary.
  TF_ASSIGN_OR_RETURN(auto output_layout, ExtractSingleLayoutFromOp(op));
  TF_ASSIGN_OR_RETURN(
      auto final_output,
      EmitRelayout(new_sum_op.getResult(),
                   Layout::FromProto(result_output_layout).ValueOrDie(),
                   *output_layout));
  op->getResult(0).replaceAllUsesWith(final_output);
  op->erase();

  return final_output.getDefiningOp();
}

}  // namespace dtensor
}  // namespace tensorflow
