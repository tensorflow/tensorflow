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

#include <string>

#include "absl/container/flat_hash_set.h"
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

StatusOr<llvm::DenseMap<int, Layout>>
UnsortedSegmentSumSPMDExpander::ComputeLayoutForward(
    mlir::Operation* op, const llvm::DenseMap<int, Layout>& input_layouts) {
  TF_ASSIGN_OR_RETURN(const auto mesh, ExtractDeviceMeshEnclosingCluster(op));
  auto unsorted_segmented_sum = llvm::cast<mlir::TF::UnsortedSegmentSumOp>(op);
  const int output_rank = ValueRank(unsorted_segmented_sum.getOutput());
  if (input_layouts.find(0) != input_layouts.end()) {
    // If the data layout exists, we can use it to forward propagate a layout
    // to the output.
    const int segment_ids_rank =
        ValueRank(unsorted_segmented_sum.getSegmentIds());

    TF_ASSIGN_OR_RETURN(
        Layout input_layout_truncated,
        input_layouts.lookup(0).Truncate(segment_ids_rank, /*end=*/true));
    return llvm::DenseMap<int, Layout>(
        {{0, input_layout_truncated.LeftPad(output_rank)}});
  }

  // When we don't have a data layout we can only output a replicated layout.
  return llvm::DenseMap<int, Layout>(
      {{0, Layout::ReplicatedOnMesh(mesh, output_rank)}});
}

StatusOr<llvm::DenseMap<int, Layout>>
UnsortedSegmentSumSPMDExpander::ComputeLayoutBackward(
    mlir::Operation* op, const llvm::DenseMap<int, Layout>& output_layouts) {
  TF_ASSIGN_OR_RETURN(const auto mesh, ExtractDeviceMeshEnclosingCluster(op));
  auto unsorted_segmented_sum = llvm::cast<mlir::TF::UnsortedSegmentSumOp>(op);

  Layout segment_ids_layout =
      Layout::ReplicatedOnMesh(mesh, ValueRank(op->getOperand(1)));
  Layout num_segments_layout = Layout::ReplicatedOnMesh(mesh, /*rank=*/0);
  if (!output_layouts.empty()) {
    // If we have an output layout, we can send it backwards to the last few
    // dimension
    const int data_rank = ValueRank(unsorted_segmented_sum.getData());
    TF_ASSIGN_OR_RETURN(Layout output_layout_truncated,
                        output_layouts.lookup(0).Truncate(1, /*end=*/true));
    return llvm::DenseMap<int, Layout>(
        {{0, output_layout_truncated.LeftPad(data_rank)},
         {1, segment_ids_layout},
         {2, num_segments_layout}});
  }
  return llvm::DenseMap<int, Layout>(
      {{0, Layout::ReplicatedOnMesh(
               mesh, /*rank=*/ValueRank(unsorted_segmented_sum.getData()))},
       {1, segment_ids_layout},
       {2, num_segments_layout}});
}

StatusOr<mlir::Operation*> UnsortedSegmentSumSPMDExpander::ExpandOp(
    mlir::Operation* op) {
  // The algorithm is simple
  //
  // 1. Relayout segment_ids to match the layout of data[:rank(segment_ids)].
  //    An improved version of this would merge the layouts of segment_ids and
  //    data into a common layout (e.g. data with layout [x,*,*,*] and
  //    segment_ids with layout [*,y] would result in [x,y,*,*] and [x,y].
  // 2. Emit a local UnsortedSegmentSum.
  // 3. Emit an AllReduce on the output.
  // 4. Emit a Relayout to the output layout.

  auto sum_op = mlir::cast<mlir::TF::UnsortedSegmentSumOp>(op);
  mlir::Value data = sum_op.getData();
  mlir::Value segment_ids = sum_op.getSegmentIds();

  TF_ASSIGN_OR_RETURN(Layout data_layout,
                      ExtractRequiredLayoutFromOperand(data));
  TF_ASSIGN_OR_RETURN(Layout segment_ids_layout,
                      ExtractRequiredLayoutFromOperand(segment_ids));

  const int data_rank = data_layout.rank();
  const int segment_ids_rank = segment_ids_layout.rank();

  TF_ASSIGN_OR_RETURN(Layout new_segment_ids_layout,
                      data_layout.Truncate(segment_ids_rank));

  absl::flat_hash_set<std::string> reduce_dimensions;
  for (int i = 0; i < segment_ids_rank; i++)
    if (new_segment_ids_layout.sharding_spec(i) != Layout::kUnshardedDim)
      reduce_dimensions.insert(new_segment_ids_layout.sharding_spec(i));

  TF_ASSIGN_OR_RETURN(
      mlir::Value new_segment_ids,
      EmitRelayout(segment_ids, segment_ids_layout, new_segment_ids_layout));

  mlir::OpBuilder builder(op);
  mlir::Operation* new_sum_op = builder.create<mlir::TF::UnsortedSegmentSumOp>(
      op->getLoc(), sum_op.getOutput().getType(), data, new_segment_ids,
      sum_op.getNumSegments());

  InferSPMDExpandedLocalShape(new_sum_op);

  TF_ASSIGN_OR_RETURN(Layout data_layout_truncated,
                      data_layout.Truncate(segment_ids_rank, /*end=*/true));
  Layout result_output_layout = data_layout_truncated.LeftPad(
      data_rank - segment_ids_rank + 1);  // This is output rank.

  TF_ASSIGN_OR_RETURN(
      new_sum_op, EmitAllReduce(builder, result_output_layout,
                                reduce_dimensions, new_sum_op, kReduceOpAdd));

  // Transform the result to the expected output_layout, if necessary.
  TF_ASSIGN_OR_RETURN(Layout output_layout,
                      ExtractRequiredSingleLayoutFromOp(op));
  TF_ASSIGN_OR_RETURN(mlir::Value final_output,
                      EmitRelayout(new_sum_op->getResult(0),
                                   result_output_layout, output_layout));
  op->getResult(0).replaceAllUsesWith(final_output);
  op->erase();

  return final_output.getDefiningOp();
}

}  // namespace dtensor
}  // namespace tensorflow
