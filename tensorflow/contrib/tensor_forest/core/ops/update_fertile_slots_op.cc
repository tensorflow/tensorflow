// Copyright 2016 The TensorFlow Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
// =============================================================================
// UpdateFertileSlots manages accumulator slots.  It assigns free or newly
// finished accumulator slots to waiting non-fertile nodes and new leaves
// according to their existing split scores (based on node pcws).
#include <unordered_map>
#include <set>

#include "tensorflow/contrib/tensor_forest/core/ops/tree_utils.h"

#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "tensorflow/core/kernels/bounds_check.h"
#include "tensorflow/core/lib/gtl/map_util.h"
#include "tensorflow/core/lib/gtl/top_n.h"


namespace tensorflow {

using shape_inference::Dimension;
using shape_inference::InferenceContext;
using shape_inference::Shape;

using gtl::TopN;
using tensorforest::CheckTensorBounds;
using tensorforest::Initialize;
using tensorforest::WeightedGiniImpurity;

REGISTER_OP("UpdateFertileSlots")
    .Attr("regression: bool = False")
    .Input("finished: int32")
    .Input("non_fertile_leaves: int32")
    .Input("non_fertile_leaf_scores: float")
    .Input("end_of_tree: int32")
    .Input("accumulator_sums: float")
    .Input("node_to_accumulator: int32")
    .Input("stale_leaves: int32")
    .Output("node_map_updates: int32")
    .Output("accumulators_cleared: int32")
    .Output("accumulators_allocated: int32")
    .SetShapeFn([](InferenceContext* c) {
      c->set_output(0, c->Matrix(c->MakeDim(2), InferenceContext::kUnknownDim));
      c->set_output(1, c->Vector(InferenceContext::kUnknownDim));
      c->set_output(2, c->Vector(InferenceContext::kUnknownDim));
      return Status::OK();
    })
    .Doc(R"doc(
Updates accumulator slots to reflect finished or newly fertile nodes.

finished:= A 1-d int32 tensor containing the indices of fertile nodes that
  are ready to decide on a split.
non_fertile_leaves:= A 1-d int32 tensor containing the indices of all the
  currently non-fertile leaves.  If there are free accumulator slots after
  deallocation, UpdateFertileSlots will consider these nodes (plus the ones
  in new_leaves) and potentially turn some of them fertile.
non_fertile_leaf_scores: `non_fertile_leaf_scores[i]` is the splitting score
  of the non-fertile leaf `non_fertile_leaves[i]`.
end_of_tree: The end of tree tensor from the previous training iteration, used
  with the finished input to calculate a list of new leaf indices created by
  GrowTree, which will be considered to become fertile if there are free
  slots.
accumulator_sums: For classification, `accumulator_sums[a][c]` records how
  many training examples have class c and have ended up in the fertile node
  associated with accumulator slot a.  It has the total sum in entry 0 for
  convenience. For regression, it is the same except it contains the sum
  of the input labels that have been seen, and entry 0 contains the number
  of training examples that have been seen.
node_to_accumulator: `node_to_accumulator[i]` is the accumulator slot used by
  fertile node i, or -1 if node i isn't fertile.
stale_leaves:= A 1-d int32 tensor containing the indices of all leaves that
  have stopped accumulating statistics because they are too old.
node_map_updates:= A 2-d int32 tensor describing the changes that need to
  be applied to the node_to_accumulator map.  Intended to be used with
  `tf.scatter_update(node_to_accumulator,
                     node_map_updates[0],
                     node_map_updates[1])`.
accumulators_cleared:= A 1-d int32 tensor containing the indices of all
  the accumulator slots that need to be cleared.
accumulators_allocated:= A 1-d int32 tensor containing the indices of all
  the accumulator slots that need to be allocated.

)doc");

class UpdateFertileSlots : public OpKernel {
 public:
  explicit UpdateFertileSlots(OpKernelConstruction* context)
      : OpKernel(context) {
    OP_REQUIRES_OK(context, context->GetAttr(
      "regression", &regression_));
  }

  void Compute(OpKernelContext* context) override {
    const Tensor& finished = context->input(0);

    const Tensor& non_fertile_leaves =  context->input(1);
    const Tensor& non_fertile_leaf_scores =  context->input(2);
    const Tensor& end_of_tree = context->input(3);

    const Tensor& accumulator_sums = context->input(4);
    const Tensor& node_to_accumulator = context->input(5);
    const Tensor& stale_leaves = context->input(6);

    OP_REQUIRES(context, finished.shape().dims() == 1,
                errors::InvalidArgument(
                    "finished should be one-dimensional"));
    OP_REQUIRES(context, non_fertile_leaves.shape().dims() == 1,
                errors::InvalidArgument(
                    "non_fertile_leaves should be one-dimensional"));
    OP_REQUIRES(context, non_fertile_leaf_scores.shape().dims() == 1,
                errors::InvalidArgument(
                    "non_fertile_leaves_scores should be one-dimensional"));
    OP_REQUIRES(context, end_of_tree.shape().dims() == 1,
                errors::InvalidArgument(
                    "end_of_tree should be one-dimensional"));
    OP_REQUIRES(context, accumulator_sums.shape().dims() == 2,
                errors::InvalidArgument(
                    "accumulator_sums should be two-dimensional"));
     OP_REQUIRES(context, node_to_accumulator.shape().dims() == 1,
                errors::InvalidArgument(
                    "node_to_accumulator should be one-dimensional"));
     OP_REQUIRES(context, stale_leaves.shape().dims() == 1,
                errors::InvalidArgument(
                    "stale_leaves should be one-dimensional"));

    OP_REQUIRES(
        context,
        non_fertile_leaves.shape().dim_size(0) ==
        non_fertile_leaf_scores.shape().dim_size(0),
        errors::InvalidArgument(
            "Number of non fertile leaves should be the same in "
            "non_fertile_leaves and non_fertile_leaf_scores."));

    // Check tensor bounds.
    if (!CheckTensorBounds(context, finished)) return;
    if (!CheckTensorBounds(context, non_fertile_leaves)) return;
    if (!CheckTensorBounds(context, non_fertile_leaf_scores)) return;
    if (!CheckTensorBounds(context, end_of_tree)) return;
    if (!CheckTensorBounds(context, accumulator_sums)) return;
    if (!CheckTensorBounds(context, node_to_accumulator)) return;
    if (!CheckTensorBounds(context, stale_leaves)) return;

    // Read finished accumulators into a set for quick lookup.
    const auto node_map = node_to_accumulator.unaligned_flat<int32>();
    const auto finished_vec = finished.unaligned_flat<int32>();
    const int32 num_finished = static_cast<int32>(finished.shape().dim_size(0));
    std::set<int32> finished_accumulators;
    for (int32 i = 0; i < num_finished; ++i) {
      const int32 node = internal::SubtleMustCopy(finished_vec(i));
      OP_REQUIRES(
          context, FastBoundsCheck(node, node_map.size()),
          errors::InvalidArgument("finished node is outside the valid range"));
      finished_accumulators.insert(node_map(node));
    }
    // Stale accumulators are also finished for the purposes of clearing
    // and re-allocating.
    const auto stale_vec = stale_leaves.unaligned_flat<int32>();
    for (int32 i = 0; i < stale_vec.size(); ++i) {
      const int32 node = internal::SubtleMustCopy(stale_vec(i));
      OP_REQUIRES(
          context, FastBoundsCheck(node, node_map.size()),
          errors::InvalidArgument("stale node is outside the valid range"));
      finished_accumulators.insert(node_map(node));
    }

    // Construct leaf heap to sort leaves to allocate accumulators to.
    const int32 num_nodes =
        static_cast<int32>(node_to_accumulator.shape().dim_size(0));
    const int32 eot = internal::SubtleMustCopy(
        end_of_tree.unaligned_flat<int32>()(0));
    // end-of-tree points to one beyond the last node, so it's allowed to go
    // up to num_nodes inclusive.
    OP_REQUIRES(
        context, FastBoundsCheck(eot, num_nodes + 1),
        errors::InvalidArgument("end-of-tree is outside the valid range"));

    const int32 num_new_leaves = std::min(num_finished * 2, num_nodes - eot);

    LeafHeapType leaf_heap(
        static_cast<int32>(non_fertile_leaves.shape().dim_size(0)) +
        num_new_leaves, OrderBySecondGreater());
    ConstructLeafHeap(
        non_fertile_leaves, non_fertile_leaf_scores, eot, num_new_leaves,
        static_cast<int32>(accumulator_sums.shape().dim_size(1)), &leaf_heap);

    // Allocate leaves.
    std::unique_ptr<HeapValuesType> values(
        leaf_heap.Extract());
    int32 accumulator = -1;  // This will first get incremented to 0.
    int32 num_accumulators_allocated = 0;
    std::unordered_map<int32, int32> accumulators_to_node;
    FindNextAccumulator(accumulator_sums, finished_accumulators, &accumulator);
    int32 i = 0;
    for (; i < values->size(); ++i) {
      const std::pair<int32, float>& node = (*values)[i];
      if (accumulator < 0) {
        VLOG(1) << "No allocators left.";
        break;
      }
      VLOG(1) << "setting node " << node.first << " to accumulator "
              << accumulator;
      ++num_accumulators_allocated;
      accumulators_to_node[accumulator] = node.first;

      FindNextAccumulator(accumulator_sums, finished_accumulators,
                          &accumulator);
    }

    // Construct and fill outputs.
    SetNodeMapUpdates(accumulators_to_node, finished, stale_leaves, context);
    SetAccumulatorsCleared(finished_accumulators,
                           accumulators_to_node, context);
    SetAccumulatorsAllocated(accumulators_to_node, context);
  }

 private:
  struct OrderBySecondGreater {
    bool operator()(const std::pair<int32, float> &left,
                    const std::pair<int32, float> &right) {
        return left.second > right.second;
    }
  };

  typedef TopN<std::pair<int32, float>, OrderBySecondGreater> LeafHeapType;
  typedef std::vector<std::pair<int32, float>> HeapValuesType;

  // Creates an update tensor for node to accumulator map.  Sets finished and
  // stale nodes to -1 (no accumulator assigned) and newly allocated nodes to
  // their accumulator.
  void SetNodeMapUpdates(
      const std::unordered_map<int32, int32>& accumulators_to_node,
      const Tensor& finished, const Tensor& stale, OpKernelContext* context) {
    // Node map updates.
    Tensor* output_node_map = nullptr;
    TensorShape node_map_shape;
    node_map_shape.AddDim(2);
    node_map_shape.AddDim(
        accumulators_to_node.size() +
        static_cast<int32>(stale.shape().dim_size(0) +
                           finished.shape().dim_size(0)));
    OP_REQUIRES_OK(context,
                   context->allocate_output(0, node_map_shape,
                                            &output_node_map));

    auto out_node = output_node_map->tensor<int32, 2>();
    int32 output_slot = 0;

    // Set finished nodes to -1.
    const auto finished_vec = finished.unaligned_flat<int32>();
    for (int32 i = 0; i < finished_vec.size(); ++i) {
      out_node(0, output_slot) = finished_vec(i);
      out_node(1, output_slot)  = -1;
      ++output_slot;
    }
    // Set stale nodes to -1.
    const auto stale_vec = stale.unaligned_flat<int32>();
    for (int32 i = 0; i < stale_vec.size(); ++i) {
      out_node(0, output_slot) = stale_vec(i);
      out_node(1, output_slot)  = -1;
      ++output_slot;
    }

    // Set newly allocated nodes to their allocator.
    for (const auto& node_alloc_pair : accumulators_to_node) {
      out_node(0, output_slot) = node_alloc_pair.second;
      out_node(1, output_slot) = node_alloc_pair.first;
      ++output_slot;
    }
  }

  // Creates output tensor for cleared accumulators. Cleared accumulators are
  // those that were finished but not re-allocated.
  void SetAccumulatorsCleared(
      const std::set<int32>& finished_accumulators,
      const std::unordered_map<int32, int32>& accumulators_to_node,
      OpKernelContext* context) {
    std::set<int32> cleared;
    for (const int32 node : finished_accumulators) {
      if (accumulators_to_node.find(node) == accumulators_to_node.end()) {
        cleared.insert(node);
      }
    }

    Tensor* output_cleared = nullptr;
    TensorShape cleared_shape;
    cleared_shape.AddDim(cleared.size());
    OP_REQUIRES_OK(context,
                   context->allocate_output(1, cleared_shape,
                                            &output_cleared));

    auto out = output_cleared->unaligned_flat<int32>();

    int32 i = 0;
    for (const int32 accumulator : cleared) {
      out(i) = accumulator;
      ++i;
    }
  }

  // Creates output tensor for accumulators that were allocated to now-fertile
  // nodes.
  void SetAccumulatorsAllocated(
      const std::unordered_map<int32, int32>& accumulators_to_node,
      OpKernelContext* context) {
    // Node map updates.
    Tensor* output_allocated = nullptr;
    TensorShape allocated_shape;
    allocated_shape.AddDim(accumulators_to_node.size());
    OP_REQUIRES_OK(context,
                   context->allocate_output(2, allocated_shape,
                                            &output_allocated));

    auto out = output_allocated->unaligned_flat<int32>();
    int32 output_slot = 0;

    // Set newly allocated nodes to their allocator.
    for (const auto& node_alloc_pair : accumulators_to_node) {
      out(output_slot) = node_alloc_pair.first;
      ++output_slot;
    }
  }

  void ConstructLeafHeap(const Tensor& non_fertile_leaves,
                         const Tensor& non_fertile_leaf_scores,
                         int32 end_of_tree, int32 num_new_leaves,
                         int32 num_classes, LeafHeapType* leaf_heap) {
    const auto leaf_vec = non_fertile_leaves.unaligned_flat<int32>();
    const auto leaf_score_vec = non_fertile_leaf_scores.unaligned_flat<float>();

    for (int32 i = 0; i < leaf_vec.size(); i++) {
      const int32 leaf = internal::SubtleMustCopy(leaf_vec(i));
      // Filter out leaves < 0, non_fertile_nodes can contain garbage at
      // startup.
      if (leaf >= 0) {
        leaf_heap->push(std::make_pair(leaf, leaf_score_vec(i)));
      }
    }

    // Add new leaves.
    Eigen::Tensor<float, 1, 1> zeros(num_classes - 1);
    zeros.setZero();
    // No data is 0 variance (for regression), not necessarily so for
    // gini (classification).
    const float zero_score = regression_ ? 0.0 : WeightedGiniImpurity(zeros);
    for (int32 leaf = end_of_tree; leaf < end_of_tree + num_new_leaves;
         leaf++) {
      leaf_heap->push(std::make_pair(leaf, zero_score));
    }
  }

  // Finds the next available or newly-finished accumulator.
  void FindNextAccumulator(Tensor totals_tensor,
                           const std::set<int32>& finished_accumulators,
                           int* current) {
    ++(*current);
    const auto totals = totals_tensor.tensor<float, 2>();
    for (; *current < totals_tensor.shape().dim_size(0); ++(*current)) {
      if (totals(*current, 0) < 0 ||
          finished_accumulators.find(*current) != finished_accumulators.end()) {
        return;
      }
    }
    *current = -1;
  }

  bool regression_;
};

REGISTER_KERNEL_BUILDER(Name("UpdateFertileSlots").Device(DEVICE_CPU),
                        UpdateFertileSlots);

}  // namespace tensorflow
