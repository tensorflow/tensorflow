// Copyright 2016 Google Inc. All Rights Reserved.
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
// BestSplits returns the index of the best candidate for each finished node.
// This decision is based on the Gini score of the pcw_candidate_split counts,
// and the right-branch-taken counts inferred from pcw_total_splits.
#include "tensorflow/contrib/tensor_forest/core/ops/tree_utils.h"

#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"


namespace tensorflow {

using tensorforest::BestFeature;


REGISTER_OP("BestSplits")
  .Input("finished_nodes: int32")
  .Input("node_to_accumulator: int32")
  .Input("pcw_candidate_splits: float")
  .Input("pcw_total_splits: float")
  .Output("split_indices: int32")
  .Doc(R"doc(
  Returns the index of the best split for each finished node.

  The best split is the split with the lowest weighted Gini impurity,
  as calculated from the statistics in `pcw_candidate_splits` and
  `pcw_total_splits`.

  finished_nodes:= A 1-d int32 tensor containing the indices of finished nodes.
  node_to_accumulator: `node_to_accumulator[i]` is the accumulator slot used by
    fertile node i, or -1 if node i isn't fertile.
  pcw_candidate_splits: `pcw_candidate_splits[a][s][c]` records how many
    training examples have class c and have ended up in the fertile node
    associated with accumulator slot a and then taken the *left* branch of
    candidate split s.
  pcw_total_splits: `pcw_total_splits[a][c]` records how many training examples
    have class c and have ended up in the fertile node associated with
    accumulator slot a.  Between that and `pcw_candidate_splits`, the number of
    examples taking the right branch of a split can be reconstructed.
  split_indices: `split_indices[i]` contains the index of the split to use for
    `finished_nodes[i]`.
)doc");


class BestSplits : public OpKernel {
 public:
  explicit BestSplits(OpKernelConstruction* context) : OpKernel(context) {}

  void Compute(OpKernelContext* context) override {
    const Tensor& finished = context->input(0);
    const Tensor& node_to_accumulator = context->input(1);
    const Tensor& pcw_candidate_splits = context->input(2);
    const Tensor& pcw_total_splits = context->input(3);

    OP_REQUIRES(context, finished.shape().dims() == 1,
                errors::InvalidArgument(
                    "finished should be one-dimensional"));
    OP_REQUIRES(context, node_to_accumulator.shape().dims() == 1,
                errors::InvalidArgument(
                    "node_to_accumulator should be one-dimensional"));

    OP_REQUIRES(context, pcw_candidate_splits.shape().dims() == 3,
                errors::InvalidArgument(
                    "pcw_candidate_splits should be three-dimensional"));
    OP_REQUIRES(context, pcw_total_splits.shape().dims() == 2,
                errors::InvalidArgument(
                    "pcw_total_splits should be two-dimensional"));

    OP_REQUIRES(
        context,
        pcw_candidate_splits.shape().dim_size(0) ==
        pcw_total_splits.shape().dim_size(0),
        errors::InvalidArgument(
            "Number of accumulators should be the same in pcw_candidate_splits "
            "and pcw_total_splits."));

    Tensor* output_splits = nullptr;
    OP_REQUIRES_OK(context,
                   context->allocate_output(0, finished.shape(),
                                            &output_splits));
    auto best_splits = output_splits->unaligned_flat<int32>();

    const auto finished_vec = finished.unaligned_flat<int32>();
    const auto node_map = node_to_accumulator.unaligned_flat<int32>();

    const int32 num_finished = finished.shape().dim_size(0);

    for (int i = 0; i < num_finished; i++) {
      const int32 node = finished_vec(i);
      const int32 accumulator = node_map(node);
      if (accumulator < 0) {
        LOG(ERROR) << "Something has gone wrong, we got a finished node that "
                   << "doesn't have an accumulator allocated to it.";
        continue;
      }
      best_splits(i) = BestFeature(pcw_total_splits,
                                   pcw_candidate_splits, accumulator);
    }
  }
};

REGISTER_KERNEL_BUILDER(Name("BestSplits").Device(DEVICE_CPU), BestSplits);

}  // namespace tensorflow
