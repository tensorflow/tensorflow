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
// FinishedNodes returns a 1-D tensor listing the nodes that are finished
// accumulating.
#include "tensorflow/contrib/tensor_forest/core/ops/tree_utils.h"

#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"

namespace tensorflow {

using tensorforest::Sum;

REGISTER_OP("FinishedNodes")
  .Attr("num_split_after_samples: int32")
  .Input("leaves: int32")
  .Input("node_to_accumulator: int32")
  .Input("pcw_total_splits: float")

  .Output("finished: int32")
  .Doc(R"doc(
  Determines which of the given leaf nodes are done accumulating.

  leaves:= A 1-d int32 tensor.  Lists the nodes that are currently leaves.
  node_to_accumulator: If the i-th node is fertile, `node_to_accumulator[i]`
   is it's accumulator slot.  Otherwise, `node_to_accumulator[i]` is -1.
  pcw_total_splits: `pcw_total_splits[a][c]` records how many training examples
   have class c and have ended up in the fertile node associated with
   accumulator slot a.  Between that and `pcw_candidate_splits`, the number of
   examples taking the right branch of a split can be reconstructed.
  finished:= A 1-d int32 tensor. Contains the nodes that have total split
   counts greater or equal to the num_split_after_samples attribute.
)doc");


class FinishedNodes : public OpKernel {
 public:
  explicit FinishedNodes(OpKernelConstruction* context)
      : OpKernel(context) {
    OP_REQUIRES_OK(context, context->GetAttr(
        "num_split_after_samples", &num_split_after_samples_));
  }

  void Compute(OpKernelContext* context) override {
    const Tensor& leaf_tensor = context->input(0);
    const Tensor& node_to_accumulator = context->input(1);
    const Tensor& pcw_total_splits = context->input(2);

    OP_REQUIRES(context, leaf_tensor.shape().dims() == 1,
                errors::InvalidArgument(
                    "leaf_tensor should be one-dimensional"));
    OP_REQUIRES(context, node_to_accumulator.shape().dims() == 1,
                errors::InvalidArgument(
                    "node_to_accumulator should be one-dimensional"));
    OP_REQUIRES(context, pcw_total_splits.shape().dims() == 2,
                errors::InvalidArgument(
                    "pcw_total_splits should be two-dimensional"));

    const auto leaves = leaf_tensor.unaligned_flat<int32>();
    const auto node_map = node_to_accumulator.unaligned_flat<int32>();

    const int32 num_leaves = leaf_tensor.shape().dim_size(0);

    std::vector<int32> finished;
    for (int i = 0; i < num_leaves; i++) {
      const int32 leaf = leaves(i);
      const int32 accumulator = node_map(leaf);
      if (accumulator < 0) {
        continue;
      }

      if (Sum<float>(pcw_total_splits.Slice(accumulator, accumulator + 1)) >=
          num_split_after_samples_) {
        finished.push_back(leaf);
      }
    }

    // Copy to output.
    Tensor* output_finished = nullptr;
    TensorShape finished_shape;
    finished_shape.AddDim(finished.size());
    OP_REQUIRES_OK(context,
                   context->allocate_output(0, finished_shape,
                                            &output_finished));
    auto out_finished = output_finished->unaligned_flat<int32>();

    for (int32 i = 0; i < finished.size(); i++) {
      out_finished(i) = finished[i];
    }
  }

 private:
  int32 num_split_after_samples_;
};

REGISTER_KERNEL_BUILDER(Name("FinishedNodes").Device(DEVICE_CPU),
                        FinishedNodes);

}  // namespace tensorflow

