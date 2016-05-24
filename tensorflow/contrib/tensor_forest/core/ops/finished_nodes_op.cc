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
#include "tensorflow/core/kernels/bounds_check.h"

namespace tensorflow {

using tensorforest::CheckTensorBounds;
using tensorforest::Sum;

REGISTER_OP("FinishedNodes")
  .Attr("num_split_after_samples: int32")
  .Input("leaves: int32")
  .Input("node_to_accumulator: int32")
  .Input("accumulator_sums: float")

  .Output("finished: int32")
  .Doc(R"doc(
  Determines which of the given leaf nodes are done accumulating.

  leaves:= A 1-d int32 tensor.  Lists the nodes that are currently leaves.
  node_to_accumulator: If the i-th node is fertile, `node_to_accumulator[i]`
    is it's accumulator slot.  Otherwise, `node_to_accumulator[i]` is -1.
  accumulator_sums: For classification, `accumulator_sums[a][c]` records how many
    training examples have class c and have ended up in the fertile node
    associated with accumulator slot a.  It has the total sum in entry 0 for
    convenience. For regression, it is the same except it contains the sum
    of the input labels that have been seen, and entry 0 contains the number
    of training examples that have been seen.
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
    const Tensor& accumulator_sums = context->input(2);

    OP_REQUIRES(context, leaf_tensor.shape().dims() == 1,
                errors::InvalidArgument(
                    "leaf_tensor should be one-dimensional"));
    OP_REQUIRES(context, node_to_accumulator.shape().dims() == 1,
                errors::InvalidArgument(
                    "node_to_accumulator should be one-dimensional"));
    OP_REQUIRES(context, accumulator_sums.shape().dims() == 2,
                errors::InvalidArgument(
                    "accumulator_sums should be two-dimensional"));

    // Check tensor bounds.
    if (!CheckTensorBounds(context, leaf_tensor)) return;
    if (!CheckTensorBounds(context, node_to_accumulator)) return;
    if (!CheckTensorBounds(context, accumulator_sums)) return;

    const auto leaves = leaf_tensor.unaligned_flat<int32>();
    const auto node_map = node_to_accumulator.unaligned_flat<int32>();
    const auto sums = accumulator_sums.tensor<float, 2>();

    const int32 num_leaves = static_cast<int32>(
        leaf_tensor.shape().dim_size(0));
    const int32 num_accumulators = static_cast<int32>(
        accumulator_sums.shape().dim_size(0));

    std::vector<int32> finished;
    for (int32 i = 0; i < num_leaves; i++) {
      const int32 leaf = internal::SubtleMustCopy(leaves(i));
      OP_REQUIRES(context, FastBoundsCheck(leaf, node_map.size()),
                  errors::InvalidArgument("leaf not in valid range."))
      const int32 accumulator = internal::SubtleMustCopy(node_map(leaf));
      if (accumulator < 0) {
        continue;
      }

      OP_REQUIRES(context, FastBoundsCheck(accumulator, num_accumulators),
                  errors::InvalidArgument("accumulator not in valid range."))

      // The first column holds the number of samples seen.
      // For classification, this should be the sum of the other columns.
      if (sums(accumulator, 0) >= num_split_after_samples_) {
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

