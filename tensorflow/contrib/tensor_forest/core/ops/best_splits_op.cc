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
// BestSplits returns the index of the best candidate for each finished node.
// This decision is based on the Gini score of the pcw_candidate_split counts,
// and the right-branch-taken counts inferred from pcw_total_splits.
#include <functional>

#include "tensorflow/contrib/tensor_forest/core/ops/tree_utils.h"

#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "tensorflow/core/kernels/bounds_check.h"

namespace tensorflow {

using shape_inference::DimensionHandle;
using shape_inference::InferenceContext;
using shape_inference::ShapeHandle;

using std::placeholders::_1;
using tensorforest::BestFeatureClassification;
using tensorforest::BestFeatureRegression;
using tensorforest::CheckTensorBounds;

REGISTER_OP("BestSplits")
    .Attr("regression: bool = false")
    .Input("finished_nodes: int32")
    .Input("node_to_accumulator: int32")
    .Input("split_sums: float")
    .Input("split_squares: float")
    .Input("accumulator_sums: float")
    .Input("accumulator_sqaures: float")
    .Output("split_indices: int32")
    .SetShapeFn([](InferenceContext* c) {
      ShapeHandle finished_nodes;
      TF_RETURN_IF_ERROR(c->WithRank(c->input(0), 1, &finished_nodes));
      c->set_output(0, c->Vector(c->Dim(finished_nodes, 0)));
      return Status::OK();
    })
    .Doc(R"doc(
  Returns the index of the best split for each finished node.

  For classification, the best split is the split with the lowest weighted
  Gini impurity, as calculated from the statistics in `split_sums` and
  `accumulator_sums`. For regression we use the lowest variance, incoporating
  the *_squares as well.

  finished_nodes:= A 1-d int32 tensor containing the indices of finished nodes.
  node_to_accumulator: `node_to_accumulator[i]` is the accumulator slot used by
    fertile node i, or -1 if node i isn't fertile.
  split_sums:= a 3-d tensor where `split_sums[a][s]` summarizes the
    training labels for examples that fall into the fertile node associated with
    accumulator slot s and have then taken the *left* branch of candidate split
    s.  For a classification problem, `split_sums[a][s][c]` is the count of such
    examples with class c and for regression problems, `split_sums[a][s]` is the
    sum of the regression labels for such examples.
  split_squares: Same as split_sums, but it contains the sum of the
    squares of the regression labels.  Only used for regression.  For
    classification problems, pass a dummy tensor into this.
  accumulator_sums:= a 2-d tensor where `accumulator_sums[a]` summarizes the
    training labels for examples that fall into the fertile node associated with
    accumulator slot s.  For a classification problem, `accumulator_sums[a][c]`
    is the count of such examples with class c and for regression problems,
    `accumulator_sums[a]` is the sum of the regression labels for such examples.
  accumulator_squares: Same as accumulator_sums, but it contains the sum of the
    squares of the regression labels.  Only used for regression.  For
    classification problems, pass a dummy tensor into this.
  split_indices: `split_indices[i]` contains the index of the split to use for
    `finished_nodes[i]`.
)doc");

class BestSplits : public OpKernel {
 public:
  explicit BestSplits(OpKernelConstruction* context) : OpKernel(context) {
    OP_REQUIRES_OK(context, context->GetAttr(
        "regression", &regression_));
  }

  void Compute(OpKernelContext* context) override {
    const Tensor& finished = context->input(0);
    const Tensor& node_to_accumulator = context->input(1);
    const Tensor& split_sums = context->input(2);
    const Tensor& split_squares = context->input(3);
    const Tensor& accumulator_sums = context->input(4);
    const Tensor& accumulator_squares = context->input(5);

    OP_REQUIRES(context, finished.shape().dims() == 1,
                errors::InvalidArgument(
                    "finished should be one-dimensional"));
    OP_REQUIRES(context, node_to_accumulator.shape().dims() == 1,
                errors::InvalidArgument(
                    "node_to_accumulator should be one-dimensional"));

    OP_REQUIRES(context, split_sums.shape().dims() == 3,
                errors::InvalidArgument(
                    "split_sums should be three-dimensional"));
    OP_REQUIRES(context, accumulator_sums.shape().dims() == 2,
                errors::InvalidArgument(
                    "accumulator_sums should be two-dimensional"));

    if (regression_) {
      OP_REQUIRES(context,
                  split_sums.shape() == split_squares.shape(),
                  errors::InvalidArgument(
                      "split_sums and split_squares should "
                      "be the same shape."));
      OP_REQUIRES(context,
                  accumulator_sums.shape() == accumulator_squares.shape(),
                  errors::InvalidArgument(
                      "accumulator_sums and accumulator_squares should "
                      "be the same shape."));
    }

    OP_REQUIRES(
        context,
        accumulator_sums.shape().dim_size(0) ==
        split_sums.shape().dim_size(0),
        errors::InvalidArgument(
            "Number of accumulators should be the same in split_sums "
            "and accumulator_sums."));

    // Check tensor bounds.
    if (!CheckTensorBounds(context, finished)) return;
    if (!CheckTensorBounds(context, node_to_accumulator)) return;
    if (!CheckTensorBounds(context, split_sums)) return;
    if (!CheckTensorBounds(context, split_squares)) return;
    if (!CheckTensorBounds(context, accumulator_sums)) return;
    if (!CheckTensorBounds(context, accumulator_squares)) return;

    Tensor* output_splits = nullptr;
    OP_REQUIRES_OK(context,
                   context->allocate_output(0, finished.shape(),
                                            &output_splits));
    auto best_splits = output_splits->unaligned_flat<int32>();

    const auto finished_vec = finished.unaligned_flat<int32>();
    const auto node_map = node_to_accumulator.unaligned_flat<int32>();

    const int32 num_finished = static_cast<int32>(finished.shape().dim_size(0));

    std::function<int32(int32)> best_feature_func =
        std::bind(BestFeatureClassification, accumulator_sums, split_sums, _1);
    if (regression_) {
       best_feature_func = std::bind(
           BestFeatureRegression, accumulator_sums, accumulator_squares,
           split_sums, split_squares, _1);
    }

    for (int32 i = 0; i < num_finished; i++) {
      const int32 node = internal::SubtleMustCopy(finished_vec(i));
      OP_REQUIRES(
          context, FastBoundsCheck(node, node_map.size()),
          errors::InvalidArgument("finished node is outside the valid range"));

      const int32 accumulator = internal::SubtleMustCopy(node_map(node));
      if (accumulator < 0) {
        LOG(ERROR) << "Something has gone wrong, we got a finished node that "
                   << "doesn't have an accumulator allocated to it.";
        continue;
      }

      best_splits(i) = best_feature_func(accumulator);
    }
  }

 private:
  bool regression_;
};

REGISTER_KERNEL_BUILDER(Name("BestSplits").Device(DEVICE_CPU), BestSplits);

}  // namespace tensorflow
