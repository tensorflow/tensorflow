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
// TreePredictions returns the per-class probabilities for each input by
// evaluating the given tree.
#include <algorithm>

#include "tensorflow/contrib/tensor_forest/core/ops/tree_utils.h"

#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"

namespace tensorflow {

using tensorforest::CHILDREN_INDEX;
using tensorforest::FEATURE_INDEX;
using tensorforest::LEAF_NODE;
using tensorforest::FREE_NODE;

using tensorforest::DecideNode;
using tensorforest::Sum;

REGISTER_OP("TreePredictions")
  .Attr("valid_leaf_threshold: float")
  .Input("input_data: float")
  .Input("tree: int32")
  .Input("tree_thresholds: float")
  .Input("node_per_class_weights: float")

  .Output("predictions: float")
  .Doc(R"doc(
  Returns the per-class probabilities for each input.

  input_data: The training batch's features as a 2-d tensor; `input_data[i][j]`
   gives the j-th feature of the i-th input.
  tree:= A 2-d int32 tensor.  `tree[i][0]` gives the index of the left child
   of the i-th node, `tree[i][0] + 1` gives the index of the right child of
   the i-th node, and `tree[i][1]` gives the index of the feature used to
   split the i-th node.
  tree_thresholds: `tree_thresholds[i]` is the value used to split the i-th
   node.
  node_per_class_weights: `node_per_class_weights[n][c]` records how many
   training examples have class c and have ended up in node n.
  predictions: `predictions[i][j]` is the probability that input i is class j.
  valid_leaf_threshold: Minimum number of samples that have arrived to a leaf
    to be considered a valid leaf, otherwise use the parent.
)doc");


class TreePredictions : public OpKernel {
 public:
  explicit TreePredictions(OpKernelConstruction* context)
      : OpKernel(context) {
    OP_REQUIRES_OK(context, context->GetAttr(
      "valid_leaf_threshold", &valid_leaf_threshold_));
  }

  void Compute(OpKernelContext* context) override {
    const Tensor& input_data = context->input(0);

    const Tensor& tree_tensor = context->input(1);
    const Tensor& tree_thresholds = context->input(2);
    const Tensor& node_per_class_weights = context->input(3);

    OP_REQUIRES(context, tree_tensor.shape().dims() == 2,
                errors::InvalidArgument(
                    "tree should be two-dimensional"));
    OP_REQUIRES(context, tree_thresholds.shape().dims() == 1,
                errors::InvalidArgument(
                    "tree_threhsolds should be one-dimensional"));
    OP_REQUIRES(context, node_per_class_weights.shape().dims() == 2,
                errors::InvalidArgument(
                    "node_pcw should be two-dimensional"));

    if (input_data.shape().dim_size(0) > 0) {
      OP_REQUIRES(context, input_data.shape().dims() == 2,
                  errors::InvalidArgument(
                      "input_data should be two-dimensional"));
    }
    OP_REQUIRES(
        context,
        tree_tensor.shape().dim_size(0) ==
        tree_thresholds.shape().dim_size(0) &&
        tree_tensor.shape().dim_size(0) ==
        node_per_class_weights.shape().dim_size(0),
        errors::InvalidArgument(
            "Number of nodes should be the same in "
            "tree, tree_thresholds and node_pcw."));

    const int32 num_classes = node_per_class_weights.shape().dim_size(1);
    const int32 num_data = input_data.shape().dim_size(0);

    Tensor* output_predictions = nullptr;
    TensorShape output_shape;
    output_shape.AddDim(num_data);
    output_shape.AddDim(num_classes);
    OP_REQUIRES_OK(context,
                   context->allocate_output(0, output_shape,
                                            &output_predictions));
    auto out = output_predictions->tensor<float, 2>();

    const auto node_pcw = node_per_class_weights.tensor<float, 2>();
    const auto tree = tree_tensor.tensor<int32, 2>();
    const auto thresholds = tree_thresholds.unaligned_flat<float>();

    for (int i = 0; i < num_data; i++) {
      const Tensor point = input_data.Slice(i, i+1);
      int node_index = 0;
      int parent = -1;
      while (true) {
        const int32 left_child = tree(node_index, CHILDREN_INDEX);
        if (left_child == LEAF_NODE) {
          float sum = Sum<float>(node_per_class_weights.Slice(
              node_index, node_index + 1));
          float parent_weight = 0.0;
          if (sum < valid_leaf_threshold_ && parent >= 0) {
            VLOG(1) << "not enough samples at leaf, including parent counts."
                    << "child sum = " << sum;
            float parent_sum = Sum<float>(node_per_class_weights.Slice(
                parent, parent + 1));
            // Weight the parent's counts just enough so that the new sum is
            // valid_leaf_threshold_, but never give any counts a weight of
            // more than 1.
            parent_weight = std::min(1.0f,
                                (valid_leaf_threshold_ - sum) / parent_sum);
            sum += parent_weight * parent_sum;
            VLOG(1) << "Sum w/ parent included = " << sum;
          }
          for (int c = 0; c < num_classes; c++) {
            float w = node_pcw(node_index, c);
            if (parent_weight > 0.0) {
              w += parent_weight * node_pcw(parent, c);
            }
            out(i, c) = w / sum;
          }
          break;
        } else if (left_child == FREE_NODE) {
          LOG(ERROR) << "Reached a free node, not good.";
          return;
        }
        parent = node_index;
        node_index = left_child +
            DecideNode(point, tree(node_index, FEATURE_INDEX),
                       thresholds(node_index));
      }
    }

    VLOG(1) << "tree: " << tree;
    VLOG(1) << "output: " << out;
  }

 private:
  float valid_leaf_threshold_;
};

REGISTER_KERNEL_BUILDER(Name("TreePredictions").Device(DEVICE_CPU),
                        TreePredictions);

}  // namespace tensorflow
