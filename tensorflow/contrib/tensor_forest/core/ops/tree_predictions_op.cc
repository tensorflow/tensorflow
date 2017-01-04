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
// TreePredictions returns the per-class probabilities for each input by
// evaluating the given tree.
#include <algorithm>

#include "tensorflow/contrib/tensor_forest/core/ops/tree_utils.h"

#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "tensorflow/core/kernels/bounds_check.h"

namespace tensorflow {

using tensorforest::CHILDREN_INDEX;
using tensorforest::FEATURE_INDEX;
using tensorforest::LEAF_NODE;
using tensorforest::FREE_NODE;

using tensorforest::CheckTensorBounds;
using tensorforest::DataColumnTypes;
using tensorforest::FeatureSpec;
using tensorforest::Sum;

using shape_inference::DimensionHandle;
using shape_inference::InferenceContext;
using shape_inference::ShapeHandle;


class TreePredictions : public OpKernel {
 public:
  explicit TreePredictions(OpKernelConstruction* context)
      : OpKernel(context) {
    OP_REQUIRES_OK(context, context->GetAttr(
      "valid_leaf_threshold", &valid_leaf_threshold_));
  }

  void Compute(OpKernelContext* context) override {
    const Tensor& input_data = context->input(0);
    const Tensor& sparse_input_indices = context->input(1);
    const Tensor& sparse_input_values = context->input(2);
    const Tensor& sparse_input_shape = context->input(3);
    const Tensor& input_spec = context->input(4);
    const Tensor& tree_tensor = context->input(5);
    const Tensor& tree_thresholds = context->input(6);
    const Tensor& node_per_class_weights = context->input(7);

    bool sparse_input = (sparse_input_indices.shape().dims() == 2);

    if (sparse_input) {
      OP_REQUIRES(context, sparse_input_values.shape().dims() == 1,
                  errors::InvalidArgument(
                      "sparse_input_values should be one-dimensional"));
      OP_REQUIRES(context, sparse_input_shape.shape().dims() == 1,
                  errors::InvalidArgument(
                      "sparse_input_shape should be one-dimensional"));
      OP_REQUIRES(context,
                  sparse_input_indices.shape().dim_size(0) ==
                  sparse_input_values.shape().dim_size(0),
                  errors::InvalidArgument(
                      "sparse_input_indices and sparse_input_values should "
                      "agree on the number of non-zero values"));
      OP_REQUIRES(context,
                  sparse_input_indices.shape().dim_size(1) ==
                  sparse_input_shape.shape().dim_size(0),
                  errors::InvalidArgument(
                      "sparse_input_indices and sparse_input_shape should "
                      "agree on the dimensionality of data points"));
    } else {
      if (input_data.shape().dim_size(0) > 0) {
        OP_REQUIRES(context, input_data.shape().dims() == 2,
                    errors::InvalidArgument(
                        "input_data should be two-dimensional"));
      }
    }

    OP_REQUIRES(context, tree_tensor.shape().dims() == 2,
                errors::InvalidArgument(
                    "tree should be two-dimensional"));
    OP_REQUIRES(context, tree_thresholds.shape().dims() == 1,
                errors::InvalidArgument(
                    "tree_threhsolds should be one-dimensional"));
    OP_REQUIRES(context, node_per_class_weights.shape().dims() == 2,
                errors::InvalidArgument(
                    "node_pcw should be two-dimensional"));

    OP_REQUIRES(
        context,
        tree_tensor.shape().dim_size(0) ==
        tree_thresholds.shape().dim_size(0) &&
        tree_tensor.shape().dim_size(0) ==
        node_per_class_weights.shape().dim_size(0),
        errors::InvalidArgument(
            "Number of nodes should be the same in "
            "tree, tree_thresholds and node_pcw."));

    // Check tensor bounds.
    if (!CheckTensorBounds(context, input_data)) return;
    if (!CheckTensorBounds(context, sparse_input_indices)) return;
    if (!CheckTensorBounds(context, sparse_input_values)) return;
    if (!CheckTensorBounds(context, sparse_input_shape)) return;
    if (!CheckTensorBounds(context, tree_tensor)) return;
    if (!CheckTensorBounds(context, tree_thresholds)) return;
    if (!CheckTensorBounds(context, node_per_class_weights)) return;

    const int32 num_classes = static_cast<int32>(
        node_per_class_weights.shape().dim_size(1));
    const int32 num_nodes = static_cast<int32>(
        tree_tensor.shape().dim_size(0));
    int32 num_data;
    std::function<bool(int, int, float,
                       tensorforest::DataColumnTypes)> decide_function;

    if (sparse_input) {
      num_data = sparse_input_shape.unaligned_flat<int64>()(0);
      decide_function = [&sparse_input_indices, &sparse_input_values](
          int32 i, int32 feature, float bias, DataColumnTypes type) {
        const auto sparse_indices = sparse_input_indices.matrix<int64>();
        const auto sparse_values = sparse_input_values.vec<float>();
        return tensorforest::DecideSparseNode(
            sparse_indices, sparse_values, i, feature, bias, type);
      };
    } else {
      num_data = static_cast<int32>(input_data.shape().dim_size(0));
      decide_function = [&input_data](
          int32 i, int32 feature, float bias, DataColumnTypes type) {
        const auto input_matrix = input_data.matrix<float>();
        return tensorforest::DecideDenseNode(
            input_matrix, i, feature, bias, type);
      };
    }

    Tensor* output_predictions = nullptr;
    TensorShape output_shape;
    output_shape.AddDim(num_data);
    output_shape.AddDim(num_classes - 1);
    OP_REQUIRES_OK(context,
                   context->allocate_output(0, output_shape,
                                            &output_predictions));
    auto out = output_predictions->tensor<float, 2>();

    const auto node_pcw = node_per_class_weights.tensor<float, 2>();
    const auto tree = tree_tensor.tensor<int32, 2>();
    const auto thresholds = tree_thresholds.unaligned_flat<float>();

    for (int i = 0; i < num_data; i++) {
      int node_index = 0;
      int parent = -1;
      while (true) {
        OP_REQUIRES(context, FastBoundsCheck(node_index, num_nodes),
                    errors::InvalidArgument("node_index not in valid range."))
        const int32 left_child = tree(node_index, CHILDREN_INDEX);
        if (left_child == LEAF_NODE) {
          const int32 flat_leaf_index = node_index * num_classes + 1;
          const int32 flat_parent_index = parent * num_classes + 1;
          std::vector<float> means(num_classes - 1);
          tensorforest::GetParentWeightedMean(
              node_pcw(node_index, 0), node_pcw.data() + flat_leaf_index,
              node_pcw(parent, 0), node_pcw.data() + flat_parent_index,
              valid_leaf_threshold_, num_classes - 1, &means);
          for (int c = 1; c < num_classes; c++) {
            out(i, c - 1) = means[c - 1];
          }
          break;
        } else if (left_child == FREE_NODE) {
          LOG(ERROR) << "Reached a free node, not good.";
          return;
        }
        parent = node_index;
        const int32 feature = tree(node_index, FEATURE_INDEX);
        node_index =
            left_child + decide_function(i, feature, thresholds(node_index),
                                         FeatureSpec(feature, input_spec));
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
