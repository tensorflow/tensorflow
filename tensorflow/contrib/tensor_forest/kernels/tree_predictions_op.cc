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

#include "tensorflow/contrib/tensor_forest/kernels/data_spec.h"
#include "tensorflow/contrib/tensor_forest/kernels/tree_utils.h"

#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "tensorflow/core/kernels/bounds_check.h"
#include "tensorflow/core/util/work_sharder.h"

namespace tensorflow {

using tensorforest::CHILDREN_INDEX;
using tensorforest::FEATURE_INDEX;
using tensorforest::LEAF_NODE;
using tensorforest::FREE_NODE;

using tensorforest::CheckTensorBounds;

namespace {
// Traverse the tree for every example from start to end. Put the resulting
// prediction probability into output_predictions[i].
void Evaluate(OpKernelContext* context,
              const std::function<bool(int, int32, float)>& decide,
              const Tensor& weights, const Tensor& tree_tensor,
              const Tensor& tree_thresholds, int valid_leaf_threshold,
              Tensor* output_predictions, int32 start, int32 end) {
  auto out = output_predictions->tensor<float, 2>();

  const auto node_pcw = weights.tensor<float, 2>();
  const auto tree = tree_tensor.tensor<int32, 2>();
  const auto thresholds = tree_thresholds.unaligned_flat<float>();

  const int32 num_classes = static_cast<int32>(weights.shape().dim_size(1));
  const int32 num_nodes = static_cast<int32>(tree_tensor.shape().dim_size(0));

  for (int i = start; i < end; i++) {
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
            valid_leaf_threshold, num_classes - 1, &means);
        const int32 start_index = i * (num_classes - 1);
        std::copy(means.begin(), means.end(), out.data() + start_index);
        break;
      } else if (left_child == FREE_NODE) {
        LOG(ERROR) << "Reached a free node, not good.";
        return;
      }
      parent = node_index;
      const int32 feature = tree(node_index, FEATURE_INDEX);
      node_index = left_child + decide(i, feature, thresholds(node_index));
    }
  }
}
}  // namespace

class TreePredictions : public OpKernel {
 public:
  explicit TreePredictions(OpKernelConstruction* context)
      : OpKernel(context) {
    OP_REQUIRES_OK(context, context->GetAttr(
      "valid_leaf_threshold", &valid_leaf_threshold_));

    string serialized_proto;
    OP_REQUIRES_OK(context, context->GetAttr("input_spec", &serialized_proto));
    input_spec_.ParseFromString(serialized_proto);
  }

  void Compute(OpKernelContext* context) override {
    const Tensor& input_data = context->input(0);
    const Tensor& sparse_input_indices = context->input(1);
    const Tensor& sparse_input_values = context->input(2);
    const Tensor& sparse_input_shape = context->input(3);
    const Tensor& tree_tensor = context->input(4);
    const Tensor& tree_thresholds = context->input(5);
    const Tensor& node_per_class_weights = context->input(6);

    int32 num_data = 0;
    if (sparse_input_indices.shape().dims() == 2) {
      const auto sparse_shape = sparse_input_shape.unaligned_flat<int64>();
      // TODO(gilberth): This is because we can't figure out the shape
      // of a sparse tensor at graph-build time, even if the dimension is
      // actually known.
      input_spec_.mutable_sparse(0)->set_size(sparse_shape(1));
      num_data = sparse_shape(0);
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
    }

    if (input_data.shape().dim_size(0) > 0) {
      const int32 dense_num_data =
          static_cast<int32>(input_data.shape().dim_size(0));
      if (num_data > 0) {
        CHECK_EQ(num_data, dense_num_data)
            << "number of examples must match for sparse + dense input.";
      }
      num_data = dense_num_data;
      OP_REQUIRES(
          context, input_data.shape().dims() == 2,
          errors::InvalidArgument("input_data should be two-dimensional"));
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

    Tensor* output_predictions = nullptr;
    TensorShape output_shape;
    output_shape.AddDim(num_data);
    output_shape.AddDim(num_classes - 1);
    OP_REQUIRES_OK(context,
                   context->allocate_output(0, output_shape,
                                            &output_predictions));

    // Lambdas to capture the eigen-tensors so we don't the conversion overhead
    // on each call to DecideNode.
    const auto get_dense = tensorforest::GetDenseFunctor(input_data);
    const auto get_sparse = tensorforest::GetSparseFunctor(sparse_input_indices,
                                                           sparse_input_values);

    auto decide = [&get_dense, &get_sparse, this](int example, int32 feature,
                                                  float threshold) {
      return tensorforest::DecideNode(get_dense, get_sparse, example, feature,
                                      threshold, input_spec_);
    };

    auto worker_threads = context->device()->tensorflow_cpu_worker_threads();
    int num_threads = worker_threads->num_threads;

    const int64 costPerUnit = 800;
    auto work = [context, &decide, &node_per_class_weights, &tree_tensor,
                 &tree_thresholds, this, &output_predictions,
                 num_data](int64 start, int64 end) {
      CHECK(start <= end);
      CHECK(end <= num_data);
      Evaluate(context, decide, node_per_class_weights, tree_tensor,
               tree_thresholds, valid_leaf_threshold_, output_predictions,
               static_cast<int32>(start), static_cast<int32>(end));
    };
    Shard(num_threads, worker_threads->workers, num_data, costPerUnit, work);
  }

 private:
  float valid_leaf_threshold_;
  tensorforest::TensorForestDataSpec input_spec_;
};

REGISTER_KERNEL_BUILDER(Name("TreePredictions").Device(DEVICE_CPU),
                        TreePredictions);

}  // namespace tensorflow
