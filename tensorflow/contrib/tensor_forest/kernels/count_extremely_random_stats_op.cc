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
// CountExtremelyRandomStats outputs count-deltas that should be added to
// the node pcws, candidate split pcws, and total split pcws.  It also outputs
// the leaves that each input arrived to for use in SampleInputs.  This is the
// only op that involves tree traversal, and is constructed so that it can
// be run in parallel on separate batches of data.
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "tensorflow/contrib/tensor_forest/kernels/data_spec.h"
#include "tensorflow/contrib/tensor_forest/kernels/tree_utils.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "tensorflow/core/kernels/bounds_check.h"
#include "tensorflow/core/lib/gtl/map_util.h"
#include "tensorflow/core/util/work_sharder.h"

namespace tensorflow {

using std::get;
using std::make_pair;
using std::make_tuple;
using std::tuple;

using tensorforest::CHILDREN_INDEX;
using tensorforest::FEATURE_INDEX;
using tensorforest::LEAF_NODE;
using tensorforest::FREE_NODE;

using tensorforest::CheckTensorBounds;
using tensorforest::DecideNode;
using tensorforest::TensorForestDataSpec;
using tensorforest::Initialize;
using tensorforest::IsAllInitialized;

using shape_inference::DimensionHandle;
using shape_inference::InferenceContext;
using shape_inference::ShapeHandle;

// A data structure to store the results of parallel tree traversal.
struct InputDataResult {
  // A list of each node that was visited.
  std::vector<int32> node_indices;
  // The accumulator of the leaf that a data point ended up at, or -1 if none.
  int32 leaf_accumulator;
  // The left-branch taken candidate splits.
  std::vector<int32> split_adds;
  // If the candidate splits for the leaf that a data point arrived at
  // were initialized or not, which determines if we add this to total
  // pcw counts or not.
  bool splits_initialized;
};


struct EvaluateParams {
  TensorForestDataSpec input_spec;
  Tensor dense_input;
  Tensor sparse_indices;
  Tensor sparse_values;
  Tensor input_labels;
  Tensor tree_tensor;
  Tensor tree_thresholds;
  Tensor node_to_accumulator;
  Tensor candidate_split_features;
  Tensor candidate_split_thresholds;
  InputDataResult* results;
};

void Evaluate(const EvaluateParams& params, int32 start, int32 end) {
  const auto tree = params.tree_tensor.tensor<int32, 2>();
  const auto thresholds = params.tree_thresholds.unaligned_flat<float>();
  const auto node_map = params.node_to_accumulator.unaligned_flat<int32>();
  const auto split_features =
      params.candidate_split_features.tensor<int32, 2>();
  const auto split_thresholds =
      params.candidate_split_thresholds.tensor<float, 2>();

  const int32 num_splits = static_cast<int32>(
      params.candidate_split_features.shape().dim_size(1));
  const int32 num_nodes = static_cast<int32>(
      params.tree_tensor.shape().dim_size(0));
  const int32 num_accumulators = static_cast<int32>(
      params.candidate_split_features.shape().dim_size(0));

  // Lambdas to capture the eigen-tensors so we don't the conversion overhead
  // on each call to DecideNode.
  const auto get_dense = tensorforest::GetDenseFunctor(params.dense_input);
  const auto get_sparse = tensorforest::GetSparseFunctor(params.sparse_indices,
                                                         params.sparse_values);

  for (int32 i = start; i < end; ++i) {
    int node_index = 0;
    params.results[i].splits_initialized = false;
    while (true) {
      params.results[i].node_indices.push_back(node_index);
      CHECK_LT(node_index, num_nodes);
      int32 left_child =
          internal::SubtleMustCopy(tree(node_index, CHILDREN_INDEX));
      if (left_child == LEAF_NODE) {
        const int32 accumulator =
            internal::SubtleMustCopy(node_map(node_index));
        params.results[i].leaf_accumulator = accumulator;
        // If the leaf is not fertile or is not yet initialized, we don't
        // count it in the candidate/total split per-class-weights because
        // it won't have any candidate splits yet.
        if (accumulator >= 0 &&
            IsAllInitialized(split_features, accumulator, num_splits)) {
          CHECK_LT(accumulator, num_accumulators);
          params.results[i].splits_initialized = true;
          for (int split = 0; split < num_splits; split++) {
            const int32 feature = split_features(accumulator, split);
            if (!DecideNode(get_dense, get_sparse, i, feature,
                            split_thresholds(accumulator, split),
                            params.input_spec)) {
              params.results[i].split_adds.push_back(split);
            }
          }
        }
        break;
      } else if (left_child == FREE_NODE) {
        LOG(ERROR) << "Reached a free node, not good.";
        params.results[i].node_indices.push_back(FREE_NODE);
        break;
      }
      const int32 feature = tree(node_index, FEATURE_INDEX);
      node_index =
          left_child + DecideNode(get_dense, get_sparse, i, feature,
                                  thresholds(node_index), params.input_spec);
    }
  }
}

class CountExtremelyRandomStats : public OpKernel {
 public:
  explicit CountExtremelyRandomStats(OpKernelConstruction* context)
      : OpKernel(context) {
    OP_REQUIRES_OK(context, context->GetAttr(
        "num_classes", &num_classes_));
    OP_REQUIRES_OK(context, context->GetAttr(
        "regression", &regression_));
    string serialized_proto;
    OP_REQUIRES_OK(context, context->GetAttr("input_spec", &serialized_proto));
    input_spec_.ParseFromString(serialized_proto);
  }

  void Compute(OpKernelContext* context) override {
    const Tensor& input_data = context->input(0);
    const Tensor& sparse_input_indices = context->input(1);
    const Tensor& sparse_input_values = context->input(2);
    const Tensor& sparse_input_shape = context->input(3);
    const Tensor& input_labels = context->input(4);
    const Tensor& input_weights = context->input(5);
    const Tensor& tree_tensor = context->input(6);
    const Tensor& tree_thresholds = context->input(7);
    const Tensor& node_to_accumulator = context->input(8);
    const Tensor& candidate_split_features = context->input(9);
    const Tensor& candidate_split_thresholds = context->input(10);
    const Tensor& birth_epochs = context->input(11);
    const Tensor& current_epoch = context->input(12);

    bool sparse_input = (sparse_input_indices.shape().dims() == 2);
    bool have_weights = (input_weights.shape().dim_size(0) > 0);
    int32 num_data = -1;

    // Check inputs.
    if (sparse_input) {
      const auto sparse_shape = sparse_input_shape.unaligned_flat<int64>();
      // TODO(gilberth): This is because we can't figure out the shape
      // of a sparse tensor at graph-build time, even if the dimension is
      // actually known.
      input_spec_.mutable_sparse(0)->set_size(sparse_shape(1));
      num_data = sparse_shape(0);

      OP_REQUIRES(context, sparse_input_shape.shape().dims() == 1,
                  errors::InvalidArgument(
                      "sparse_input_shape should be one-dimensional"));
      OP_REQUIRES(context,
                  sparse_input_shape.shape().dim_size(0) == 2,
                  errors::InvalidArgument(
                      "The sparse input data should be two-dimensional"));
      OP_REQUIRES(context, sparse_input_values.shape().dims() == 1,
                  errors::InvalidArgument(
                      "sparse_input_values should be one-dimensional"));
      OP_REQUIRES(context, sparse_input_indices.shape().dims() == 2,
                  errors::InvalidArgument(
                      "The sparse input data should be two-dimensional"));
      OP_REQUIRES(context,
                  sparse_input_indices.shape().dim_size(0) ==
                  sparse_input_values.shape().dim_size(0),
                  errors::InvalidArgument(
                      "sparse_input_indices and sparse_input_values should "
                      "agree on the number of non-zero values"));
    }

    if (input_data.shape().dim_size(0) > 0) {
      const int32 dense_num_data =
          static_cast<int32>(input_data.shape().dim_size(0));
      if (num_data > 0) {
        CHECK_EQ(num_data, dense_num_data)
            << "number of examples must match for sparse + dense input.";
      }
      num_data = dense_num_data;

      OP_REQUIRES(context, input_data.shape().dims() == 2,
                  errors::InvalidArgument(
                      "input_data should be two-dimensional"));
      OP_REQUIRES(
          context,
          input_data.shape().dim_size(0) == input_labels.shape().dim_size(0),
          errors::InvalidArgument(
              "Number of inputs should be the same in "
              "input_data and input_labels."));
    }

    if (have_weights) {
      OP_REQUIRES(
          context,
          input_weights.shape().dim_size(0) == input_labels.shape().dim_size(0),
          errors::InvalidArgument(
              "Number of inputs should be the same in input_weights and "
              "input_labels."));
    }

    OP_REQUIRES(context, input_labels.shape().dims() >= 1,
                errors::InvalidArgument(
                    "input_labels should be at least one-dimensional"));
    OP_REQUIRES(context, tree_tensor.shape().dims() == 2,
            errors::InvalidArgument(
                "tree should be two-dimensional"));
    OP_REQUIRES(context, tree_thresholds.shape().dims() == 1,
            errors::InvalidArgument(
                "tree_thresholds should be one-dimensional"));
    OP_REQUIRES(context, node_to_accumulator.shape().dims() == 1,
            errors::InvalidArgument(
                "node_to_accumulator should be one-dimensional"));
    OP_REQUIRES(context, candidate_split_features.shape().dims() == 2,
            errors::InvalidArgument(
                "candidate_split_features should be two-dimensional"));
    OP_REQUIRES(context, candidate_split_thresholds.shape().dims() == 2,
            errors::InvalidArgument(
                "candidate_split_thresholds should be two-dimensional"));
    OP_REQUIRES(context, birth_epochs.shape().dims() == 1,
            errors::InvalidArgument(
                "birth_epochs should be one-dimensional"));
    OP_REQUIRES(context, current_epoch.shape().dims() == 1,
            errors::InvalidArgument(
                "current_epoch should be one-dimensional"));

    OP_REQUIRES(
        context,
        tree_tensor.shape().dim_size(0) ==
        tree_thresholds.shape().dim_size(0) &&
        tree_tensor.shape().dim_size(0) ==
        node_to_accumulator.shape().dim_size(0) &&
        tree_tensor.shape().dim_size(0) ==
        birth_epochs.shape().dim_size(0),
        errors::InvalidArgument(
            "Number of nodes should be the same in "
            "tree, tree_thresholds, node_to_accumulator, and birth_epoch."));
    OP_REQUIRES(
        context,
        candidate_split_features.shape() == candidate_split_thresholds.shape(),
        errors::InvalidArgument(
            "candidate_split_features and candidate_split_thresholds should be "
            "the same shape."));
    OP_REQUIRES(
        context,
        current_epoch.shape().dim_size(0) == 1,
        errors::InvalidArgument(
            "The current_epoch should be a tensor of shape (1)."));

    // Check tensor bounds.
    if (!CheckTensorBounds(context, input_data)) return;
    if (!CheckTensorBounds(context, sparse_input_indices)) return;
    if (!CheckTensorBounds(context, sparse_input_values)) return;
    if (!CheckTensorBounds(context, sparse_input_shape)) return;
    if (!CheckTensorBounds(context, input_labels)) return;
    if (!CheckTensorBounds(context, input_weights)) return;
    if (!CheckTensorBounds(context, tree_tensor)) return;
    if (!CheckTensorBounds(context, tree_thresholds)) return;
    if (!CheckTensorBounds(context, node_to_accumulator)) return;
    if (!CheckTensorBounds(context, candidate_split_features)) return;
    if (!CheckTensorBounds(context, candidate_split_thresholds)) return;
    if (!CheckTensorBounds(context, birth_epochs)) return;
    if (!CheckTensorBounds(context, current_epoch)) return;

    // Evaluate input data in parallel.
    const int32 epoch = current_epoch.unaligned_flat<int32>()(0);

    std::unique_ptr<InputDataResult[]> results(new InputDataResult[num_data]);
    auto worker_threads = context->device()->tensorflow_cpu_worker_threads();
    int num_threads = worker_threads->num_threads;
    EvaluateParams params;
    params.dense_input = input_data;
    params.sparse_indices = sparse_input_indices;
    params.sparse_values = sparse_input_values;
    params.input_spec = input_spec_;
    params.input_labels = input_labels;
    params.tree_tensor = tree_tensor;
    params.tree_thresholds = tree_thresholds;
    params.node_to_accumulator = node_to_accumulator;
    params.candidate_split_features = candidate_split_features;
    params.candidate_split_thresholds = candidate_split_thresholds;
    params.results = results.get();
    // Require at least 100 inputs per thread.  I guess that's about 800 cost
    // per unit.  This isn't well defined.
    const int64 costPerUnit = 800;
    auto work = [&params, num_data](int64 start, int64 end) {
      CHECK(start <= end);
      CHECK(end <= num_data);
      Evaluate(params, static_cast<int32>(start), static_cast<int32>(end));
    };
    Shard(num_threads, worker_threads->workers, num_data, costPerUnit, work);

    const int32 num_nodes = static_cast<int32>(tree_tensor.shape().dim_size(0));
    if (regression_) {
      ProcessResultsRegression(context, input_labels, input_weights,
                               birth_epochs, epoch, std::move(results),
                               num_nodes);
    } else {
      ProcessResultsClassification(context, input_labels, input_weights,
                                   birth_epochs, epoch, std::move(results),
                                   num_nodes);
    }
  }

 protected:
  void ProcessResultsClassification(OpKernelContext* context,
                                    const Tensor& input_labels,
                                    const Tensor& input_weights,
                                    const Tensor& birth_epochs, int32 epoch,
                                    std::unique_ptr<InputDataResult[]> results,
                                    int32 num_nodes) {
    const int32 num_data = static_cast<int32>(input_labels.shape().dim_size(0));
    const auto labels = input_labels.unaligned_flat<float>();
    const auto start_epochs = birth_epochs.unaligned_flat<int32>();
    const auto weights = input_weights.unaligned_flat<float>();

    // Unused outputs for classification.  Still have to specify them or
    // tensorflow complains.
    Tensor* dummy = nullptr;
    TensorShape dummy_shape;
    dummy_shape.AddDim(0);
    OP_REQUIRES_OK(context,
                   context->allocate_output(1, dummy_shape, &dummy));
    OP_REQUIRES_OK(context,
                   context->allocate_output(4, dummy_shape, &dummy));
    OP_REQUIRES_OK(context,
                   context->allocate_output(7, dummy_shape, &dummy));

    // node pcw delta
    Tensor* output_node_pcw_sums_delta = nullptr;
    TensorShape node_pcw_sums_shape;
    node_pcw_sums_shape.AddDim(num_nodes);
    node_pcw_sums_shape.AddDim(num_classes_);
    OP_REQUIRES_OK(context,
                   context->allocate_output(0, node_pcw_sums_shape,
                                            &output_node_pcw_sums_delta));
    Initialize<float>(*output_node_pcw_sums_delta, 0);
    auto out_node_sums = output_node_pcw_sums_delta->tensor<float, 2>();

    // leaves
    Tensor* output_leaves = nullptr;
    TensorShape leaves_shape;
    leaves_shape.AddDim(num_data);
    OP_REQUIRES_OK(context,
                   context->allocate_output(8, leaves_shape, &output_leaves));
    auto out_leaves = output_leaves->unaligned_flat<int32>();

    // <accumulator, class> -> count delta
    PairMapType<float> total_delta;
    // <accumulator, split, class> -> count delta
    TupleMapType<float> split_delta;

    for (int32 i = 0; i < num_data; ++i) {
      out_leaves(i) = results[i].node_indices.back();
      float w = 1.0;
      if (weights.size() > 0) {
        w = weights(i);
      }

      const int32 label = internal::SubtleMustCopy(
          static_cast<int32>(labels(i)));
      // Labels that come from sparse tensors can have missing values.
      if (label < 0) {
        continue;
      }
      const int32 column = label + 1;
      CHECK_LT(column, num_classes_);
      const int32 accumulator = results[i].leaf_accumulator;
      for (const int32 node : results[i].node_indices) {
        if (epoch > start_epochs(node) + 1) {
          continue;
        }
        out_node_sums(node, column) += w;
        out_node_sums(node, 0) += w;
      }

      if (epoch > start_epochs(out_leaves(i)) + 1) {
        continue;
      }
      if (accumulator >= 0 && results[i].splits_initialized) {
        total_delta[make_pair(accumulator, column)] += w;
        total_delta[make_pair(accumulator, 0)] += w;
        for (const int32 split : results[i].split_adds) {
          split_delta[make_tuple(accumulator, split, column)] += w;
          split_delta[make_tuple(accumulator, split, 0)] += w;
        }
      }
    }

    // candidate splits pcw indices
    Tensor* output_candidate_pcw_indices = nullptr;
    TensorShape candidate_pcw_shape;
    candidate_pcw_shape.AddDim(split_delta.size());
    candidate_pcw_shape.AddDim(3);
    OP_REQUIRES_OK(context,
                   context->allocate_output(2, candidate_pcw_shape,
                                            &output_candidate_pcw_indices));
    auto out_candidate_indices =
        output_candidate_pcw_indices->tensor<int32, 2>();

    // candidate splits pcw delta
    Tensor* output_candidate_pcw_delta = nullptr;
    TensorShape candidate_pcw_delta_shape;
    candidate_pcw_delta_shape.AddDim(split_delta.size());
    OP_REQUIRES_OK(context,
                   context->allocate_output(3, candidate_pcw_delta_shape,
                                            &output_candidate_pcw_delta));
    auto out_candidate = output_candidate_pcw_delta->unaligned_flat<float>();

    // total splits indices
    Tensor* output_total_pcw_indices = nullptr;
    TensorShape total_pcw_shape;
    total_pcw_shape.AddDim(total_delta.size());
    total_pcw_shape.AddDim(2);
    OP_REQUIRES_OK(context,
                   context->allocate_output(5, total_pcw_shape,
                                            &output_total_pcw_indices));
    auto out_total_indices = output_total_pcw_indices->tensor<int32, 2>();

    // total splits delta
    Tensor* output_total_pcw_delta = nullptr;
    TensorShape total_pcw_delta_shape;
    total_pcw_delta_shape.AddDim(total_delta.size());
    OP_REQUIRES_OK(context,
                   context->allocate_output(6, total_pcw_delta_shape,
                                            &output_total_pcw_delta));
    auto out_total = output_total_pcw_delta->unaligned_flat<float>();

    // Copy total deltas to output.
    int32 output_slot = 0;
    for (const auto& updates : total_delta) {
      out_total_indices(output_slot, 0) = updates.first.first;
      out_total_indices(output_slot, 1) = updates.first.second;
      out_total(output_slot) = updates.second;
      ++output_slot;
    }

    // Copy split deltas to output.
    output_slot = 0;
    for (const auto& updates : split_delta) {
      out_candidate_indices(output_slot, 0) = get<0>(updates.first);
      out_candidate_indices(output_slot, 1) = get<1>(updates.first);
      out_candidate_indices(output_slot, 2) = get<2>(updates.first);
      out_candidate(output_slot) = updates.second;
      ++output_slot;
    }
  }

  void ProcessResultsRegression(OpKernelContext* context,
                                const Tensor& input_labels,
                                const Tensor& input_weights,
                                const Tensor& birth_epochs, const int32 epoch,
                                std::unique_ptr<InputDataResult[]> results,
                                int32 num_nodes) {
    const int32 num_data = static_cast<int32>(input_labels.shape().dim_size(0));
    int32 num_outputs = 1;
    if (input_labels.shape().dims() > 1) {
        num_outputs = static_cast<int32>(input_labels.shape().dim_size(1));
    }
    const auto labels = input_labels.unaligned_flat<float>();
    const auto start_epochs = birth_epochs.unaligned_flat<int32>();
    const auto weights = input_weights.unaligned_flat<float>();

    // node pcw delta
    Tensor* output_node_pcw_sums_delta = nullptr;
    TensorShape node_pcw_sums_shape;
    node_pcw_sums_shape.AddDim(num_nodes);
    node_pcw_sums_shape.AddDim(num_classes_);
    OP_REQUIRES_OK(context,
                   context->allocate_output(0, node_pcw_sums_shape,
                                            &output_node_pcw_sums_delta));
    Initialize<float>(*output_node_pcw_sums_delta, 0);
    auto out_node_sums = output_node_pcw_sums_delta->tensor<float, 2>();

    Tensor* output_node_pcw_squares_delta = nullptr;
    TensorShape node_pcw_squares_shape;
    node_pcw_squares_shape.AddDim(num_nodes);
    node_pcw_squares_shape.AddDim(num_classes_);
    OP_REQUIRES_OK(context,
                   context->allocate_output(1, node_pcw_squares_shape,
                                            &output_node_pcw_squares_delta));
    Initialize<float>(*output_node_pcw_squares_delta, 0);
    auto out_node_squares = output_node_pcw_squares_delta->tensor<float, 2>();

    // leaves
    Tensor* output_leaves = nullptr;
    TensorShape leaves_shape;
    leaves_shape.AddDim(num_data);
    OP_REQUIRES_OK(context,
                   context->allocate_output(8, leaves_shape, &output_leaves));
    auto out_leaves = output_leaves->unaligned_flat<int32>();

    // <accumulator> -> label index
    std::unordered_map<int32, std::unordered_set<int32>> total_delta;
    // <accumulator, split> -> label index
    PairMapType<std::unordered_set<int32>> split_delta;

    for (int32 i = 0; i < num_data; ++i) {
      const int32 accumulator = results[i].leaf_accumulator;
      float w = 1.0;
      if (weights.size() > 0) {
        w = weights(i);
      }

      for (const int32 node : results[i].node_indices) {
        if (epoch > start_epochs(node) + 1) {
          continue;
        }
        for (int32 j = 0; j < num_outputs; ++j) {
          const float output = labels(i * num_outputs + j);
          out_node_sums(node, j + 1) += w * output;
          out_node_squares(node, j + 1) += w * output * output;
          out_node_sums(node, 0) += w;
          out_node_squares(node, 0) += w;
        }
      }
      out_leaves(i) = results[i].node_indices.back();
      if (epoch > start_epochs(out_leaves(i)) + 1) {
        continue;
      }
      if (accumulator >= 0 && results[i].splits_initialized) {
        total_delta[accumulator].insert(i);
        for (const int32 split : results[i].split_adds) {
          split_delta[make_pair(accumulator, split)].insert(i);
        }
      }
    }

    // candidate splits pcw indices
    Tensor* output_candidate_pcw_indices = nullptr;
    TensorShape candidate_pcw_shape;
    candidate_pcw_shape.AddDim(split_delta.size());
    candidate_pcw_shape.AddDim(2);
    OP_REQUIRES_OK(context,
                   context->allocate_output(2, candidate_pcw_shape,
                                            &output_candidate_pcw_indices));
    auto out_candidate_indices =
        output_candidate_pcw_indices->tensor<int32, 2>();

    // candidate splits pcw delta
    // sums
    Tensor* output_candidate_pcw_sums = nullptr;
    TensorShape candidate_pcw_sums_shape;
    candidate_pcw_sums_shape.AddDim(split_delta.size());
    candidate_pcw_sums_shape.AddDim(num_classes_);
    OP_REQUIRES_OK(context,
                   context->allocate_output(3, candidate_pcw_sums_shape,
                                            &output_candidate_pcw_sums));
    Initialize<float>(*output_candidate_pcw_sums, 0);
    auto out_split_sums = output_candidate_pcw_sums->tensor<float, 2>();

    // squares
    Tensor* output_candidate_pcw_squares = nullptr;
    TensorShape candidate_pcw_squares_shape;
    candidate_pcw_squares_shape.AddDim(split_delta.size());
    candidate_pcw_squares_shape.AddDim(num_classes_);
    OP_REQUIRES_OK(context,
                   context->allocate_output(4, candidate_pcw_squares_shape,
                                            &output_candidate_pcw_squares));
    Initialize<float>(*output_candidate_pcw_squares, 0);
    auto out_split_squares = output_candidate_pcw_squares->tensor<float, 2>();

    // total splits indices
    Tensor* output_total_pcw_indices = nullptr;
    TensorShape total_pcw_shape;
    total_pcw_shape.AddDim(total_delta.size());
    total_pcw_shape.AddDim(1);
    OP_REQUIRES_OK(context,
                   context->allocate_output(5, total_pcw_shape,
                                            &output_total_pcw_indices));
    auto out_total_indices = output_total_pcw_indices->unaligned_flat<int32>();

    // total splits delta
    // sums
    Tensor* output_total_pcw_sums = nullptr;
    TensorShape total_pcw_sums_shape;
    total_pcw_sums_shape.AddDim(total_delta.size());
    total_pcw_sums_shape.AddDim(num_classes_);
    OP_REQUIRES_OK(context,
                   context->allocate_output(6, total_pcw_sums_shape,
                                            &output_total_pcw_sums));
    Initialize<float>(*output_total_pcw_sums, 0);
    auto out_total_sums = output_total_pcw_sums->tensor<float, 2>();

    // squares
    Tensor* output_total_pcw_squares = nullptr;
    TensorShape total_pcw_squares_shape;
    total_pcw_squares_shape.AddDim(total_delta.size());
    total_pcw_squares_shape.AddDim(num_classes_);
    OP_REQUIRES_OK(context,
                   context->allocate_output(7, total_pcw_squares_shape,
                                            &output_total_pcw_squares));
    Initialize<float>(*output_total_pcw_squares, 0);
    auto out_total_squares = output_total_pcw_squares->tensor<float, 2>();

    // Copy total deltas to output.
    int32 output_slot = 0;
    for (const auto& updates : total_delta) {
      out_total_indices(output_slot) = updates.first;
      for (const int32 i : updates.second) {
        for (int32 j = 0; j < num_outputs; ++j) {
          const float output = labels(i * num_outputs + j);
          out_total_sums(output_slot, j + 1) += output;
          out_total_squares(output_slot, j + 1) += output * output;
        }
      }
      out_total_sums(output_slot, 0) += updates.second.size();
      out_total_squares(output_slot, 0) += updates.second.size();
      ++output_slot;
    }

    // Copy split deltas to output.
    output_slot = 0;
    for (const auto& updates : split_delta) {
      out_candidate_indices(output_slot, 0) = updates.first.first;
      out_candidate_indices(output_slot, 1) = updates.first.second;
      for (const int32 i : updates.second) {
        for (int32 j = 0; j < num_outputs; ++j) {
          const float output = labels(i * num_outputs + j);
          out_split_sums(output_slot, j + 1) += output;
          out_split_squares(output_slot, j + 1) += output * output;
        }
      }
      out_split_sums(output_slot, 0) += updates.second.size();
      out_split_squares(output_slot, 0) += updates.second.size();
      ++output_slot;
    }
  }

  struct PairIntHash {
   public:
    std::size_t operator()(const std::pair<int32, int32>& x) const {
      // Bit-rotate x.first by 16 bits before xor-ing to minimize hash
      // collisions in the frequent case when both elements of the pair are
      // small.
      return (x.first << 16 | x.first >> 16) ^ x.second;
    }
  };
  template <typename V>
  using PairMapType =
      std::unordered_map<std::pair<int32, int32>, V, PairIntHash>;

  struct TupleIntHash {
   public:
    std::size_t operator()(const std::tuple<int32, int32, int32>& x) const {
      const int32 first = get<0>(x);
      const int32 second = get<1>(x);
      // Again, we bit-rotate (once by 16 bits, and once by 8 bits) to minimize
      // hash collisions among small values.
      return (first << 16 | first >> 16) ^ (second << 8 | second >> 24) ^
          get<2>(x);
    }
  };
  template <typename V>
  using TupleMapType = std::unordered_map<tuple<int32, int32, int32>, V,
      TupleIntHash>;

  int32 num_classes_;
  bool regression_;
  tensorforest::TensorForestDataSpec input_spec_;
};


REGISTER_KERNEL_BUILDER(Name("CountExtremelyRandomStats").Device(DEVICE_CPU),
                        CountExtremelyRandomStats);

}  // namespace tensorflow
