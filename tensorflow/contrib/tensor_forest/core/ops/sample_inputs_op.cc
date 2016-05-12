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
// SampleInputs initializes candidate splits/threshold values randomly
// from incoming data for not-yet-initialized fertile nodes.
#include <ctime>
#include <unordered_map>
#include <set>

#include "tensorflow/contrib/tensor_forest/core/ops/tree_utils.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/kernels/bounds_check.h"
#include "tensorflow/core/lib/random/philox_random.h"
#include "tensorflow/core/lib/random/simple_philox.h"
#include "tensorflow/core/platform/logging.h"

namespace tensorflow {

using tensorforest::CheckTensorBounds;
using tensorforest::IsAllInitialized;


REGISTER_OP("SampleInputs")
  .Attr("split_initializations_per_input: int32")
  .Attr("split_sampling_random_seed: int32")
  .Input("input_data: float")
  .Input("node_to_accumulator: int32")
  .Input("leaves: int32")
  .Input("candidate_split_features: int32")
  .Input("candidate_split_thresholds: float")
  .Output("accumulators_to_update: int32")
  .Output("new_split_feature_rows: int32")
  .Output("new_split_threshold_rows: float")
  .Doc(R"doc(
  Initializes candidate splits for newly fertile nodes.

  In an extremely random forest, we don't consider all possible threshold
  values for a candidate split feature, but rather only a sampling of them.
  This Op takes those samples from the training data in `input_data`.  The
  feature and threshold samples are stored in tensors that are indexed by
  accumulator slot, so for each input, we must first look up which leaf
  it ended up in (using `leaves`) and then which accumulator slot if any
  that leaf maps to (using `node_to_accumulator`).

  The attribute `split_initializations_per_input` controls how many splits
  a single training example can initialize, and the attribute
  `split_sampling_random_seed` sets the random number generator's seed
  (a value of 0 means use the current time as the seed).

  input_data: The features for the current batch of training data.
    `input_data[i][j]` is the j-th feature of the i-th input.
  node_to_accumulator: For a fertile node i, node_to_accumulator[i] is the
    associated accumulator slot.  For non-fertile nodes, it is -1.
  leaves: `leaves[i]` is the leaf that the i-th input landed in, as
    calculated by CountExtremelyRandomStats.
  candidate_split_features: The current features for the candidate splits;
    `candidate_split_features[a][s]` is the index of the feature being
    considered by split s in accumulator slot a.
  candidate_split_thresholds: The current thresholds for the candidate splits;
    `candidate_split_thresholds[a][s]` is the threshold value being
    considered by split s in accumulator slot a.
  accumulators_to_update: A list of the accumulators to change in the
    candidate_split_features and candidate_split_thresholds tensors.
  new_split_feature_rows: The new values for the candidate_split_features
    tensor.  Intended to be used with
    `tf.scatter_update(candidate_split_features,
                       accumulators_to_update,
                       new_split_feature_rows)`
  new_split_threshold_rows:  The new values for the candidate_split_thresholds
    tensor.  Intended to be used with
    `tf.scatter_update(candidate_split_thresholds,
                       accumulators_to_update,
                       new_split_feature_thresholds)`
)doc");

class SampleInputs : public OpKernel {
 public:
  explicit SampleInputs(OpKernelConstruction* context)
      : OpKernel(context) {
    OP_REQUIRES_OK(context, context->GetAttr(
        "split_initializations_per_input", &split_initializations_per_input_));
    OP_REQUIRES_OK(context, context->GetAttr(
        "split_sampling_random_seed", &split_sampling_random_seed_));
    // Set up the random number generator.
    if (split_sampling_random_seed_ == 0) {
      uint64 time_seed = static_cast<uint64>(std::clock());
      single_rand_ = std::unique_ptr<random::PhiloxRandom>(
          new random::PhiloxRandom(time_seed));
    } else {
      single_rand_ = std::unique_ptr<random::PhiloxRandom>(
          new random::PhiloxRandom(split_sampling_random_seed_));
    }

    rng_ = std::unique_ptr<random::SimplePhilox>(
        new random::SimplePhilox(single_rand_.get()));
  }

  void Compute(OpKernelContext* context) override {
    const Tensor& input_data = context->input(0);
    const Tensor& node_to_accumulator = context->input(1);
    const Tensor& leaves = context->input(2);
    const Tensor& split_features = context->input(3);
    const Tensor& split_thresholds = context->input(4);

    OP_REQUIRES(context, input_data.shape().dims() == 2,
                errors::InvalidArgument(
                    "input_data should be two-dimensional"));
    OP_REQUIRES(context, node_to_accumulator.shape().dims() == 1,
                errors::InvalidArgument(
                    "node_to_accumulator should be one-dimensional"));
    OP_REQUIRES(context, leaves.shape().dims() == 1,
                errors::InvalidArgument(
                    "leaves should be one-dimensional"));
    OP_REQUIRES(context, split_features.shape().dims() == 2,
                errors::InvalidArgument(
                    "split_features should be two-dimensional"));
    OP_REQUIRES(context, split_thresholds.shape().dims() == 2,
                errors::InvalidArgument(
                    "split_thresholds should be two-dimensional"));

    OP_REQUIRES(
        context,
        split_features.shape() == split_thresholds.shape(),
        errors::InvalidArgument(
            "split_features and split_thresholds should be the same shape."));

    // Check tensor bounds.
    if (!CheckTensorBounds(context, input_data)) return;
    if (!CheckTensorBounds(context, node_to_accumulator)) return;
    if (!CheckTensorBounds(context, leaves)) return;
    if (!CheckTensorBounds(context, split_features)) return;
    if (!CheckTensorBounds(context, split_thresholds)) return;

    const auto inputs = input_data.tensor<float, 2>();
    const auto leaves_vec = leaves.unaligned_flat<int32>();
    const auto node_map = node_to_accumulator.unaligned_flat<int32>();
    const auto features = split_features.tensor<int32, 2>();
    const auto thresholds = split_thresholds.tensor<float, 2>();

    const int32 num_data = static_cast<int32>(leaves.shape().dim_size(0));
    const int32 num_splits = static_cast<int32>(
        split_features.shape().dim_size(1));
    const int32 num_features = static_cast<int32>(
        input_data.shape().dim_size(1));
    const int32 num_accumulators = static_cast<int32>(
        split_features.shape().dim_size(0));

    std::unordered_map<int32, std::set<int32>> accumulator_to_leaves;

    // The first pass just calculates num_output_accumulators.
    for (int32 i = 0; i < num_data; i++) {
      const int32 leaf = internal::SubtleMustCopy(leaves_vec(i));
      OP_REQUIRES(context, FastBoundsCheck(leaf, node_map.size()),
                  errors::InvalidArgument("leaf not in valid range."))
      const int32 accumulator = internal::SubtleMustCopy(node_map(leaf));

      // Check for non-fertile node or fertile node that is already
      // initialized.
      if (accumulator >= 0 &&
          !IsAllInitialized(
              split_features.Slice(accumulator, accumulator + 1))) {
        accumulator_to_leaves[accumulator].insert(i);
      }
    }

    // Now we can allocate the outputs.
    int32 num_output_accumulators = static_cast<int32>(
        accumulator_to_leaves.size());
    VLOG(1) << "num output accumulators = " << num_output_accumulators;
    Tensor* accumulators_tensor = nullptr;
    TensorShape accumulators_shape;
    accumulators_shape.AddDim(num_output_accumulators);
    OP_REQUIRES_OK(context,
                   context->allocate_output(0, accumulators_shape,
                                            &accumulators_tensor));
    auto accumulators_flat = accumulators_tensor->tensor<int32, 1>();

    Tensor* new_split_feature_rows_tensor = nullptr;
    TensorShape new_split_feature_rows_shape;
    new_split_feature_rows_shape.AddDim(num_output_accumulators);
    new_split_feature_rows_shape.AddDim(num_splits);
    OP_REQUIRES_OK(context,
                   context->allocate_output(1, new_split_feature_rows_shape,
                                            &new_split_feature_rows_tensor));
    auto new_split_feature_rows_flat =
        new_split_feature_rows_tensor->tensor<int32, 2>();

    Tensor* new_split_threshold_rows_tensor = nullptr;
    TensorShape new_split_threshold_rows_shape;
    new_split_threshold_rows_shape.AddDim(num_output_accumulators);
    new_split_threshold_rows_shape.AddDim(num_splits);
    OP_REQUIRES_OK(context,
                   context->allocate_output(2, new_split_threshold_rows_shape,
                                            &new_split_threshold_rows_tensor));
    auto new_split_threshold_rows_flat =
        new_split_threshold_rows_tensor->tensor<float, 2>();

    // The second pass fills out the outputs.
    int output_slot = 0;
    for (const auto& active : accumulator_to_leaves) {
      const int32 accumulator = active.first;
      OP_REQUIRES(context, FastBoundsCheck(accumulator, num_accumulators),
                  errors::InvalidArgument("accumulator not in valid range."))
      const std::set<int32> inputs_for_accumulator = active.second;
      VLOG(1) << "Accumulator " << accumulator
                  << " gets new output slot " << output_slot;
      accumulators_flat(output_slot) = accumulator;

      // scatter_update updates entire rows, so we first copy the existing
      // rows into the output tensors, and then write over the values we
      // want to change.
      for (int split = 0; split < num_splits; split++) {
        new_split_feature_rows_flat(output_slot, split) =
            features(accumulator, split);
        new_split_threshold_rows_flat(output_slot, split) =
            thresholds(accumulator, split);
      }

      for (const int32 i : inputs_for_accumulator) {
        VLOG(2) << "Looking at data # " << i;

        int32 num_inits = split_initializations_per_input_;
        for (int split = 0; split < num_splits && num_inits > 0; split++) {
          if (new_split_feature_rows_flat(output_slot, split) < 0) {
            VLOG(1) << "Over-writing @ " << output_slot << "," << split;
            const int32 index = rng_->Uniform(num_features);
            new_split_feature_rows_flat(output_slot, split) = index;
            new_split_threshold_rows_flat(output_slot, split) =
                inputs(i, index);
            --num_inits;
          }
        }
      }
      ++output_slot;
    }
  }

 private:
  int32 split_initializations_per_input_;
  int32 split_sampling_random_seed_;
  std::unique_ptr<random::PhiloxRandom> single_rand_;
  std::unique_ptr<random::SimplePhilox> rng_;
};

REGISTER_KERNEL_BUILDER(Name("SampleInputs").Device(DEVICE_CPU), SampleInputs);

}  // namespace tensorflow
