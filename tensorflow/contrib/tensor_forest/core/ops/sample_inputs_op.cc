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
// SampleInputs initializes candidate splits/threshold values randomly
// from incoming data for not-yet-initialized fertile nodes.
#include <ctime>
#include <unordered_map>
#include <set>

#include "tensorflow/contrib/tensor_forest/core/ops/tree_utils.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "tensorflow/core/kernels/bounds_check.h"
#include "tensorflow/core/lib/random/distribution_sampler.h"
#include "tensorflow/core/lib/random/philox_random.h"
#include "tensorflow/core/lib/random/simple_philox.h"
#include "tensorflow/core/platform/logging.h"

namespace tensorflow {

using shape_inference::DimensionHandle;
using shape_inference::InferenceContext;
using shape_inference::ShapeHandle;

using tensorforest::CheckTensorBounds;
using tensorforest::IsAllInitialized;


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

  // Returns true if index and val were successfully set.
  template <typename T>
  bool GetRandomFeatureDense(const T& inputs, int32 num_features,
                             int32 input_index, int32* index, float* val) {
    *index = rng_->Uniform(num_features);
    *val = inputs(input_index, *index);
    return true;
  }

  // Returns true if index and val were successfully set.
  template <typename T1, typename T2>
  bool GetRandomFeatureSparse(const T1& sparse_indices, const T2& sparse_values,
                              int32 input_index, int32* index, float* val) {
    int32 low = 0;
    int32 high = sparse_values.dimension(0);
    while (low < high) {
      int32 vi = low + rng_->Uniform(high - low);
      int64 i = internal::SubtleMustCopy(sparse_indices(vi, 0));
      if (i == input_index) {
        int64 ind = internal::SubtleMustCopy(sparse_indices(vi, 1));
        CHECK(ind < kint32max);
        *index = static_cast<int32>(ind);
        *val = sparse_values(vi);
        return true;
      }
      if (i < input_index) {
        low = vi + 1;
      } else {
        high = vi;
      }
    }

    // If we get here, an example was empty.  That's unfortunate, but we try
    // to continue anyway by trying to look at another example.
    LOG(WARNING) << "Could not find any values for input " << input_index
                 << " inside sparse_input_indices";
    return false;
  }

  // increment_input implements a "++" operation for the situation when
  // you want to do something n times on an underlying iterator.
  // In an ideal world, this would be a built-in iterator adaptor.
  template <typename T>
  static void increment_input(const int n, T* it, int* count) {
    *count += 1;
    if (*count == n) {
      *count = 0;
      (*it)++;
    }
  }

  void Compute(OpKernelContext* context) override {
    const Tensor& input_data = context->input(0);
    const Tensor& sparse_input_indices = context->input(1);
    const Tensor& sparse_input_values = context->input(2);
    const Tensor& sparse_input_shape = context->input(3);
    const Tensor& input_weights = context->input(4);
    const Tensor& node_to_accumulator = context->input(5);
    const Tensor& leaves = context->input(6);
    const Tensor& split_features = context->input(7);
    const Tensor& split_thresholds = context->input(8);

    bool sparse_input = (sparse_input_indices.shape().dims() == 2);

    bool have_weights = (input_weights.shape().dim_size(0) > 0);

    if (sparse_input) {
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
      if (have_weights) {
        OP_REQUIRES(context, sparse_input_shape.unaligned_flat<int64>()(0) ==
                                 input_weights.shape().dim_size(0),
                    errors::InvalidArgument(
                        "sparse_input_values and input_weights should agree "
                        "on the number of inputs"));
      }
    } else {
      OP_REQUIRES(context, input_data.shape().dims() == 2,
                  errors::InvalidArgument(
                  "input_data should be two-dimensional"));
      if (have_weights) {
        OP_REQUIRES(context, input_data.shape().dim_size(0) ==
                                 input_weights.shape().dim_size(0),
                    errors::InvalidArgument(
                        "input_data and input_weights should agree on the "
                        "number of inputs"));
      }
    }

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
    if (!CheckTensorBounds(context, sparse_input_indices)) return;
    if (!CheckTensorBounds(context, sparse_input_values)) return;
    if (!CheckTensorBounds(context, sparse_input_shape)) return;
    if (!CheckTensorBounds(context, input_weights)) return;
    if (!CheckTensorBounds(context, node_to_accumulator)) return;
    if (!CheckTensorBounds(context, leaves)) return;
    if (!CheckTensorBounds(context, split_features)) return;
    if (!CheckTensorBounds(context, split_thresholds)) return;

    int32 num_features;
    std::function<bool(int32, int32*, float*)> get_random_feature;
    // TODO(thomaswc): Figure out a way to avoid calling .vec, etc. over and
    // over again
    if (sparse_input) {
      num_features = sparse_input_shape.unaligned_flat<int64>()(1);
      get_random_feature = [&sparse_input_indices, &sparse_input_values, this](
          int32 input_index, int32* index, float* val) -> bool {
        const auto sparse_indices = sparse_input_indices.matrix<int64>();
        const auto sparse_values = sparse_input_values.vec<float>();
        return GetRandomFeatureSparse(sparse_indices, sparse_values,
                                      input_index, index, val);
      };
    } else {
      num_features = static_cast<int32>(input_data.shape().dim_size(1));
      get_random_feature = [&input_data, num_features, this](
          int32 input_index, int32* index, float* val) -> bool {
        const auto inputs = input_data.tensor<float, 2>();
        return GetRandomFeatureDense(inputs, num_features, input_index, index,
                                     val);
      };
    }

    const auto leaves_vec = leaves.unaligned_flat<int32>();
    const auto node_map = node_to_accumulator.unaligned_flat<int32>();
    const auto features = split_features.tensor<int32, 2>();
    const auto thresholds = split_thresholds.tensor<float, 2>();
    const auto weights = input_weights.unaligned_flat<float>();

    const int32 num_data = static_cast<int32>(leaves.shape().dim_size(0));
    const int32 num_splits = static_cast<int32>(
        split_features.shape().dim_size(1));
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

      auto it = inputs_for_accumulator.begin();
      int input_used_count = 0;
      for (int split = 0;
           split < num_splits && it != inputs_for_accumulator.end(); split++) {
        if (new_split_feature_rows_flat(output_slot, split) < 0) {
          if (have_weights) {
            // If we have weights, we probabilistically reject inputs with
            // low weight.  Which means we might have to look at a bunch of
            // inputs -- maybe even all of them -- to fill this slot.
            while (it != inputs_for_accumulator.end()) {
              float w = weights(*it);
              if (rng_->RandFloat() <= w) {
                break;
              }
              increment_input(split_initializations_per_input_, &it,
                              &input_used_count);
            }
            if (it == inputs_for_accumulator.end()) {
              break;
            }
          }
          int32 index;
          float val;
          const bool success = get_random_feature(*it, &index, &val);
          CHECK(index >= 0) << "sample inputs chose negative feature: "
                            << index;
          increment_input(split_initializations_per_input_, &it,
                          &input_used_count);
          if (success) {
            VLOG(1) << "Over-writing @ " << output_slot << "," << split;
            new_split_feature_rows_flat(output_slot, split) = index;
            new_split_threshold_rows_flat(output_slot, split) = val;
          } else {
            LOG(ERROR) << "get_random_feature failed, bailing on output for "
                       << "accumulator " << accumulator;
            break;
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
