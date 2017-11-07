// Copyright 2017 The TensorFlow Authors. All Rights Reserved.
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
#ifndef THIRD_PARTY_TENSORFLOW_CONTRIB_TENSOR_FOREST_KERNELS_V4_INPUT_DATA_H_
#define THIRD_PARTY_TENSORFLOW_CONTRIB_TENSOR_FOREST_KERNELS_V4_INPUT_DATA_H_
#include <ctime>
#include <unordered_map>
#include "google/protobuf/any.pb.h"
#include "google/protobuf/wrappers.pb.h"
#include "tensorflow/contrib/decision_trees/proto/generic_tree_model.pb.h"
#include "tensorflow/contrib/tensor_forest/kernels/data_spec.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_types.h"
#include "tensorflow/core/lib/random/philox_random.h"
#include "tensorflow/core/lib/random/simple_philox.h"

namespace tensorflow {
namespace tensorforest {

typedef TTypes<const float, 2>::ConstTensor DenseStorageType;
typedef TTypes<const int64, 2>::ConstTensor SparseIndicesStorageType;
typedef TTypes<const float, 1>::ConstTensor SparseValuesStorageType;

class TensorDataSet {
 public:
  TensorDataSet(const tensorforest::TensorForestDataSpec& input_spec,
                int32 seed)
      : dense_data_(nullptr),
        sparse_indices_(nullptr),
        sparse_values_(nullptr),
        input_spec_(input_spec),
        split_sampling_random_seed_(seed) {
    int column_count = 0;
    for (int i = 0; i < input_spec_.dense_size(); ++i) {
      for (int j = 0; j < input_spec_.dense(i).size(); ++j) {
        decision_trees::FeatureId id;
        id.mutable_id()->set_value(strings::StrCat(column_count));
        available_features_.push_back(id);
        ++column_count;
      }
    }

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
  virtual ~TensorDataSet() {}

  void set_input_tensors(const Tensor& dense, const Tensor& sparse_indices,
                         const Tensor& sparse_values,
                         const Tensor& sparse_shape);

  float get_input_value(int offset, int col) {
    return (*dense_data_)(offset, col);
  }

  int NumItems() const {
    if (dense_data_ != nullptr) {
      return dense_data_->dimensions()[0];
    } else if (sparse_indices_ != nullptr) {
      return sparse_batch_size_;
    } else {
      return 0;
    }
  }

  // This looks up a value by example and int32_id, which is much faster than
  // GetFeature.
  float GetExampleValue(int example,
                        const decision_trees::FeatureId& feature_id) const;

  // Same as overload with FeatureId, but if you already have the feature as
  // an int32 you can avoid the atoi32.
  virtual float GetExampleValue(int example, int32 feature_id) const;

  int num_features() {
    return available_features_.size();
  }

  const Tensor& original_tensor() const { return original_dense_tensor_; }

  bool Decide(const decision_trees::BinaryNode& node, int example) const;

  // Randomly samples a feature from example, returns its id in feature_name,
  // the value in bias, and it's type from input_spec in type.
  void RandomSample(int example, decision_trees::FeatureId* feature_name,
                    float* bias, int* type) const;

 private:
  std::unique_ptr<DenseStorageType> dense_data_;
  std::unique_ptr<SparseIndicesStorageType> sparse_indices_;
  std::unique_ptr<SparseValuesStorageType> sparse_values_;
  int sparse_batch_size_;

  Tensor original_dense_tensor_;
  const tensorforest::TensorForestDataSpec input_spec_;
  std::vector<decision_trees::FeatureId> available_features_;

  int32 split_sampling_random_seed_;
  std::unique_ptr<random::PhiloxRandom> single_rand_;
  std::unique_ptr<random::SimplePhilox> rng_;
};
}  // namespace tensorforest
}  // namespace tensorflow

#endif  // THIRD_PARTY_TENSORFLOW_CONTRIB_TENSOR_FOREST_KERNELS_V4_INPUT_DATA_H_
