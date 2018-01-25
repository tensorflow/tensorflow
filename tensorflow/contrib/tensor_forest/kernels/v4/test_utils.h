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
#ifndef TENSORFLOW_CONTRIB_TENSOR_FOREST_KERNELS_V4_TEST_UTILS_H_
#define TENSORFLOW_CONTRIB_TENSOR_FOREST_KERNELS_V4_TEST_UTILS_H_
#include "tensorflow/contrib/tensor_forest/kernels/v4/input_data.h"
#include "tensorflow/contrib/tensor_forest/kernels/v4/input_target.h"

namespace tensorflow {
namespace tensorforest {

class TestableInputTarget : public StoredInputTarget<std::vector<float>> {
 public:
  TestableInputTarget(const std::vector<float>& t, const std::vector<float>& w,
                      int num_t)
      : StoredInputTarget(new std::vector<float>(t), new std::vector<float>(w),
                          num_t) {}

  int NumItems() const {
    return target_->size();
  }

  int32 GetTargetAsClassIndex(int example_index,
                              int target_index) const override {
    return static_cast<int32>(
        GetTargetAsContinuous(example_index, target_index));
  }

  float GetTargetWeight(int example_index) const override {
    const size_t num_weights = weight_->size();
    return num_weights > 0 && example_index < num_weights
               ? (*weight_)[example_index]
               : 1.0;
  }

  float GetTargetAsContinuous(int example_index,
                              int target_index) const override {
    QCHECK_LT(target_index, num_targets_);
    return (*target_)[example_index * num_targets_ + target_index];
  }
};


class TestableDataSet : public TensorDataSet {
 public:
  TestableDataSet(const std::vector<float>& data, int num_features)
      : TensorDataSet(TensorForestDataSpec(), 11),
        num_features_(num_features),
        data_(data) {}

  float GetExampleValue(int example, int32 feature_id) const override {
    return data_[example * num_features_ + feature_id];
  }

 protected:
  int num_features_;
  std::vector<float> data_;
};

}  // namespace tensorforest
}  // namespace tensorflow

#endif  // TENSORFLOW_CONTRIB_TENSOR_FOREST_KERNELS_V4_TEST_UTILS_H_
