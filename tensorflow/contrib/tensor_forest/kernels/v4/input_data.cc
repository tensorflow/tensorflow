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
#include "tensorflow/contrib/tensor_forest/kernels/v4/input_data.h"
#include "tensorflow/contrib/decision_trees/proto/generic_tree_model_extensions.pb.h"
#include "tensorflow/contrib/tensor_forest/kernels/tree_utils.h"
#include "tensorflow/core/lib/strings/numbers.h"

namespace tensorflow {
namespace tensorforest {
namespace {

bool DecideInequalityTest(const decision_trees::InequalityTest& test,
                          float value) {
  float bias = test.threshold().float_value();
  switch (test.type()) {
    case decision_trees::InequalityTest::LESS_OR_EQUAL:
      return value <= bias;

    case decision_trees::InequalityTest::LESS_THAN:
      return value < bias;

    case decision_trees::InequalityTest::GREATER_OR_EQUAL:
      return value >= bias;

    case decision_trees::InequalityTest::GREATER_THAN:
      return value > bias;

    default:
      return false;
  }
}

bool DecideMatchingValuesTest(const decision_trees::MatchingValuesTest& test,
                              float value) {
  for (const decision_trees::Value& test_value : test.value()) {
    if (test_value.float_value() == value) {
      return true;
    }
  }
  return false;
}

}  // namespace

bool TensorDataSet::Decide(const decision_trees::BinaryNode& node,
                           int example) const {
  // TODO(gilberth): Support missing values.
  float val = 0;
  const auto& test = node.inequality_left_child_test();

  if (test.has_oblique()) {
    for (int i = 0; i < test.oblique().features_size(); ++i) {
      val += test.oblique().weights(i) *
             GetExampleValue(example, test.oblique().features(i));
    }
  } else {
    val = GetExampleValue(example, test.feature_id());
  }

  if (node.has_inequality_left_child_test()) {
    return DecideInequalityTest(node.inequality_left_child_test(), val);
  } else {
    decision_trees::MatchingValuesTest test;
    if (node.custom_left_child_test().UnpackTo(&test)) {
      return DecideMatchingValuesTest(test, val);
    } else {
      return false;
    }
  }
}

float TensorDataSet::GetExampleValue(
    int example, const decision_trees::FeatureId& feature_id) const {
  int32 feature;
  safe_strto32(feature_id.id().value(), &feature);
  if (feature >= input_spec_.dense_features_size()) {
    return FindSparseValue(*sparse_indices_, *sparse_values_, example, feature);
  } else {
    return (*dense_data_)(example, feature);
  }
}

float TensorDataSet::GetExampleValue(int example, int32 feature_id) const {
  if (feature_id >= input_spec_.dense_features_size()) {
    return FindSparseValue(*sparse_indices_, *sparse_values_, example,
                           feature_id);
  } else {
    return (*dense_data_)(example, feature_id);
  }
}

void TensorDataSet::set_input_tensors(const Tensor& dense,
                                      const Tensor& sparse_indices,
                                      const Tensor& sparse_values,
                                      const Tensor& sparse_shape) {
  if (dense.shape().dims() == 2) {
    dense_data_.reset(new DenseStorageType(dense.tensor<float, 2>()));
  }
  if (sparse_indices.shape().dims() == 2) {
    sparse_indices_.reset(new SparseIndicesStorageType(
        sparse_indices.tensor<int64, 2>()));
    sparse_values_.reset(new SparseValuesStorageType(
        sparse_values.tensor<float, 1>()));
    sparse_batch_size_ = sparse_shape.tensor<int64, 1>()(0);
  }
  original_dense_tensor_ = dense;
}

void TensorDataSet::RandomSample(int example,
                                 decision_trees::FeatureId* feature_id,
                                 float* bias, int* type) const {
  int32 num_total_features = input_spec_.dense_features_size();
  int64 sparse_input_start;
  if (sparse_indices_ != nullptr) {
    const int32 num_sparse = tensorforest::GetNumSparseFeatures(
        *sparse_indices_, example, &sparse_input_start);
    if (sparse_input_start >= 0) {
      num_total_features += num_sparse;
    }
  }
  int rand_feature = rng_->Uniform(num_total_features);
  if (rand_feature < available_features_.size()) {  // it's dense.
    *feature_id = available_features_[rand_feature];
    *type = input_spec_.GetDenseFeatureType(rand_feature);
  } else {
    const int32 sparse_index =
        sparse_input_start + rand_feature - input_spec_.dense_features_size();
    const int32 saved_index =
        (*sparse_indices_)(sparse_index, 1) + input_spec_.dense_features_size();
    *feature_id = decision_trees::FeatureId();
    feature_id->mutable_id()->set_value(strings::StrCat(saved_index));

    // TODO(gilberth): Remove this shortcut when different sparse types are
    // allowed.
    *type = input_spec_.sparse(0).original_type();
  }

  *bias = GetExampleValue(example, *feature_id);
}

}  // namespace tensorforest
}  // namespace tensorflow
