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
#include "tensorflow/contrib/tensor_forest/kernels/v4/decision_node_evaluator.h"
#include "tensorflow/core/lib/strings/numbers.h"

namespace tensorflow {
namespace tensorforest {

std::unique_ptr<DecisionNodeEvaluator> CreateDecisionNodeEvaluator(
    const decision_trees::TreeNode& node) {
  const decision_trees::BinaryNode& bnode = node.binary_node();
  return CreateBinaryDecisionNodeEvaluator(bnode, bnode.left_child_id().value(),
                                           bnode.right_child_id().value());
}

std::unique_ptr<DecisionNodeEvaluator> CreateBinaryDecisionNodeEvaluator(
    const decision_trees::BinaryNode& bnode, int32 left, int32 right) {
  if (bnode.has_inequality_left_child_test()) {
    const auto& test = bnode.inequality_left_child_test();
    if (test.has_oblique()) {
      return std::unique_ptr<ObliqueInequalityDecisionNodeEvaluator>(
          new ObliqueInequalityDecisionNodeEvaluator(test, left, right));
    } else {
      return std::unique_ptr<InequalityDecisionNodeEvaluator>(
          new InequalityDecisionNodeEvaluator(test, left, right));
    }
  } else {
    decision_trees::MatchingValuesTest test;
    if (bnode.custom_left_child_test().UnpackTo(&test)) {
      return std::unique_ptr<MatchingValuesDecisionNodeEvaluator>(
          new MatchingValuesDecisionNodeEvaluator(test, left, right));
    } else {
      LOG(ERROR) << "Unknown split test: " << bnode.DebugString();
      return nullptr;
    }
  }
}

InequalityDecisionNodeEvaluator::InequalityDecisionNodeEvaluator(
    const decision_trees::InequalityTest& test, int32 left, int32 right)
    : BinaryDecisionNodeEvaluator(left, right) {
  safe_strto32(test.feature_id().id().value(), &feature_num_);
  threshold_ = test.threshold().float_value();
  include_equals_ =
      test.type() == decision_trees::InequalityTest::LESS_OR_EQUAL;
}

int32 InequalityDecisionNodeEvaluator::Decide(
    const std::unique_ptr<TensorDataSet>& dataset, int example) const {
  const float val = dataset->GetExampleValue(example, feature_num_);
  if (val < threshold_ || (include_equals_ && val == threshold_)) {
    return left_child_id_;
  } else {
    return right_child_id_;
  }
}

ObliqueInequalityDecisionNodeEvaluator::ObliqueInequalityDecisionNodeEvaluator(
    const decision_trees::InequalityTest& test, int32 left, int32 right)
    : BinaryDecisionNodeEvaluator(left, right) {
  for (int i = 0; i < test.oblique().features_size(); ++i) {
    int32 val;
    safe_strto32(test.oblique().features(i).id().value(), &val);
    feature_num_.push_back(val);
    feature_weights_.push_back(test.oblique().weights(i));
  }
  threshold_ = test.threshold().float_value();
}

int32 ObliqueInequalityDecisionNodeEvaluator::Decide(
    const std::unique_ptr<TensorDataSet>& dataset, int example) const {
  float val = 0;
  for (int i = 0; i < feature_num_.size(); ++i) {
    val += feature_weights_[i] *
           dataset->GetExampleValue(example, feature_num_[i]);
  }

  if (val <= threshold_) {
    return left_child_id_;
  } else {
    return right_child_id_;
  }
}

MatchingValuesDecisionNodeEvaluator::MatchingValuesDecisionNodeEvaluator(
    const decision_trees::MatchingValuesTest& test, int32 left, int32 right)
    : BinaryDecisionNodeEvaluator(left, right) {
  safe_strto32(test.feature_id().id().value(), &feature_num_);
  for (const auto& val : test.value()) {
    values_.push_back(val.float_value());
  }
  inverse_ = test.inverse();
}

int32 MatchingValuesDecisionNodeEvaluator::Decide(
    const std::unique_ptr<TensorDataSet>& dataset, int example) const {
  const float val = dataset->GetExampleValue(example, feature_num_);
  for (float testval : values_) {
    if (val == testval) {
      return inverse_ ? right_child_id_ : left_child_id_;
    }
  }

  return inverse_ ? left_child_id_ : right_child_id_;
}

}  // namespace tensorforest
}  // namespace tensorflow
