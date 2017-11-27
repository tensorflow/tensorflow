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
#include "tensorflow/contrib/decision_trees/proto/generic_tree_model.pb.h"
#include "tensorflow/contrib/tensor_forest/kernels/v4/test_utils.h"
#include "tensorflow/core/platform/test.h"

namespace tensorflow {
namespace {

using tensorflow::tensorforest::InequalityDecisionNodeEvaluator;
using tensorflow::tensorforest::MatchingValuesDecisionNodeEvaluator;
using tensorflow::tensorforest::ObliqueInequalityDecisionNodeEvaluator;
using tensorflow::decision_trees::InequalityTest;
using tensorflow::decision_trees::MatchingValuesTest;

TEST(InequalityDecisionNodeEvaluatorTest, TestLessOrEqual) {
  InequalityTest test;
  test.mutable_feature_id()->mutable_id()->set_value("0");
  test.mutable_threshold()->set_float_value(3.0);
  test.set_type(InequalityTest::LESS_OR_EQUAL);
  std::unique_ptr<InequalityDecisionNodeEvaluator> eval(
      new InequalityDecisionNodeEvaluator(test, 0, 1));

  std::unique_ptr<tensorflow::tensorforest::TensorDataSet> dataset(
      new tensorflow::tensorforest::TestableDataSet(
          {0.0, 1.0, 2.0, 3.0, 4.0, 5.0}, 1));

  ASSERT_EQ(eval->Decide(dataset, 2), 0);
  ASSERT_EQ(eval->Decide(dataset, 3), 0);
  ASSERT_EQ(eval->Decide(dataset, 4), 1);
}

TEST(InequalityDecisionNodeEvaluatorTest, TestStrictlyLess) {
  InequalityTest test;
  test.mutable_feature_id()->mutable_id()->set_value("0");
  test.mutable_threshold()->set_float_value(3.0);
  test.set_type(InequalityTest::LESS_THAN);
  std::unique_ptr<InequalityDecisionNodeEvaluator> eval(
      new InequalityDecisionNodeEvaluator(test, 0, 1));

  std::unique_ptr<tensorflow::tensorforest::TensorDataSet> dataset(
      new tensorflow::tensorforest::TestableDataSet(
          {0.0, 1.0, 2.0, 3.0, 4.0, 5.0}, 1));

  ASSERT_EQ(eval->Decide(dataset, 2), 0);
  ASSERT_EQ(eval->Decide(dataset, 3), 1);
  ASSERT_EQ(eval->Decide(dataset, 4), 1);
}

TEST(MatchingDecisionNodeEvaluatorTest, Basic) {
  MatchingValuesTest test;
  test.mutable_feature_id()->mutable_id()->set_value("0");
  test.add_value()->set_float_value(3.0);
  test.add_value()->set_float_value(5.0);

  std::unique_ptr<MatchingValuesDecisionNodeEvaluator> eval(
      new MatchingValuesDecisionNodeEvaluator(test, 0, 1));

  std::unique_ptr<tensorflow::tensorforest::TensorDataSet> dataset(
      new tensorflow::tensorforest::TestableDataSet(
          {0.0, 1.0, 2.0, 3.0, 4.0, 5.0}, 1));

  ASSERT_EQ(eval->Decide(dataset, 2), 1);
  ASSERT_EQ(eval->Decide(dataset, 3), 0);
  ASSERT_EQ(eval->Decide(dataset, 4), 1);
  ASSERT_EQ(eval->Decide(dataset, 5), 0);
}

TEST(MatchingDecisionNodeEvaluatorTest, Inverse) {
  MatchingValuesTest test;
  test.mutable_feature_id()->mutable_id()->set_value("0");
  test.add_value()->set_float_value(3.0);
  test.add_value()->set_float_value(5.0);
  test.set_inverse(true);

  std::unique_ptr<MatchingValuesDecisionNodeEvaluator> eval(
      new MatchingValuesDecisionNodeEvaluator(test, 0, 1));

  std::unique_ptr<tensorflow::tensorforest::TensorDataSet> dataset(
      new tensorflow::tensorforest::TestableDataSet(
          {0.0, 1.0, 2.0, 3.0, 4.0, 5.0}, 1));

  ASSERT_EQ(eval->Decide(dataset, 2), 0);
  ASSERT_EQ(eval->Decide(dataset, 3), 1);
  ASSERT_EQ(eval->Decide(dataset, 4), 0);
  ASSERT_EQ(eval->Decide(dataset, 5), 1);
}

TEST(ObliqueDecisionNodeEvaluatorTest, Basic) {
  InequalityTest test;
  auto* feat1 = test.mutable_oblique()->add_features();
  feat1->mutable_id()->set_value("0");
  test.mutable_oblique()->add_weights(1.0);
  auto* feat2 = test.mutable_oblique()->add_features();
  feat2->mutable_id()->set_value("1");
  test.mutable_oblique()->add_weights(1.0);

  test.mutable_threshold()->set_float_value(3.0);
  test.set_type(InequalityTest::LESS_OR_EQUAL);

  std::unique_ptr<ObliqueInequalityDecisionNodeEvaluator> eval(
      new ObliqueInequalityDecisionNodeEvaluator(test, 0, 1));

  std::unique_ptr<tensorflow::tensorforest::TensorDataSet> dataset(
      new tensorflow::tensorforest::TestableDataSet(
          {0.0, 1.0, 2.0, 3.0, 4.0, 5.0}, 2));

  ASSERT_EQ(eval->Decide(dataset, 0), 0);
  ASSERT_EQ(eval->Decide(dataset, 1), 1);
}

}  // namespace
}  // namespace tensorflow

