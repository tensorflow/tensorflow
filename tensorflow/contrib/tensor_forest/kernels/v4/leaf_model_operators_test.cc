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
#include "tensorflow/contrib/tensor_forest/kernels/v4/leaf_model_operators.h"
#include "tensorflow/contrib/decision_trees/proto/generic_tree_model.pb.h"
#include "tensorflow/contrib/tensor_forest/kernels/v4/test_utils.h"
#include "tensorflow/contrib/tensor_forest/proto/tensor_forest_params.pb.h"
#include "tensorflow/core/platform/protobuf.h"
#include "tensorflow/core/platform/test.h"
#include "tensorflow/core/platform/types.h"

namespace tensorflow {
namespace {

using tensorflow::decision_trees::Leaf;
using tensorflow::tensorforest::DenseClassificationLeafModelOperator;
using tensorflow::tensorforest::LeafModelOperator;
using tensorflow::tensorforest::SparseClassificationLeafModelOperator;
using tensorflow::tensorforest::SparseOrDenseClassificationLeafModelOperator;
using tensorflow::tensorforest::LeafStat;
using tensorflow::tensorforest::RegressionLeafModelOperator;
using tensorflow::tensorforest::TestableInputTarget;
using tensorflow::tensorforest::TensorForestParams;

const int32 kNumClasses = 3;

constexpr char kRegressionStatProto[] =
  "weight_sum: 3 "
  "regression { "
  "mean_output { "
    "value { "
    "  float_value: 27 "
    "} "
    "value { "
    "  float_value: 282 "
    "} "
    "value { "
    "  float_value: 10 "
    "} "
  "} "
  "mean_output_squares { "
    "value {"
    "  float_value: 245"
    "}"
    "value {"
    "  float_value: 26564"
    "}"
    "value {"
    "  float_value: 46"
    "}"
  "}"
"}";

void TestClassificationNormalUse(const std::unique_ptr<LeafModelOperator>& op) {
  std::unique_ptr<LeafStat> leaf(new LeafStat);
  op->InitModel(leaf.get());

  Leaf l;
  op->ExportModel(*leaf, &l);

  // Make sure it was initialized correctly.
  for (int i = 0; i < kNumClasses; ++i) {
    EXPECT_EQ(op->GetOutputValue(l, i), 0);
  }

  std::vector<float> labels = {1, 0, 1};
  std::vector<float> weights = {2.3, 20.3, 1.1};
  std::unique_ptr<TestableInputTarget> target(
      new TestableInputTarget(labels, weights, 1));

  // Update and check value.
  op->UpdateModel(leaf.get(), target.get(), 0);
  op->UpdateModel(leaf.get(), target.get(), 1);
  op->UpdateModel(leaf.get(), target.get(), 2);

  op->ExportModel(*leaf, &l);
  EXPECT_FLOAT_EQ(op->GetOutputValue(l, 1), 3.4);
}


TEST(DenseLeafModelOperatorsTest, NormalUse) {
  TensorForestParams params;
  params.set_num_outputs(kNumClasses);
  std::unique_ptr<LeafModelOperator> op(
      new DenseClassificationLeafModelOperator(params));
  TestClassificationNormalUse(op);
}

TEST(SparseLeafModelOperatorsTest, NormalUse) {
  TensorForestParams params;
  params.set_num_outputs(kNumClasses);
  std::unique_ptr<LeafModelOperator> op(
      new SparseClassificationLeafModelOperator(params));
  TestClassificationNormalUse(op);
}

TEST(DenseLeafModelOperatorsTest, InitWithExisting) {
  TensorForestParams params;
  params.set_num_outputs(kNumClasses);
  std::unique_ptr<LeafModelOperator> op(
      new DenseClassificationLeafModelOperator(params));

  std::unique_ptr<LeafStat> stat(new LeafStat);
  stat->mutable_classification()
      ->mutable_dense_counts()
      ->add_value()
      ->set_float_value(1.1);
  stat->mutable_classification()
      ->mutable_dense_counts()
      ->add_value()
      ->set_float_value(2.2);
  stat->mutable_classification()
      ->mutable_dense_counts()
      ->add_value()
      ->set_float_value(3.3);

  std::unique_ptr<Leaf> leaf(new Leaf);

  op->ExportModel(*stat, leaf.get());

  // Make sure it was initialized correctly.
  EXPECT_EQ(leaf->vector().value_size(), kNumClasses);
  EXPECT_FLOAT_EQ(op->GetOutputValue(*leaf, 0), 1.1);
  EXPECT_FLOAT_EQ(op->GetOutputValue(*leaf, 1), 2.2);
  EXPECT_FLOAT_EQ(op->GetOutputValue(*leaf, 2), 3.3);
}

TEST(SparseOrDenseClassificationLeafModelOperator, InitWithExisting) {
  TensorForestParams params;
  params.set_num_outputs(kNumClasses);
  std::unique_ptr<LeafModelOperator> op(
      new SparseOrDenseClassificationLeafModelOperator(params));

  std::unique_ptr<LeafStat> stat(new LeafStat);
  (*stat->mutable_classification()
        ->mutable_sparse_counts()
        ->mutable_sparse_value())[0]
      .set_float_value(1.1);
  (*stat->mutable_classification()
        ->mutable_sparse_counts()
        ->mutable_sparse_value())[1]
      .set_float_value(2.2);
  (*stat->mutable_classification()
        ->mutable_sparse_counts()
        ->mutable_sparse_value())[2]
      .set_float_value(3.3);

  std::unique_ptr<Leaf> leaf(new Leaf);

  op->ExportModel(*stat, leaf.get());

  // Make sure it was initialized correctly.
  EXPECT_FLOAT_EQ(op->GetOutputValue(*leaf, 0), 1.1);
  EXPECT_FLOAT_EQ(op->GetOutputValue(*leaf, 1), 2.2);
  EXPECT_FLOAT_EQ(op->GetOutputValue(*leaf, 2), 3.3);
}

TEST(SparseLeafModelOperatorsTest, InitWithExisting) {
  TensorForestParams params;
  params.set_num_outputs(kNumClasses);
  std::unique_ptr<LeafModelOperator> op(
      new SparseClassificationLeafModelOperator(params));
  std::unique_ptr<LeafStat> stat(new LeafStat);
  (*stat->mutable_classification()
        ->mutable_sparse_counts()
        ->mutable_sparse_value())[0]
      .set_float_value(1.1);
  (*stat->mutable_classification()
        ->mutable_sparse_counts()
        ->mutable_sparse_value())[1]
      .set_float_value(2.2);
  (*stat->mutable_classification()
        ->mutable_sparse_counts()
        ->mutable_sparse_value())[2]
      .set_float_value(3.3);

  std::unique_ptr<Leaf> leaf(new Leaf);

  op->ExportModel( *stat, leaf.get());

  // Make sure it was initialized correctly.
  EXPECT_FLOAT_EQ(op->GetOutputValue(*leaf, 0), 1.1);
  EXPECT_FLOAT_EQ(op->GetOutputValue(*leaf, 1), 2.2);
  EXPECT_FLOAT_EQ(op->GetOutputValue(*leaf, 2), 3.3);

  // check default value.
  EXPECT_FLOAT_EQ(op->GetOutputValue(*leaf, 100), 0);
  EXPECT_EQ(leaf->sparse_vector().sparse_value().size(), kNumClasses);
}


TEST(RegressionLeafModelOperatorsTest, NormalUse) {
  TensorForestParams params;
  params.set_num_outputs(kNumClasses);
  std::unique_ptr<LeafModelOperator> op(
      new RegressionLeafModelOperator(params));

  std::unique_ptr<LeafStat> stat(new LeafStat());
  const string contents(kRegressionStatProto);
  ::tensorflow::protobuf::TextFormat::ParseFromString(contents, stat.get());

  std::unique_ptr<Leaf> leaf(new Leaf);
  op->ExportModel(*stat, leaf.get());

  // Make sure it was initialized correctly.
  EXPECT_FLOAT_EQ(op->GetOutputValue(*leaf, 0), 9);
  EXPECT_FLOAT_EQ(op->GetOutputValue(*leaf, 1), 94);
  EXPECT_FLOAT_EQ(op->GetOutputValue(*leaf, 2), 3.3333333);
}

}  // namespace
}  // namespace tensorflow
