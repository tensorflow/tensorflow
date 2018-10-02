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
#include "tensorflow/contrib/tensor_forest/kernels/v4/grow_stats.h"

#include "tensorflow/contrib/decision_trees/proto/generic_tree_model.pb.h"
#include "tensorflow/contrib/tensor_forest/kernels/v4/test_utils.h"
#include "tensorflow/contrib/tensor_forest/proto/tensor_forest_params.pb.h"
#include "tensorflow/core/platform/protobuf.h"
#include "tensorflow/core/platform/test.h"
#include "tensorflow/core/platform/types.h"

namespace tensorflow {
namespace {

using tensorflow::decision_trees::BinaryNode;
using tensorflow::decision_trees::FeatureId;
using tensorflow::decision_trees::InequalityTest;
using tensorflow::tensorforest::DenseClassificationGrowStats;
using tensorflow::tensorforest::FertileSlot;
using tensorflow::tensorforest::FixedSizeClassStats;
using tensorflow::tensorforest::FixedSizeSparseClassificationGrowStats;
using tensorflow::tensorforest::GrowStats;
using tensorflow::tensorforest::LeastSquaresRegressionGrowStats;
using tensorflow::tensorforest::SparseClassificationGrowStats;
using tensorflow::tensorforest::SPLIT_FINISH_BASIC;
using tensorflow::tensorforest::SPLIT_FINISH_DOMINATE_HOEFFDING;
using tensorflow::tensorforest::SPLIT_PRUNE_HOEFFDING;
using tensorflow::tensorforest::TensorForestParams;
using tensorflow::tensorforest::TestableInputTarget;

BinaryNode MakeSplit(const string& feat, float val) {
  BinaryNode split;
  InequalityTest* test = split.mutable_inequality_left_child_test();
  FeatureId feature_id;
  feature_id.mutable_id()->set_value(feat);
  *test->mutable_feature_id() = feature_id;
  test->mutable_threshold()->set_float_value(val);
  test->set_type(InequalityTest::LESS_OR_EQUAL);

  return split;
}

void RunBatch(GrowStats* stats, const TestableInputTarget* target) {
  std::unique_ptr<tensorflow::tensorforest::TensorDataSet> dataset(
      new tensorflow::tensorforest::TestableDataSet(
          {1.0, 2.0, 3.0, 4.0, 5.0, 6.0}, 2));

  stats->AddSplit(MakeSplit("0", 10.0), dataset, target, 0);
  stats->AddSplit(MakeSplit("1", 4.0), dataset, target, 0);

  for (int i = 0; i < target->NumItems(); ++i) {
    stats->AddExample(dataset, target, i);
  }
}

TEST(GrowStatsDenseClassificationTest, Basic) {
  TensorForestParams params;
  params.set_num_outputs(2);
  params.mutable_split_after_samples()->set_constant_value(2);
  params.mutable_num_splits_to_consider()->set_constant_value(2);
  std::unique_ptr<DenseClassificationGrowStats> stat(
      new DenseClassificationGrowStats(params, 1));
  stat->Initialize();

  std::vector<float> labels = {1, 0, 1};
  std::vector<float> weights = {2.3, 20.3, 1.1};
  std::unique_ptr<TestableInputTarget> target(
      new TestableInputTarget(labels, weights, 1));

  RunBatch(stat.get(), target.get());
  CHECK(stat->IsFinished());

  FertileSlot slot;
  stat->PackToProto(&slot);

  string serialized = slot.DebugString();

  std::unique_ptr<DenseClassificationGrowStats> new_stat(
      new DenseClassificationGrowStats(params, 1));
  new_stat->ExtractFromProto(slot);
  FertileSlot second_one;
  new_stat->PackToProto(&second_one);
  string serialized_again = second_one.DebugString();
  ASSERT_EQ(serialized_again, serialized);
}

class TestableRunningStats : public DenseClassificationGrowStats {
 public:
  TestableRunningStats(const TensorForestParams& params, int32 depth)
      : DenseClassificationGrowStats(params, depth) {}

  float test_left_sum(int split) { return get_left_gini()->sum(split); }
  float test_left_square(int split) { return get_left_gini()->square(split); }
  float test_right_sum(int split) { return get_right_gini()->sum(split); }
  float test_right_square(int split) { return get_right_gini()->square(split); }
};

TEST(GrowStatsDenseClassificationTest, BasicRunningStats) {
  TensorForestParams params;
  params.set_num_outputs(2);
  params.mutable_split_after_samples()->set_constant_value(2);
  params.mutable_num_splits_to_consider()->set_constant_value(2);
  params.set_use_running_stats_method(true);
  std::unique_ptr<TestableRunningStats> stat(
      new TestableRunningStats(params, 1));
  stat->Initialize();

  std::vector<float> labels = {1, 0, 1};
  std::vector<float> weights = {2.3, 20.3, 1.1};
  std::unique_ptr<TestableInputTarget> target(
      new TestableInputTarget(labels, weights, 1));

  RunBatch(stat.get(), target.get());
  CHECK(stat->IsFinished());

  ASSERT_FLOAT_EQ(stat->test_left_sum(0), 2.3 + 20.3 + 1.1);
  ASSERT_FLOAT_EQ(stat->test_left_square(0), 3.4 * 3.4 + 20.3 * 20.3);
  ASSERT_FLOAT_EQ(stat->test_right_sum(0), 0.0);
  ASSERT_FLOAT_EQ(stat->test_right_square(0), 0.0);

  ASSERT_FLOAT_EQ(stat->test_left_sum(1), 2.3 + 20.3);
  ASSERT_FLOAT_EQ(stat->test_left_square(1), 2.3 * 2.3 + 20.3 * 20.3);
  ASSERT_FLOAT_EQ(stat->test_right_sum(1), 1.1);
  ASSERT_FLOAT_EQ(stat->test_right_square(1), 1.1 * 1.1);

  FertileSlot slot;
  stat->PackToProto(&slot);

  string serialized = slot.DebugString();

  std::unique_ptr<DenseClassificationGrowStats> new_stat(
      new DenseClassificationGrowStats(params, 1));
  new_stat->ExtractFromProto(slot);
  FertileSlot second_one;
  new_stat->PackToProto(&second_one);
  string serialized_again = second_one.DebugString();
  ASSERT_EQ(serialized_again, serialized);
}

class TestableFinishEarly : public DenseClassificationGrowStats {
 public:
  TestableFinishEarly(const TensorForestParams& params, int32 depth)
      : DenseClassificationGrowStats(params, depth), num_times_called_(0) {}

  int num_times_called_;

 protected:
  void CheckFinishEarlyHoeffding() override { ++num_times_called_; }
};

TEST(GrowStatsDenseClassificationTest, TestFinishEarly) {
  TensorForestParams params;
  params.set_num_outputs(2);
  params.mutable_split_after_samples()->set_constant_value(2);
  params.mutable_num_splits_to_consider()->set_constant_value(2);
  params.mutable_min_split_samples()->set_constant_value(15);
  params.mutable_dominate_fraction()->set_constant_value(0.99);
  auto* finish = params.mutable_finish_type();
  finish->set_type(SPLIT_FINISH_DOMINATE_HOEFFDING);
  finish->mutable_check_every_steps()->set_constant_value(5);
  std::unique_ptr<TestableFinishEarly> stat(new TestableFinishEarly(params, 1));
  stat->Initialize();

  std::vector<float> labels = {1, 0, 1};
  std::vector<float> weights = {1, 1, 1};
  std::unique_ptr<TestableInputTarget> target(
      new TestableInputTarget(labels, weights, 1));
  std::unique_ptr<tensorflow::tensorforest::TensorDataSet> dataset(
      new tensorflow::tensorforest::TestableDataSet(
          {1.0, 2.0, 3.0, 4.0, 5.0, 6.0}, 2));

  // Run through the 3 examples
  RunBatch(stat.get(), target.get());

  ASSERT_EQ(stat->num_times_called_, 0);

  // Go over min_split_samples.
  for (int i = 0; i < 13; ++i) {
    stat->AddExample(dataset, target.get(), 0);
  }

  ASSERT_EQ(stat->num_times_called_, 1);

  // More examples up to 55.
  for (int i = 0; i < 39; ++i) {
    stat->AddExample(dataset, target.get(), 0);
  }

  ASSERT_EQ(stat->num_times_called_, 9);
}

TEST(GrowStatsDenseClassificationTest, TestCheckPruneHoeffding) {
  TensorForestParams params;
  params.set_num_outputs(2);
  params.mutable_split_after_samples()->set_constant_value(2000);
  params.mutable_num_splits_to_consider()->set_constant_value(2);
  params.mutable_min_split_samples()->set_constant_value(15);
  params.mutable_dominate_fraction()->set_constant_value(0.99);
  auto* finish = params.mutable_finish_type();
  finish->set_type(SPLIT_FINISH_BASIC);
  finish->mutable_check_every_steps()->set_constant_value(100);
  params.mutable_pruning_type()->set_type(SPLIT_PRUNE_HOEFFDING);
  params.mutable_pruning_type()
      ->mutable_prune_every_samples()
      ->set_constant_value(1);

  // On each iteration, we add two examples, one of class 0 and one
  // of class 1.  Split #0 classifies them perfectly, while split #1
  // sends them both to the left.
  std::vector<float> labels = {0, 1};
  std::vector<float> weights = {1, 1};
  TestableInputTarget target(labels, weights, 1);
  std::unique_ptr<tensorflow::tensorforest::TensorDataSet> dataset(
      new tensorflow::tensorforest::TestableDataSet({-1.0, -1.0, 1.0, -1.0},
                                                    2));

  DenseClassificationGrowStats stats(params, 1);
  stats.Initialize();
  stats.AddSplit(MakeSplit("0", 0.0), dataset, &target, 0);
  stats.AddSplit(MakeSplit("1", 0.0), dataset, &target, 0);

  // Math time!
  // After 2n samples,
  // split 0 has smoothed counts (n+1,1);(1,n+1) and
  // split 1 has smoothed counts (n+1,n+1);(1,1)
  // split 0 smoothed ginis are both 1 - (n+1)^2/(n+2)^2 - 1/(n+2)^2 and
  // split 1 smoothed ginis are 1 - 2 (n+1)^2 / (2n+2)^2 and 1 - 2 (1/4) = 1/2
  // split 0 weighted smoothed ginis are both n (1 - (n^2 + 2n + 2) / (n+2)^2)
  // split 1 weighted smoothed ginis are 0 and 2n (1 - 2(n+1)^2 / (2n+2)^2)
  // split 0 split score = 2n (1 - (n^2 + 2n + 2) / (n+2)^2)
  // split 1 spilt score = 2n (1 - 2(n+1)^2 / (2n+2)^2)
  // split 1 score - split 0 score =
  //    2n ( (n^2 + 2n + 2) / (n+2)^2 - 2(n+1)^2 / (2n+2)^2 )
  //  = 2n ( (n^2 + 2n + 2) (2n+2)^2 - 2(n+1)^2 (n+2)^2 ) / ((n+2)^2 (2n+2)^2 )
  //  = 2n ((n^2+2n+2)(4n^2+8n+4) - 2(n^2+2n+1)(n^2+4n+4)) / ((n+2)^2 (2n+2)^2)
  //  = 2n (4n^4+8n^3+4n^2+8n^3+16n^2+8n+8n^2+16n+8
  //         - (2n^4+8n^3+8n^2+4n^3+16n^2+16n+2n^2+8n+8)) / ((n+2)^2 (2n+2)^2)
  //  = 2n (2n^4 + 4n^3 + 2n^2) / ((n+2)^2 (2n+2)^2)
  //  = 4n^3 (n^2 + 2n + 1) / ((n+2)^2 (2n+2)^2)
  //  = n^3  / (n+2)^2
  //  Meanwhile, after 2n samples,
  //  epsilon = 2n (1 - 1/2) sqrt(0.5 ln(1/0.01) / 2n)
  //          = n sqrt( ln(10) / 2n)
  //  Graphical comparison says that epsilon is greater between 0 and 4.5,
  //  and then the split score difference is greater for n >= 5.
  // n = 1
  stats.AddExample(dataset, &target, 0);
  stats.AddExample(dataset, &target, 1);
  ASSERT_EQ(stats.num_splits(), 2);

  // n = 2
  stats.AddExample(dataset, &target, 0);
  stats.AddExample(dataset, &target, 1);
  ASSERT_EQ(stats.num_splits(), 2);

  // n = 3
  stats.AddExample(dataset, &target, 0);
  stats.AddExample(dataset, &target, 1);
  ASSERT_EQ(stats.num_splits(), 2);

  // n = 4
  stats.AddExample(dataset, &target, 0);
  stats.AddExample(dataset, &target, 1);
  ASSERT_EQ(stats.num_splits(), 2);

  // n = 5
  stats.AddExample(dataset, &target, 0);
  stats.AddExample(dataset, &target, 1);
  ASSERT_EQ(stats.num_splits(), 1);

  // n = 6
  stats.AddExample(dataset, &target, 0);
  stats.AddExample(dataset, &target, 1);
  ASSERT_EQ(stats.num_splits(), 1);
}

TEST(GrowStatsLeastSquaresRegressionTest, Basic) {
  TensorForestParams params;
  params.set_num_outputs(1);
  params.mutable_split_after_samples()->set_constant_value(2);
  params.mutable_num_splits_to_consider()->set_constant_value(2);
  std::unique_ptr<LeastSquaresRegressionGrowStats> stat(
      new LeastSquaresRegressionGrowStats(params, 1));
  stat->Initialize();

  std::vector<float> labels = {2.3, 5.6, 1.1};
  std::unique_ptr<TestableInputTarget> target(
      new TestableInputTarget(labels, {}, 1));
  std::vector<int> branches = {1, 0, 1, 1, 0, 0};

  RunBatch(stat.get(), target.get());
  CHECK(stat->IsFinished());

  FertileSlot slot;
  stat->PackToProto(&slot);

  string serialized = slot.DebugString();

  std::unique_ptr<LeastSquaresRegressionGrowStats> new_stat(
      new LeastSquaresRegressionGrowStats(params, 1));
  new_stat->ExtractFromProto(slot);
  FertileSlot second_one;
  new_stat->PackToProto(&second_one);
  string serialized_again = second_one.DebugString();

  ASSERT_EQ(serialized_again, serialized);
}

TEST(GrowStatsSparseClassificationTest, Basic) {
  TensorForestParams params;
  params.set_num_outputs(2);
  params.mutable_split_after_samples()->set_constant_value(2);
  params.mutable_num_splits_to_consider()->set_constant_value(2);
  std::unique_ptr<SparseClassificationGrowStats> stat(
      new SparseClassificationGrowStats(params, 1));
  stat->Initialize();

  std::vector<float> labels = {100, 1000, 1};
  std::vector<float> weights = {2.3, 20.3, 1.1};
  std::unique_ptr<TestableInputTarget> target(
      new TestableInputTarget(labels, weights, 1));
  std::vector<int> branches = {1, 0, 1, 1, 0, 0};

  RunBatch(stat.get(), target.get());
  CHECK(stat->IsFinished());

  FertileSlot slot;
  stat->PackToProto(&slot);

  string serialized = slot.DebugString();

  std::unique_ptr<SparseClassificationGrowStats> new_stat(
      new SparseClassificationGrowStats(params, 1));
  new_stat->ExtractFromProto(slot);
  FertileSlot second_one;
  new_stat->PackToProto(&second_one);
  string serialized_again = second_one.DebugString();
  ASSERT_EQ(serialized_again, serialized);
}

TEST(FixedSizeClassStats, Exact) {
  FixedSizeClassStats stats(10, 100);

  stats.accumulate(1, 1.0);
  stats.accumulate(2, 2.0);
  stats.accumulate(3, 3.0);

  EXPECT_EQ(stats.get_weight(1), 1.0);
  EXPECT_EQ(stats.get_weight(2), 2.0);
  EXPECT_EQ(stats.get_weight(3), 3.0);

  float sum;
  float square;
  stats.set_sum_and_square(&sum, &square);

  EXPECT_EQ(sum, 6.0);
  EXPECT_EQ(square, 14.0);
}

TEST(FixedSizeClassStats, Approximate) {
  FixedSizeClassStats stats(5, 10);

  for (int i = 1; i <= 10; i++) {
    stats.accumulate(i, i * 1.0);
  }

  // We should be off by no more than *half* of the least weight
  // in the class_weights_, which is 7.
  float tolerance = 3.5;
  for (int i = 1; i <= 10; i++) {
    float diff = stats.get_weight(i) - i * 1.0;
    EXPECT_LE(diff, tolerance);
    EXPECT_GE(diff, -tolerance);
  }
}

TEST(GrowStatsFixedSizeSparseClassificationTest, Basic) {
  TensorForestParams params;
  params.set_num_outputs(2);
  params.set_num_classes_to_track(5);
  params.mutable_split_after_samples()->set_constant_value(2);
  params.mutable_num_splits_to_consider()->set_constant_value(2);
  std::unique_ptr<FixedSizeSparseClassificationGrowStats> stat(
      new FixedSizeSparseClassificationGrowStats(params, 1));
  stat->Initialize();

  std::vector<float> labels = {100, 1000, 1};
  std::vector<float> weights = {2.3, 20.3, 1.1};
  std::unique_ptr<TestableInputTarget> target(
      new TestableInputTarget(labels, weights, 1));
  std::vector<int> branches = {1, 0, 1, 1, 0, 0};

  RunBatch(stat.get(), target.get());
  CHECK(stat->IsFinished());

  FertileSlot slot;
  stat->PackToProto(&slot);

  string serialized = slot.DebugString();

  std::unique_ptr<FixedSizeSparseClassificationGrowStats> new_stat(
      new FixedSizeSparseClassificationGrowStats(params, 1));
  new_stat->ExtractFromProto(slot);
  FertileSlot second_one;
  new_stat->PackToProto(&second_one);
  string serialized_again = second_one.DebugString();
  ASSERT_EQ(serialized_again, serialized);
}

}  // namespace
}  // namespace tensorflow
