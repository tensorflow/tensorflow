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
#include "tensorflow/contrib/boosted_trees/lib/learner/common/accumulators/feature-stats-accumulator.h"

#include "tensorflow/core/framework/tensor_testutil.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/lib/core/status_test_util.h"
#include "tensorflow/core/platform/test.h"

namespace tensorflow {
namespace boosted_trees {
namespace learner {
namespace {

struct TestStats {
  TestStats& operator+=(const TestStats& other) {
    s1 += other.s1;
    s2 += other.s2;
    return (*this);
  }

  float s1;
  float s2;
};

struct TestStatsAccumulator {
  void operator()(const TestStats& from, TestStats* to) const { (*to) += from; }
};
#define EXPECT_STATS_EQ(val1, val2)  \
  EXPECT_FLOAT_EQ(val1.s1, val2.s1); \
  EXPECT_FLOAT_EQ(val1.s2, val2.s2);

using FeatureStatsAccumulator =
    FeatureStatsAccumulator<TestStats, TestStatsAccumulator>;

class FeatureStatsAccumulatorTest : public ::testing::Test {};

TEST_F(FeatureStatsAccumulatorTest, Empty) {
  FeatureStatsAccumulator accumulator(1);
  TestStats stats = {0, 0};

  EXPECT_STATS_EQ(stats, accumulator.GetStats(0, 2, 1, 234));
}

TEST_F(FeatureStatsAccumulatorTest, OneFeatureOneGrad) {
  FeatureStatsAccumulator accumulator(1);
  TestStats stats = {-12.023f, 8.2f};
  accumulator.AddStats(0, 2, 1, 234, stats);

  EXPECT_STATS_EQ(stats, accumulator.GetStats(0, 2, 1, 234));
}

TEST_F(FeatureStatsAccumulatorTest, OneFeatureAggregateGrad) {
  FeatureStatsAccumulator accumulator(1);
  TestStats stats1 = {-12.023f, 8.2f};
  accumulator.AddStats(0, 2, 1, 234, stats1);
  TestStats stats2 = {4.46f, 1.9f};
  accumulator.AddStats(0, 2, 1, 234, stats2);
  TestStats expected = {-7.563f, 10.1f};
  EXPECT_STATS_EQ(expected, accumulator.GetStats(0, 2, 1, 234));
}

TEST_F(FeatureStatsAccumulatorTest, TwoFeaturesOneGrad) {
  FeatureStatsAccumulator accumulator(1);
  TestStats stats1 = {-12.023f, 8.2f};
  accumulator.AddStats(0, 1, 0, 34, stats1);
  TestStats stats2 = {4.46f, 1.9f};
  accumulator.AddStats(0, 1, 0, 91, stats2);

  EXPECT_STATS_EQ(stats1, accumulator.GetStats(0, 1, 0, 34));
  EXPECT_STATS_EQ(stats2, accumulator.GetStats(0, 1, 0, 91));
}

}  // namespace
}  // namespace learner
}  // namespace boosted_trees
}  // namespace tensorflow
