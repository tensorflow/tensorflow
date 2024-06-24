/* Copyright 2021 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include "tensorflow/core/kernels/batching_util/batch_stats.h"

#include <gtest/gtest.h>
#include "absl/time/time.h"
#include "tensorflow/core/platform/test.h"

namespace tensorflow::serving {
namespace {

TEST(BatchStatsTest, GlobalBatchStatsAlwaysReturnsTheSameInstance) {
  ASSERT_EQ(&GlobalBatchStats(), &GlobalBatchStats());
}

TEST(BatchStatsTest, BasicOperation) {
  BatchStats stats;
  stats.model(/* model_name= */ "m", /* op_name= */ "o")
      .batch_size(1)
      .tpu_cost()
      .Register(absl::Hours(5));
  ASSERT_EQ(stats.model(/* model_name= */ "m", /* op_name= */ "o")
                .batch_size(1)
                .tpu_cost()
                .mean(),
            absl::Hours(5));
}

TEST(BatchStatsTest, ModelBatchStatsAreUniqueForEachModel) {
  BatchStats stats;
  ASSERT_NE(&stats.model(/* model_name= */ "m", /* op_name= */ "o"),
            &stats.model(/* model_name= */ "m", /* op_name= */ "o2"));
}

TEST(BatchStatsTest, BatchSizeStatsAreUniqueForEachBatchSize) {
  ModelBatchStats stats;
  ASSERT_NE(&stats.batch_size(1), &stats.batch_size(2));
}

TEST(BatchStatsTest, CostTrackerStartsWithNoMean) {
  CostTracker tracker;

  ASSERT_FALSE(tracker.mean().has_value());
}

TEST(BatchStatsTest, CostTrackerMeanIsCorrect) {
  CostTracker tracker;
  tracker.Register(absl::Hours(5));
  tracker.Register(absl::Hours(7));

  ASSERT_EQ(*tracker.mean(), absl::Hours(6));
}

}  // namespace

}  // namespace tensorflow::serving
