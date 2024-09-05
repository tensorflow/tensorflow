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

#include <tuple>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/time/time.h"
#include "tensorflow/core/platform/test.h"

namespace tensorflow::serving {
namespace {

using ::testing::UnorderedElementsAre;

TEST(BatchStatsTest, GlobalBatchStatsRegistryAlwaysReturnsTheSameInstance) {
  ASSERT_EQ(&GlobalBatchStatsRegistry(), &GlobalBatchStatsRegistry());
}

TEST(BatchStatsTest, BasicOperation) {
  BatchStatsRegistry stats;
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
  BatchStatsRegistry stats;
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

TEST(BatchStatsTest, ProcessedSizeIsCorrect) {
  ModelBatchStats stats;

  stats.RegisterProcessedSize(5);
  stats.RegisterProcessedSize(7);

  ASSERT_EQ(stats.cumulative_processed_size(), 12);
}

TEST(BatchStatsTest, ModelOpNamesAreCorrect) {
  BatchStatsRegistry stats;

  // Register a cost for model "m" and op "o".
  stats.model(/* model_name= */ "m", /* op_name= */ "o")
      .batch_size(1)
      .tpu_cost()
      .Register(absl::Hours(5));

  // Register a cost for model "m2" and op "o".
  stats.model(/* model_name= */ "m2", /* op_name= */ "o")
      .batch_size(1)
      .tpu_cost()
      .Register(absl::Hours(7));

  // Register another cost for model "m" and op "o" (but different batch size).
  stats.model(/* model_name= */ "m", /* op_name= */ "o")
      .batch_size(2)
      .tpu_cost()
      .Register(absl::Hours(4));

  // Register a cost for model "m" and op "o2".
  stats.model(/* model_name= */ "m", /* op_name= */ "o2")
      .batch_size(1)
      .tpu_cost()
      .Register(absl::Hours(1));

  // Check that the model/op names are correct.
  ASSERT_THAT(stats.ModelAndOpNames(),
              UnorderedElementsAre(
                  std::tuple(/* model_name= */ "m", /* op_name= */ "o"),
                  std::tuple(/* model_name= */ "m", /* op_name= */ "o2"),
                  std::tuple(/* model_name= */ "m2", /* op_name= */ "o")));
}

TEST(BatchStatsTest, BatchSizesAreCorrect) {
  ModelBatchStats stats;

  // Register costs for batch sizes 1, 2, and 4.
  stats.batch_size(1).tpu_cost().Register(absl::Hours(5));
  stats.batch_size(4).tpu_cost().Register(absl::Hours(7));
  stats.batch_size(1).tpu_cost().Register(absl::Hours(4));
  stats.batch_size(2).tpu_cost().Register(absl::Hours(1));

  // Check that the batch sizes are correct.
  ASSERT_THAT(stats.BatchSizes(), UnorderedElementsAre(1, 2, 4));
}

TEST(BatchStatsTest, BatchTimeoutIsCorrect) {
  ModelBatchStats stats;

  // Originally the batch timeout is -1 if unassigned.
  ASSERT_EQ(stats.batch_timeout_micros(), -1);

  // Assign a batch timeout of 100 microseconds.
  stats.SetBatchTimeoutMicros(100);
  ASSERT_EQ(stats.batch_timeout_micros(), 100);
}

TEST(BatchStatsTest, NumBatchThreadsIsCorrect) {
  ModelBatchStats stats;

  // Originally the number of batch threads is -1 if unassigned.
  ASSERT_EQ(stats.num_batch_threads(), -1);

  // Assign a number of per-model batch threads.
  stats.SetNumBatchThreads(16);
  ASSERT_EQ(stats.num_batch_threads(), 16);
}

}  // namespace

}  // namespace tensorflow::serving
