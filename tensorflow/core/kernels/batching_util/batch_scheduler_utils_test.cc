/* Copyright 2024 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/core/kernels/batching_util/batch_scheduler_utils.h"

#include <cstddef>
#include <memory>
#include <vector>

#include <gtest/gtest.h>
#include "absl/time/time.h"
#include "tensorflow/core/kernels/batching_util/batch_scheduler.h"
#include "tensorflow/core/kernels/batching_util/batch_stats.h"

namespace tensorflow {
namespace serving {

namespace {

TEST(GetNextAllowedBatchSizeTest, PaddingDisallowed) {
  EXPECT_EQ(GetNextAllowedBatchSize(3, {2, 4, 8}, true), 3);
}

TEST(GetNextAllowedBatchSizeTest, EmptyAllowedBatchSizes) {
  EXPECT_EQ(GetNextAllowedBatchSize(3, {}, false), 3);
}

TEST(GetNextAllowedBatchSizeTest, NextAllowedBatchSizeFound) {
  EXPECT_EQ(GetNextAllowedBatchSize(3, {2, 4, 8}, false), 4);
}

TEST(GetNextAllowedBatchSizeTest, AlreadyAllowedBatchSize) {
  EXPECT_EQ(GetNextAllowedBatchSize(2, {2, 4, 8}, false), 2);
}

TEST(GetNextAllowedBatchSizeTest, GreaterThanAllowedBatchSize) {
  EXPECT_EQ(GetNextAllowedBatchSize(10, {2, 4, 8}, false), 10);
}

TEST(GetPrevAllowedBatchSizeTest, PaddingDisallowed) {
  EXPECT_EQ(GetPrevAllowedBatchSize(3, {2, 4, 8}, true), 3);
}

TEST(GetPrevAllowedBatchSizeTest, EmptyAllowedBatchSizes) {
  EXPECT_EQ(GetPrevAllowedBatchSize(3, {}, false), 3);
}

TEST(GetPrevAllowedBatchSizeTest, PrevAllowedBatchSizeFound) {
  EXPECT_EQ(GetPrevAllowedBatchSize(3, {1, 2, 4, 8}, false), 2);
}

TEST(GetPrevAllowedBatchSizeTest, NoSmallerAllowedBatchSizeFound) {
  EXPECT_EQ(GetPrevAllowedBatchSize(3, {4, 8}, false), 3);
}

TEST(GetPrevAllowedBatchSizeTest, AlreadyAllowedBatchSize) {
  EXPECT_EQ(GetPrevAllowedBatchSize(2, {1, 2, 4, 8}, false), 2);
}

TEST(GetPrevAllowedBatchSizeTest, GreaterThanMaxAllowedBatchSize) {
  EXPECT_EQ(GetPrevAllowedBatchSize(10, {2, 4, 8}, false), 8);
}

class FakeTask : public BatchTask {
 public:
  explicit FakeTask(size_t size) : size_(size) {}

  size_t size() const override { return size_; }

 private:
  const size_t size_;
};

TEST(MaybeBatchDownTest, PadUp) {
  Batch<FakeTask> batch;
  batch.AddTask(std::make_unique<FakeTask>(1));
  batch.AddTask(std::make_unique<FakeTask>(1));
  batch.AddTask(std::make_unique<FakeTask>(1));
  batch.Close();

  std::vector<std::unique_ptr<FakeTask>> out_trimmed_tasks;

  MaybeBatchDown(
      /* batch= */ batch, /* allowed_batch_sizes= */ {1, 2, 4, 8},
      /* disable_padding= */ false,
      /* batch_padding_policy= */ kPadUpPolicy,
      /* model_batch_stats= */ nullptr,
      /* out_trimmed_tasks= */ out_trimmed_tasks);

  // The batch must stay unchanged (for the batch resource to then pad it to the
  // next allowed batch size, thus ending up in a pad-up behavior.)
  EXPECT_EQ(batch.size(), 3);
}

TEST(MaybeBatchDownTest, BatchDown) {
  Batch<FakeTask> batch;
  batch.AddTask(std::make_unique<FakeTask>(1));
  batch.AddTask(std::make_unique<FakeTask>(1));
  batch.AddTask(std::make_unique<FakeTask>(1));
  batch.Close();

  std::vector<std::unique_ptr<FakeTask>> out_trimmed_tasks;

  MaybeBatchDown(
      /* batch= */ batch, /* allowed_batch_sizes= */ {1, 2, 4, 8},
      /* disable_padding= */ false,
      /* batch_padding_policy= */ kBatchDownPolicy,
      /* model_batch_stats= */ nullptr,
      /* out_trimmed_tasks= */ out_trimmed_tasks);

  // The scheduler should trim the batch to a smaller allowed size that requires
  // no padding.
  EXPECT_EQ(batch.size(), 2);
  // The trimmed part.
  EXPECT_EQ(out_trimmed_tasks.size(), 1);
}

TEST(MaybeBatchDownTest, BatchDownDoesNotSplitTasks) {
  // Add tasks for size 3, but the second task is large and will have to be
  // split if doing batch-down.
  Batch<FakeTask> batch;
  batch.AddTask(std::make_unique<FakeTask>(1));
  batch.AddTask(std::make_unique<FakeTask>(2));
  batch.Close();

  std::vector<std::unique_ptr<FakeTask>> out_trimmed_tasks;

  MaybeBatchDown(
      /* batch= */ batch, /* allowed_batch_sizes= */ {1, 2, 4, 8},
      /* disable_padding= */ false,
      /* batch_padding_policy= */ kBatchDownPolicy,
      /* model_batch_stats= */ nullptr,
      /* out_trimmed_tasks= */ out_trimmed_tasks);

  // The batch must stay unchanged due the fact that the current implementation
  // doesn's support splitting large tasks.
  EXPECT_EQ(batch.size(), 3);
}

TEST(MaybeBatchDownTest, BatchDownDoesNothingWhenTheBatchSizeIsAlreadyAllowed) {
  Batch<FakeTask> batch;
  batch.AddTask(std::make_unique<FakeTask>(1));
  batch.AddTask(std::make_unique<FakeTask>(1));
  batch.AddTask(std::make_unique<FakeTask>(1));
  batch.AddTask(std::make_unique<FakeTask>(1));
  batch.Close();

  std::vector<std::unique_ptr<FakeTask>> out_trimmed_tasks;

  MaybeBatchDown(
      /* batch= */ batch, /* allowed_batch_sizes= */ {1, 2, 4, 8},
      /* disable_padding= */ false,
      /* batch_padding_policy= */ kBatchDownPolicy,
      /* model_batch_stats= */ nullptr,
      /* out_trimmed_tasks= */ out_trimmed_tasks);

  // The batch should stay unchanged because it's already of an allowed size.
  EXPECT_EQ(batch.size(), 4);
}

TEST(MaybeBatchDownTest, BatchDownDoesNothingWhenNoSmallerAllowedSize) {
  Batch<FakeTask> batch;
  batch.AddTask(std::make_unique<FakeTask>(1));
  batch.AddTask(std::make_unique<FakeTask>(1));
  batch.AddTask(std::make_unique<FakeTask>(1));
  batch.Close();

  std::vector<std::unique_ptr<FakeTask>> out_trimmed_tasks;

  MaybeBatchDown(
      /* batch= */ batch, /* allowed_batch_sizes= */ {4, 8},
      /* disable_padding= */ false,
      /* batch_padding_policy= */ kBatchDownPolicy,
      /* model_batch_stats= */ nullptr,
      /* out_trimmed_tasks= */ out_trimmed_tasks);

  // Can't batch down because there is no smaller allowed size.
  EXPECT_EQ(batch.size(), 3);
}

TEST(MaybeBatchDownTest, MinimizeTpuCostPerRequestPicksBatchDown) {
  Batch<FakeTask> batch;
  batch.AddTask(std::make_unique<FakeTask>(1));
  batch.AddTask(std::make_unique<FakeTask>(1));
  batch.AddTask(std::make_unique<FakeTask>(1));
  batch.Close();

  ModelBatchStats model_batch_stats;
  model_batch_stats.batch_size(2).tpu_cost().Register(absl::Seconds(2));
  model_batch_stats.batch_size(4).tpu_cost().Register(absl::Seconds(3.1));

  std::vector<std::unique_ptr<FakeTask>> out_trimmed_tasks;
  MaybeBatchDown(
      /* batch= */ batch, /* allowed_batch_sizes= */ {2, 4},
      /* disable_padding= */ false,
      /* batch_padding_policy= */ kMinimizeTpuCostPerRequestPolicy,
      /* model_batch_stats= */ &model_batch_stats,
      /* out_trimmed_tasks= */ out_trimmed_tasks);

  EXPECT_EQ(batch.size(), 2);
}

TEST(MaybeBatchDownTest, MinimizeTpuCostPerRequestPicksPadUp) {
  Batch<FakeTask> batch;
  batch.AddTask(std::make_unique<FakeTask>(1));
  batch.AddTask(std::make_unique<FakeTask>(1));
  batch.AddTask(std::make_unique<FakeTask>(1));
  batch.Close();

  ModelBatchStats model_batch_stats;
  model_batch_stats.batch_size(2).tpu_cost().Register(absl::Seconds(2));
  model_batch_stats.batch_size(4).tpu_cost().Register(absl::Seconds(2.9));

  std::vector<std::unique_ptr<FakeTask>> out_trimmed_tasks;
  MaybeBatchDown(
      /* batch= */ batch, /* allowed_batch_sizes= */ {2, 4},
      /* disable_padding= */ false,
      /* batch_padding_policy= */ kMinimizeTpuCostPerRequestPolicy,
      /* model_batch_stats= */ &model_batch_stats,
      /* out_trimmed_tasks= */ out_trimmed_tasks);

  EXPECT_EQ(batch.size(), 3);
}

TEST(MaybeBatchDownTest, MinimizeTpuCostPerRequestIsOkWithMissingCosts) {
  Batch<FakeTask> batch;
  batch.AddTask(std::make_unique<FakeTask>(1));
  batch.AddTask(std::make_unique<FakeTask>(1));
  batch.AddTask(std::make_unique<FakeTask>(1));
  batch.Close();

  ModelBatchStats model_batch_stats;
  model_batch_stats.batch_size(2).tpu_cost().Register(absl::Seconds(2));
  // Not adding costs for batch 4.

  std::vector<std::unique_ptr<FakeTask>> out_trimmed_tasks;
  MaybeBatchDown(
      /* batch= */ batch, /* allowed_batch_sizes= */ {2, 4},
      /* disable_padding= */ false,
      /* batch_padding_policy= */ kMinimizeTpuCostPerRequestPolicy,
      /* model_batch_stats= */ &model_batch_stats,
      /* out_trimmed_tasks= */ out_trimmed_tasks);

  // No expectations as we do not expect a particular behavior. We just care
  // that we don't crash.
}

TEST(MaybeBatchDownTest, MinimizeTpuCostPerRequestDoesPadUpWhenNoModelStats) {
  Batch<FakeTask> batch;
  batch.AddTask(std::make_unique<FakeTask>(1));
  batch.AddTask(std::make_unique<FakeTask>(1));
  batch.AddTask(std::make_unique<FakeTask>(1));
  batch.Close();

  std::vector<std::unique_ptr<FakeTask>> out_trimmed_tasks;
  MaybeBatchDown(
      /* batch= */ batch, /* allowed_batch_sizes= */ {2, 4},
      /* disable_padding= */ false,
      /* batch_padding_policy= */ kMinimizeTpuCostPerRequestPolicy,
      /* model_batch_stats= */ nullptr,
      /* out_trimmed_tasks= */ out_trimmed_tasks);

  EXPECT_EQ(batch.size(), 3);
}

}  // namespace

}  // namespace serving
}  // namespace tensorflow
