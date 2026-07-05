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

TEST(MaybeBatchDownTest, EmptyBatch) {
  Batch<FakeTask> batch;
  batch.Close();

  std::vector<std::unique_ptr<FakeTask>> out_trimmed_tasks;

  MaybeBatchDown(
      /* batch= */ batch, /* allowed_batch_sizes= */ {1, 2, 4, 8},
      /* disable_padding= */ false,
      /* batch_padding_policy= */ kBatchDownPolicy,
      /* model_batch_stats= */ nullptr,
      /* out_trimmed_tasks= */ out_trimmed_tasks);

  EXPECT_TRUE(batch.empty());
  EXPECT_TRUE(out_trimmed_tasks.empty());
}

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

TEST(ApplyBatchPaddingPolicyTest, ZeroCandidateSize) {
  EXPECT_EQ(
      ApplyBatchPaddingPolicy(0, {2, 4}, false, kBatchDownPolicy, nullptr), 0);
}

TEST(ApplyBatchPaddingPolicyTest, PadUp) {
  EXPECT_EQ(ApplyBatchPaddingPolicy(3, {2, 4}, false, kPadUpPolicy, nullptr),
            3);
}

TEST(ApplyBatchPaddingPolicyTest, BatchDown) {
  EXPECT_EQ(
      ApplyBatchPaddingPolicy(3, {2, 4}, false, kBatchDownPolicy, nullptr), 2);
}

TEST(ApplyBatchPaddingPolicyTest,
     BatchDownDoesNothingWhenTheBatchSizeIsAlreadyAllowed) {
  EXPECT_EQ(
      ApplyBatchPaddingPolicy(2, {2, 4}, false, kBatchDownPolicy, nullptr), 2);
}

TEST(ApplyBatchPaddingPolicyTest,
     BatchDownDoesNothingWhenNoSmallerAllowedSize) {
  EXPECT_EQ(
      ApplyBatchPaddingPolicy(1, {2, 4}, false, kBatchDownPolicy, nullptr), 1);
}

TEST(ApplyBatchPaddingPolicyTest,
     BatchDownDoesNothingWhenCandidateExceedsAllAllowedSizes) {
  // When candidate_size > max(allowed_batch_sizes), pad_up_size falls back to
  // candidate_size and the early-return guard fires.
  // This should not happen in practice since the candidate size is capped to
  // the max execution batch size, which is equal to the max of the allowed
  // batch sizes.
  EXPECT_EQ(
      ApplyBatchPaddingPolicy(10, {2, 4, 8}, false, kBatchDownPolicy, nullptr),
      10);
}

TEST(ApplyBatchPaddingPolicyTest, MinimizeTpuCostPerRequestPicksBatchDown) {
  ModelBatchStats model_batch_stats;
  model_batch_stats.batch_size(2).tpu_cost().Register(absl::Seconds(2));
  model_batch_stats.batch_size(4).tpu_cost().Register(absl::Seconds(3.1));

  EXPECT_EQ(ApplyBatchPaddingPolicy(3, {2, 4}, false,
                                    kMinimizeTpuCostPerRequestPolicy,
                                    &model_batch_stats),
            2);
}

TEST(ApplyBatchPaddingPolicyTest, MinimizeTpuCostPerRequestPicksPadUp) {
  ModelBatchStats model_batch_stats;
  model_batch_stats.batch_size(2).tpu_cost().Register(absl::Seconds(2));
  model_batch_stats.batch_size(4).tpu_cost().Register(absl::Seconds(2.9));

  EXPECT_EQ(ApplyBatchPaddingPolicy(3, {2, 4}, false,
                                    kMinimizeTpuCostPerRequestPolicy,
                                    &model_batch_stats),
            3);
}

TEST(ApplyBatchPaddingPolicyTest,
     MinimizeTpuCostPerRequestMissingCostsReturnsCandidateSize) {
  ModelBatchStats model_batch_stats;
  model_batch_stats.batch_size(2).tpu_cost().Register(absl::Seconds(2));

  EXPECT_EQ(ApplyBatchPaddingPolicy(3, {2, 4}, false,
                                    kMinimizeTpuCostPerRequestPolicy,
                                    &model_batch_stats),
            3);
}

TEST(ApplyBatchPaddingPolicyTest,
     MinimizeTpuCostPerRequestNoModelStatsReturnsCandidateSize) {
  EXPECT_EQ(ApplyBatchPaddingPolicy(3, {2, 4}, false,
                                    kMinimizeTpuCostPerRequestPolicy, nullptr),
            3);
}

TEST(ApplyBatchPaddingPolicyTest, UnsupportedPolicy) {
  EXPECT_EQ(ApplyBatchPaddingPolicy(3, {2, 4}, false, "UNSUPPORTED", nullptr),
            3);
}

}  // namespace

}  // namespace serving
}  // namespace tensorflow
