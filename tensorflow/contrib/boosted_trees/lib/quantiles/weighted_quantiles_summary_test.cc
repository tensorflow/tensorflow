// Copyright 2016 The TensorFlow Authors. All Rights Reserved.
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
#include "tensorflow/contrib/boosted_trees/lib/quantiles/weighted_quantiles_summary.h"

#include "tensorflow/core/lib/random/philox_random.h"
#include "tensorflow/core/lib/random/simple_philox.h"
#include "tensorflow/core/platform/test.h"
#include "tensorflow/core/platform/test_benchmark.h"

namespace tensorflow {
namespace {

using Buffer = boosted_trees::quantiles::WeightedQuantilesBuffer<float, float>;
using BufferEntry =
    boosted_trees::quantiles::WeightedQuantilesBuffer<float,
                                                      float>::BufferEntry;
using Summary =
    boosted_trees::quantiles::WeightedQuantilesSummary<float, float>;
using SummaryEntry =
    boosted_trees::quantiles::WeightedQuantilesSummary<float,
                                                       float>::SummaryEntry;

class WeightedQuantilesSummaryTest : public ::testing::Test {
 protected:
  void SetUp() override {
    // Constructs a buffer of 10 weighted unique entries.
    buffer1_.reset(new Buffer(10, 1000));
    buffer1_->PushEntry(5, 9);
    buffer1_->PushEntry(2, 3);
    buffer1_->PushEntry(-1, 7);
    buffer1_->PushEntry(-7, 1);
    buffer1_->PushEntry(3, 2);
    buffer1_->PushEntry(-2, 3);
    buffer1_->PushEntry(21, 8);
    buffer1_->PushEntry(-13, 4);
    buffer1_->PushEntry(8, 2);
    buffer1_->PushEntry(-5, 6);

    // Constructs a buffer of 7 weighted unique entries.
    buffer2_.reset(new Buffer(7, 1000));
    buffer2_->PushEntry(9, 2);
    buffer2_->PushEntry(-7, 3);
    buffer2_->PushEntry(2, 1);
    buffer2_->PushEntry(4, 13);
    buffer2_->PushEntry(0, 5);
    buffer2_->PushEntry(-5, 3);
    buffer2_->PushEntry(11, 3);
  }

  void TearDown() override { buffer1_->Clear(); }

  std::unique_ptr<Buffer> buffer1_;
  std::unique_ptr<Buffer> buffer2_;
  const double buffer1_min_value_ = -13;
  const double buffer1_max_value_ = 21;
  const double buffer1_total_weight_ = 45;
  const double buffer2_min_value_ = -7;
  const double buffer2_max_value_ = 11;
  const double buffer2_total_weight_ = 30;
};

TEST_F(WeightedQuantilesSummaryTest, BuildFromBuffer) {
  Summary summary;
  summary.BuildFromBufferEntries(buffer1_->GenerateEntryList());

  // We expect no approximation error because no compress operation occured.
  EXPECT_EQ(summary.ApproximationError(), 0);

  // Check first and last elements in the summary.
  const auto& entries = summary.GetEntryList();
  // First element's rmin should be zero.
  EXPECT_EQ(summary.MinValue(), buffer1_min_value_);
  EXPECT_EQ(entries.front(), SummaryEntry(-13, 4, 0, 4));
  // Last element's rmax should be cumulative weight.
  EXPECT_EQ(summary.MaxValue(), buffer1_max_value_);
  EXPECT_EQ(entries.back(), SummaryEntry(21, 8, 37, 45));
  // Check total weight.
  EXPECT_EQ(summary.TotalWeight(), buffer1_total_weight_);
}

TEST_F(WeightedQuantilesSummaryTest, CompressSeparately) {
  for (int new_size = 9; new_size >= 2; --new_size) {
    Summary summary;
    summary.BuildFromBufferEntries(buffer1_->GenerateEntryList());
    summary.Compress(new_size);

    // Expect a max approximation error of 1 / n
    // ie. eps0 + 1/n but eps0 = 0.
    EXPECT_TRUE(summary.Size() >= new_size && summary.Size() <= new_size + 2);
    EXPECT_LE(summary.ApproximationError(), 1.0 / new_size);

    // Min/Max elements and total weight should not change.
    EXPECT_EQ(summary.MinValue(), buffer1_min_value_);
    EXPECT_EQ(summary.MaxValue(), buffer1_max_value_);
    EXPECT_EQ(summary.TotalWeight(), buffer1_total_weight_);
  }
}

TEST_F(WeightedQuantilesSummaryTest, CompressSequentially) {
  Summary summary;
  summary.BuildFromBufferEntries(buffer1_->GenerateEntryList());
  for (int new_size = 9; new_size >= 2; new_size -= 2) {
    double prev_eps = summary.ApproximationError();
    summary.Compress(new_size);

    // Expect a max approximation error of prev_eps + 1 / n.
    EXPECT_TRUE(summary.Size() >= new_size && summary.Size() <= new_size + 2);
    EXPECT_LE(summary.ApproximationError(), prev_eps + 1.0 / new_size);

    // Min/Max elements and total weight should not change.
    EXPECT_EQ(summary.MinValue(), buffer1_min_value_);
    EXPECT_EQ(summary.MaxValue(), buffer1_max_value_);
    EXPECT_EQ(summary.TotalWeight(), buffer1_total_weight_);
  }
}

TEST_F(WeightedQuantilesSummaryTest, CompressRandomized) {
  // Check multiple size compressions and ensure approximation bounds
  // are always respected.
  int prev_size = 1;
  int size = 2;
  float max_value = 1 << 20;
  while (size < (1 << 16)) {
    // Create buffer of size from uniform random elements.
    Buffer buffer(size, size << 4);
    random::PhiloxRandom philox(13);
    random::SimplePhilox rand(&philox);
    for (int i = 0; i < size; ++i) {
      buffer.PushEntry(rand.RandFloat() * max_value,
                       rand.RandFloat() * max_value);
    }

    // Create summary and compress.
    Summary summary;
    summary.BuildFromBufferEntries(buffer.GenerateEntryList());
    int new_size = std::max(rand.Uniform(size), 2u);
    summary.Compress(new_size);

    // Ensure approximation error is acceptable.
    EXPECT_TRUE(summary.Size() >= new_size && summary.Size() <= new_size + 2);
    EXPECT_LE(summary.ApproximationError(), 1.0 / new_size);

    // Update size to next fib number.
    size_t last_size = size;
    size += prev_size;
    prev_size = last_size;
  }
}

TEST_F(WeightedQuantilesSummaryTest, MergeSymmetry) {
  // Create two separate summaries and merge.
  Summary summary1;
  summary1.BuildFromBufferEntries(buffer1_->GenerateEntryList());
  Summary summary2;
  summary2.BuildFromBufferEntries(buffer2_->GenerateEntryList());

  // Merge summary 2 into 1 and verify.
  summary1.Merge(summary2);
  EXPECT_EQ(summary1.ApproximationError(), 0.0);
  EXPECT_EQ(summary1.MinValue(),
            std::min(buffer1_min_value_, buffer2_min_value_));
  EXPECT_EQ(summary1.MaxValue(),
            std::max(buffer1_max_value_, buffer2_max_value_));
  EXPECT_EQ(summary1.TotalWeight(),
            buffer1_total_weight_ + buffer2_total_weight_);
  EXPECT_EQ(summary1.Size(), 14);  // 14 unique values.

  // Merge summary 1 into 2 and verify same result.
  summary1.BuildFromBufferEntries(buffer1_->GenerateEntryList());
  summary2.Merge(summary1);
  EXPECT_EQ(summary2.ApproximationError(), 0.0);
  EXPECT_EQ(summary2.MinValue(),
            std::min(buffer1_min_value_, buffer2_min_value_));
  EXPECT_EQ(summary2.MaxValue(),
            std::max(buffer1_max_value_, buffer2_max_value_));
  EXPECT_EQ(summary2.TotalWeight(),
            buffer1_total_weight_ + buffer2_total_weight_);
  EXPECT_EQ(summary2.Size(), 14);  // 14 unique values.
}

TEST_F(WeightedQuantilesSummaryTest, CompressThenMerge) {
  // Create two separate summaries and merge.
  Summary summary1;
  summary1.BuildFromBufferEntries(buffer1_->GenerateEntryList());
  Summary summary2;
  summary2.BuildFromBufferEntries(buffer2_->GenerateEntryList());

  // Compress summaries.
  summary1.Compress(5);  // max error is 1/5.
  const auto eps1 = 1.0 / 5;
  EXPECT_LE(summary1.ApproximationError(), eps1);
  summary2.Compress(3);  // max error is 1/3.
  const auto eps2 = 1.0 / 3;
  EXPECT_LE(summary2.ApproximationError(), eps2);

  // Merge guarantees an approximation error of max(eps1, eps2).
  // Merge summary 2 into 1 and verify.
  summary1.Merge(summary2);
  EXPECT_LE(summary1.ApproximationError(), std::max(eps1, eps2));
  EXPECT_EQ(summary1.MinValue(),
            std::min(buffer1_min_value_, buffer2_min_value_));
  EXPECT_EQ(summary1.MaxValue(),
            std::max(buffer1_max_value_, buffer2_max_value_));
  EXPECT_EQ(summary1.TotalWeight(),
            buffer1_total_weight_ + buffer2_total_weight_);
}

}  // namespace
}  // namespace tensorflow
