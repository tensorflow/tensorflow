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
#include "tensorflow/contrib/boosted_trees/lib/quantiles/weighted_quantiles_buffer.h"

#include "tensorflow/core/lib/random/philox_random.h"
#include "tensorflow/core/lib/random/simple_philox.h"
#include "tensorflow/core/platform/test.h"
#include "tensorflow/core/platform/test_benchmark.h"

namespace tensorflow {
namespace {

using Buffer =
    boosted_trees::quantiles::WeightedQuantilesBuffer<double, double>;
using BufferEntry =
    boosted_trees::quantiles::WeightedQuantilesBuffer<double,
                                                      double>::BufferEntry;

class WeightedQuantilesBufferTest : public ::testing::Test {};

TEST_F(WeightedQuantilesBufferTest, Invalid) {
  EXPECT_DEATH(
      ({
        boosted_trees::quantiles::WeightedQuantilesBuffer<double, double>
            buffer(2, 0);
      }),
      "Invalid buffer specification");
  EXPECT_DEATH(
      ({
        boosted_trees::quantiles::WeightedQuantilesBuffer<double, double>
            buffer(0, 2);
      }),
      "Invalid buffer specification");
}

TEST_F(WeightedQuantilesBufferTest, PushEntryNotFull) {
  Buffer buffer(20, 100);
  buffer.PushEntry(5, 9);
  buffer.PushEntry(2, 3);
  buffer.PushEntry(-1, 7);
  buffer.PushEntry(3, 0);  // This entry will be ignored.

  EXPECT_FALSE(buffer.IsFull());
  EXPECT_EQ(buffer.Size(), 3);
}

TEST_F(WeightedQuantilesBufferTest, PushEntryFull) {
  // buffer capacity is 4.
  Buffer buffer(2, 100);
  buffer.PushEntry(5, 9);
  buffer.PushEntry(2, 3);
  buffer.PushEntry(-1, 7);
  buffer.PushEntry(2, 1);

  std::vector<BufferEntry> expected;
  expected.emplace_back(-1, 7);
  expected.emplace_back(2, 4);
  expected.emplace_back(5, 9);

  // At this point, we have pushed 4 entries and we expect the buffer to be
  // full.
  EXPECT_TRUE(buffer.IsFull());
  EXPECT_EQ(buffer.GenerateEntryList(), expected);
  EXPECT_FALSE(buffer.IsFull());
}

TEST_F(WeightedQuantilesBufferTest, PushEntryFullDeath) {
  // buffer capacity is 4.
  Buffer buffer(2, 100);
  buffer.PushEntry(5, 9);
  buffer.PushEntry(2, 3);
  buffer.PushEntry(-1, 7);
  buffer.PushEntry(2, 1);

  std::vector<BufferEntry> expected;
  expected.emplace_back(-1, 7);
  expected.emplace_back(2, 4);
  expected.emplace_back(5, 9);

  // At this point, we have pushed 4 entries and we expect the buffer to be
  // full.
  EXPECT_TRUE(buffer.IsFull());
  // Can't push any more entries before clearing.
  EXPECT_DEATH(({ buffer.PushEntry(6, 6); }), "Buffer already full");
}

}  // namespace
}  // namespace tensorflow
