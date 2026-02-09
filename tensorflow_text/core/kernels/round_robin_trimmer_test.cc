// Copyright 2025 TF.Text Authors.
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

#include "tensorflow_text/core/kernels/round_robin_trimmer.h"

#include <utility>
#include <vector>
#include <tuple>

#include <gmock/gmock.h>
#include <gtest/gtest.h>

namespace tensorflow {
namespace text {
namespace {

using ::testing::ElementsAreArray;

struct TestSpec {
  int max_sequence_length;
  std::vector<int> vals_a_row_1;
  std::vector<int> vals_a_row_2;
  std::vector<int> vals_b_row_1;
  std::vector<int> vals_b_row_2;
  std::vector<bool> mask_a_row_1;
  std::vector<bool> mask_a_row_2;
  std::vector<bool> mask_b_row_1;
  std::vector<bool> mask_b_row_2;
};

class RoundRobinTrimmerTest : public testing::TestWithParam<TestSpec> {
 protected:
  using Segment = std::vector<int>;
  using SegmentBatch =  std::vector<Segment>;
  using Splits = std::vector<int>;
  using Masks = std::vector<bool>;
  using MasksBatch = std::vector<Masks>;

  std::vector<SegmentBatch> GetRaggedInput() {
    SegmentBatch a = {input_a_row_1, input_a_row_2};
    SegmentBatch b = {input_b_row_1, input_b_row_2};

    return {a, b};
  }

  std::vector<Segment> GetFirstBatch() {
    return {input_a_row_1, input_b_row_1};
  }

  std::vector<Segment> GetSecondBatch() {
    return {input_a_row_2, input_b_row_2};
  }

  std::pair<std::vector<Segment>, std::vector<Splits>> GetFlatInput() {
    Segment a_vals(input_a_row_1.begin(), input_a_row_1.end());
    a_vals.insert(a_vals.end(), input_a_row_2.begin(), input_a_row_2.end());
    Segment b_vals(input_b_row_1.begin(), input_b_row_1.end());
    b_vals.insert(b_vals.end(), input_b_row_2.begin(), input_b_row_2.end());

    Splits a_splits = {0};
    a_splits.push_back(input_a_row_1.size());
    a_splits.push_back(a_splits.back() + input_a_row_2.size());
    Splits b_splits = {0};
    b_splits.push_back(input_b_row_1.size());
    b_splits.push_back(b_splits.back() + input_b_row_2.size());

    std::vector<Segment> vals = {a_vals, b_vals};
    std::vector<Splits> splits = {a_splits, b_splits};
    return std::make_pair(vals, splits);
  }

  template <typename T>
  std::vector<T> Concat(std::vector<T> a, std::vector<T> b) {
    std::vector<T> result(a.begin(), a.end());
    result.insert(result.end(), b.begin(), b.end());
    return result;
  }

 private:
  const Segment input_a_row_1 = {1, 2, 3, 4, 5};
  const Segment input_a_row_2 = {6, 7};
  const Segment input_b_row_1 = {10, 20, 30, 40, 50};
  const Segment input_b_row_2 = {60, 70};
};

static const std::vector<TestSpec>& params = {
    {
      .max_sequence_length = 10,
      .vals_a_row_1 = {1, 2, 3, 4, 5},
      .vals_a_row_2 = {6, 7},
      .vals_b_row_1 = {10, 20, 30, 40, 50},
      .vals_b_row_2 = {60, 70},
      .mask_a_row_1 = {true, true, true, true, true},
      .mask_a_row_2 = {true, true},
      .mask_b_row_1 = {true, true, true, true, true},
      .mask_b_row_2 = {true, true},
    },
    {
      .max_sequence_length = 6,
      .vals_a_row_1 = {1, 2, 3},
      .vals_a_row_2 = {6, 7},
      .vals_b_row_1 = {10, 20, 30},
      .vals_b_row_2 = {60, 70},
      .mask_a_row_1 = {true, true, true, false, false},
      .mask_a_row_2 = {true, true},
      .mask_b_row_1 = {true, true, true, false, false},
      .mask_b_row_2 = {true, true},
    },
    {
      .max_sequence_length = 3,
      .vals_a_row_1 = {1, 2},
      .vals_a_row_2 = {6, 7},
      .vals_b_row_1 = {10},
      .vals_b_row_2 = {60},
      .mask_a_row_1 = {true, true, false, false, false},
      .mask_a_row_2 = {true, true},
      .mask_b_row_1 = {true, false, false, false, false},
      .mask_b_row_2 = {true, false},
    },
    {
      .max_sequence_length = 0,
      .vals_a_row_1 = {},
      .vals_a_row_2 = {},
      .vals_b_row_1 = {},
      .vals_b_row_2 = {},
      .mask_a_row_1 = {false, false, false, false, false},
      .mask_a_row_2 = {false, false},
      .mask_b_row_1 = {false, false, false, false, false},
      .mask_b_row_2 = {false, false},
    }
};

TEST_P(RoundRobinTrimmerTest, GenerateMasks) {
  TestSpec p = GetParam();
  RoundRobinTrimmer<int, int> t(p.max_sequence_length);
  std::vector<Masks> masks1 = t.GenerateMasks(GetFirstBatch());
  EXPECT_THAT(masks1[0], ElementsAreArray(p.mask_a_row_1));
  EXPECT_THAT(masks1[1], ElementsAreArray(p.mask_b_row_1));
  std::vector<Masks> masks2 = t.GenerateMasks(GetSecondBatch());
  EXPECT_THAT(masks2[0], ElementsAreArray(p.mask_a_row_2));
  EXPECT_THAT(masks2[1], ElementsAreArray(p.mask_b_row_2));
}

TEST_P(RoundRobinTrimmerTest, GenerateMasks_flat) {
  TestSpec p = GetParam();
  RoundRobinTrimmer<int, int> t(p.max_sequence_length);
  std::vector<Masks> masks = t.GenerateMasksBatch(GetFlatInput().second);
  EXPECT_THAT(masks[0],
              ElementsAreArray(Concat<bool>(p.mask_a_row_1, p.mask_a_row_2)));
  EXPECT_THAT(masks[1],
              ElementsAreArray(Concat<bool>(p.mask_b_row_1, p.mask_b_row_2)));
}

TEST_P(RoundRobinTrimmerTest, Trim) {
  TestSpec p = GetParam();
  RoundRobinTrimmer<int, int> t(p.max_sequence_length);
  std::vector<Segment> vals1 = GetFirstBatch();
  t.Trim(&vals1);
  EXPECT_THAT(vals1[0], ElementsAreArray(p.vals_a_row_1));
  EXPECT_THAT(vals1[1], ElementsAreArray(p.vals_b_row_1));
  std::vector<Segment> vals2 = GetSecondBatch();
  t.Trim(&vals2);
  EXPECT_THAT(vals2[0], ElementsAreArray(p.vals_a_row_2));
  EXPECT_THAT(vals2[1], ElementsAreArray(p.vals_b_row_2));
}

TEST_P(RoundRobinTrimmerTest, Trim_flat) {
  TestSpec p = GetParam();
  RoundRobinTrimmer<int, int> t(p.max_sequence_length);
  auto [input_vals, input_splits] = GetFlatInput();
  auto [vals, splits] = t.TrimBatch(input_vals, input_splits);
  EXPECT_THAT(vals[0],
              ElementsAreArray(Concat<int>(p.vals_a_row_1, p.vals_a_row_2)));
  EXPECT_THAT(vals[1],
              ElementsAreArray(Concat<int>(p.vals_b_row_1, p.vals_b_row_2)));
  std::vector<int> result_splits = { 0 };
  result_splits.push_back(p.vals_a_row_1.size());
  result_splits.push_back(p.vals_a_row_1.size() + p.vals_a_row_2.size());
  EXPECT_THAT(splits[0], ElementsAreArray(result_splits));
  result_splits = { 0 };
  result_splits.push_back(p.vals_b_row_1.size());
  result_splits.push_back(p.vals_b_row_1.size() + p.vals_b_row_2.size());
  EXPECT_THAT(splits[1], ElementsAreArray(result_splits));
}

TEST_P(RoundRobinTrimmerTest, Trim_int64) {
  TestSpec p = GetParam();
  RoundRobinTrimmer<int64_t, int64_t> t(p.max_sequence_length);
  auto [input_vals, input_splits] = GetFlatInput();
  std::vector<std::vector<int64_t>> input_splits_64(input_splits.size());
  for (int i = 0; i < input_splits.size(); ++i)
      input_splits_64[i].insert(input_splits_64[i].end(),
          input_splits[i].begin(), input_splits[i].end());
  std::vector<std::vector<int64_t>> input_vals_64(input_vals.size());
  for (int i = 0; i < input_vals.size(); ++i)
      input_vals_64[i].insert(input_vals_64[i].end(),
          input_vals[i].begin(), input_vals[i].end());
  auto [vals, splits] = t.TrimBatch(input_vals_64, input_splits_64);
  EXPECT_THAT(vals[0],
              ElementsAreArray(Concat<int>(p.vals_a_row_1, p.vals_a_row_2)));
  EXPECT_THAT(vals[1],
              ElementsAreArray(Concat<int>(p.vals_b_row_1, p.vals_b_row_2)));
  std::vector<int> result_splits = { 0 };
  result_splits.push_back(p.vals_a_row_1.size());
  result_splits.push_back(p.vals_a_row_1.size() + p.vals_a_row_2.size());
  EXPECT_THAT(splits[0], ElementsAreArray(result_splits));
  result_splits = { 0 };
  result_splits.push_back(p.vals_b_row_1.size());
  result_splits.push_back(p.vals_b_row_1.size() + p.vals_b_row_2.size());
  EXPECT_THAT(splits[1], ElementsAreArray(result_splits));
}

INSTANTIATE_TEST_SUITE_P(RoundRobinTrimmerTestSuite,
                         RoundRobinTrimmerTest,
                         testing::ValuesIn(params));

}  // namespace
}  // namespace text
}  // namespace tensorflow
