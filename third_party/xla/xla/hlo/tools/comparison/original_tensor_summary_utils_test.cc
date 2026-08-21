/* Copyright 2026 The OpenXLA Authors.

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

#include "xla/hlo/tools/comparison/original_tensor_summary_utils.h"

#include <algorithm>
#include <cstdint>
#include <memory>
#include <tuple>
#include <vector>

#include "absl/hash/hash_testing.h"
#include "absl/status/status.h"
#include "absl/strings/str_cat.h"
#include "absl/types/span.h"
#include "google/protobuf/repeated_ptr_field.h"
#include "xla/array.h"
#include "xla/hlo/ir/hlo_sharding.h"
#include "xla/hlo/tools/comparison/comparison_result.pb.h"
#include "xla/hlo/tools/comparison/tensor_summary_util.h"
#include "xla/tools/debug_event.pb.h"
#include "xla/tsl/platform/status_matchers.h"
#include "xla/tsl/platform/test.h"
#include "xla/tsl/util/proto/proto_matchers.h"

namespace xla::numerics::comparison {
namespace {

using ::testing::ElementsAreArray;
using ::testing::FloatEq;
using ::testing::Matches;
using ::testing::UnorderedPointwise;
using ::tsl::proto_testing::EqualsProto;
using ::tsl::testing::StatusIs;
using ::xla::LogHloOutputMetadata;

using FloatBlockSummary = ::xla::comparison::FloatBlockSummary;
using FloatSummary = ::xla::comparison::FloatSummary;
using TensorTransformation = tensor_transformation::TensorTransformation;
using DimSplitSpec = ::xla::comparison::DimSplitSpec;
using Reshape = tensor_transformation::Reshape;
using Unshard = tensor_transformation::Unshard;
using Broadcast = tensor_transformation::Broadcast;

FloatBlockSummary CreateBlockSummary(absl::Span<const int64_t> block_indices,
                                     float count, float min_val = 0.0f,
                                     float max_val = 0.0f) {
  return {
      /*block_indices=*/{block_indices.begin(), block_indices.end()},
      /*min=*/min_val,
      /*max=*/max_val,
      /*mean=*/0.0,
      /*stddev=*/0.0,
      /*count=*/count,
      /*nan_count=*/0.0,
      /*pos_inf_count=*/0.0,
      /*neg_inf_count=*/0.0,
      /*zero_count=*/0.0,
  };
}

// Ignores mean and stddev
MATCHER(FloatBlockSummaryEqImpl, "") {
  const auto& a = ::testing::get<0>(arg);
  const auto& b = ::testing::get<1>(arg);
  return a.block_indices == b.block_indices && Matches(FloatEq(a.min))(b.min) &&
         Matches(FloatEq(a.max))(b.max) && Matches(FloatEq(a.count))(b.count);
}

MATCHER_P(FloatSummaryEq, expected, "") {
  return Matches(ElementsAreArray(expected.split_spec))(arg.split_spec) &&
         Matches(UnorderedPointwise(FloatBlockSummaryEqImpl(),
                                    expected.block_summaries))(
             arg.block_summaries);
}

// No splits in summary.
TEST(ApplyReshapeToSummaryTest, NoSplits) {
  FloatSummary summary;
  summary.block_summaries.push_back(CreateBlockSummary({}, 120));
  const std::vector<int64_t> current_shape = {10, 12};
  const std::vector<int64_t> new_shape = {120};

  FloatSummary result =
      ApplyReshapeToSummary(summary, current_shape, new_shape);
  EXPECT_THAT(result, FloatSummaryEq(summary));
}

// Reshape with no change to split dimensions.
// shape [2, 3, 4, 5] -> [2, 12, 5], split on dim 0 and 3.
TEST(ApplyReshapeToSummaryTest, ReshapeWithoutChangingSplits) {
  FloatSummary summary;
  summary.split_spec = {{/*dim_index=*/0, /*block_count=*/2},
                        {/*dim_index=*/3, /*block_count=*/5}};
  summary.block_summaries = {
      CreateBlockSummary({0, 0}, 12), CreateBlockSummary({0, 1}, 12),
      CreateBlockSummary({0, 2}, 12), CreateBlockSummary({0, 3}, 12),
      CreateBlockSummary({0, 4}, 12), CreateBlockSummary({1, 0}, 12),
      CreateBlockSummary({1, 1}, 12), CreateBlockSummary({1, 2}, 12),
      CreateBlockSummary({1, 3}, 12), CreateBlockSummary({1, 4}, 12),
  };
  const std::vector<int64_t> current_shape = {2, 3, 4, 5};
  const std::vector<int64_t> new_shape = {2, 12, 5};

  FloatSummary result =
      ApplyReshapeToSummary(summary, current_shape, new_shape);

  FloatSummary expected_summary;
  expected_summary.split_spec = {{/*dim_index=*/0, /*block_count=*/2},
                                 {/*dim_index=*/2, /*block_count=*/5}};
  expected_summary.block_summaries = summary.block_summaries;
  EXPECT_THAT(result, FloatSummaryEq(expected_summary));
}

// Reshape merges split dimensions.
// shape [2, 6] -> [12], split on dim 0 and 1.
TEST(ApplyReshapeToSummaryTest, ReshapeMergesSplits) {
  FloatSummary summary;
  summary.split_spec = {{/*dim_index=*/0, /*block_count=*/2},
                        {/*dim_index=*/1, /*block_count=*/2}};
  summary.block_summaries = {
      CreateBlockSummary({0, 0}, 3, 0.0, 1.0),
      CreateBlockSummary({0, 1}, 3, 0.0, 2.0),
      CreateBlockSummary({1, 0}, 3, 0.0, 3.0),
      CreateBlockSummary({1, 1}, 3, 0.0, 4.0),
  };
  const std::vector<int64_t> current_shape = {2, 6};
  const std::vector<int64_t> new_shape = {12};

  FloatSummary result =
      ApplyReshapeToSummary(summary, current_shape, new_shape);

  FloatSummary expected_summary;
  expected_summary.split_spec = {};
  expected_summary.block_summaries = {CreateBlockSummary({}, 12, 0.0, 4.0)};
  EXPECT_THAT(result, FloatSummaryEq(expected_summary));
}

// Reshape merges some split dimensions but not others.
// shape [2, 3, 4, 5] -> [2, 12, 5], split on dim 0, 1, 3.
// Dim 1 split should be merged. Dims 0 and 3 splits should be preserved.
TEST(ApplyReshapeToSummaryTest, ReshapeMergesSomeSplits) {
  FloatSummary summary;
  summary.split_spec = {{/*dim_index=*/0, /*block_count=*/2},
                        {/*dim_index=*/1, /*block_count=*/3},
                        {/*dim_index=*/3, /*block_count=*/5}};
  for (int i = 0; i < 2; ++i) {
    for (int j = 0; j < 3; ++j) {
      for (int k = 0; k < 5; ++k) {
        summary.block_summaries.push_back(CreateBlockSummary(
            {(int64_t)i, (int64_t)j, (int64_t)k}, 4, 0.0, (float)j));
      }
    }
  }

  const std::vector<int64_t> current_shape = {2, 3, 4, 5};
  const std::vector<int64_t> new_shape = {2, 12, 5};

  FloatSummary result =
      ApplyReshapeToSummary(summary, current_shape, new_shape);

  FloatSummary expected_summary;
  // New split spec should be on dim 0 of new shape, and dim 2 of new shape
  // (from dim 3 of old shape).
  expected_summary.split_spec = {{/*dim_index=*/0, /*block_count=*/2},
                                 {/*dim_index=*/2, /*block_count=*/5}};
  // Each new block combines 3 old blocks, so count is 4*3=12.
  // Max comes from j=2, so max should be 2.0 for all blocks.
  expected_summary.block_summaries = {
      CreateBlockSummary({0, 0}, 12, 0.0, 2.0),
      CreateBlockSummary({0, 1}, 12, 0.0, 2.0),
      CreateBlockSummary({0, 2}, 12, 0.0, 2.0),
      CreateBlockSummary({0, 3}, 12, 0.0, 2.0),
      CreateBlockSummary({0, 4}, 12, 0.0, 2.0),
      CreateBlockSummary({1, 0}, 12, 0.0, 2.0),
      CreateBlockSummary({1, 1}, 12, 0.0, 2.0),
      CreateBlockSummary({1, 2}, 12, 0.0, 2.0),
      CreateBlockSummary({1, 3}, 12, 0.0, 2.0),
      CreateBlockSummary({1, 4}, 12, 0.0, 2.0),
  };

  EXPECT_THAT(result, FloatSummaryEq(expected_summary));
}

// 1D to 2D reshape.
// shape [12] -> [3, 4], split on dim 0.
TEST(ApplyReshapeToSummaryTest, 1DTo2D) {
  FloatSummary summary;
  summary.split_spec = {{/*dim_index=*/0, /*block_count=*/3}};
  summary.block_summaries = {
      CreateBlockSummary({0}, 4, 0.0, 1.0),
      CreateBlockSummary({1}, 4, 0.0, 2.0),
      CreateBlockSummary({2}, 4, 0.0, 3.0),
  };
  const std::vector<int64_t> current_shape = {12};
  const std::vector<int64_t> new_shape = {3, 4};

  FloatSummary result =
      ApplyReshapeToSummary(summary, current_shape, new_shape);

  FloatSummary expected_summary;
  expected_summary.split_spec = {};
  expected_summary.block_summaries = {CreateBlockSummary({}, 12, 0.0, 3.0)};
  EXPECT_THAT(result, FloatSummaryEq(expected_summary));
}

// 2D to 1D reshape, suffix match
// shape [1, 12] -> [12], no split
TEST(ApplyReshapeToSummaryTest, 2DTo1DSuffix) {
  FloatSummary summary;
  summary.block_summaries.push_back(CreateBlockSummary({}, 12));
  const std::vector<int64_t> current_shape = {1, 12};
  const std::vector<int64_t> new_shape = {12};
  FloatSummary result =
      ApplyReshapeToSummary(summary, current_shape, new_shape);
  EXPECT_THAT(result, FloatSummaryEq(summary));
}

// No common prefix or suffix
// shape [2, 3] -> [3, 2], split on dim 0
TEST(ApplyReshapeToSummaryTest, NoCommonPrefixOrSuffix) {
  FloatSummary summary;
  summary.split_spec = {{/*dim_index=*/0, /*block_count=*/2}};
  summary.block_summaries = {
      CreateBlockSummary({0}, 3, 0.0, 1.0),
      CreateBlockSummary({1}, 3, 0.0, 2.0),
  };
  const std::vector<int64_t> current_shape = {2, 3};
  const std::vector<int64_t> new_shape = {3, 2};
  FloatSummary result =
      ApplyReshapeToSummary(summary, current_shape, new_shape);
  FloatSummary expected_summary;
  expected_summary.split_spec = {};
  expected_summary.block_summaries = {CreateBlockSummary({}, 6, 0.0, 2.0)};
  EXPECT_THAT(result, FloatSummaryEq(expected_summary));
}

// in [2,3,4], out [6,4], split on dim 0,1,2
// prefix empty, suffix dim 2 -> dim 1
TEST(ApplyReshapeToSummaryTest, PrefixEmptySuffixOne) {
  FloatSummary summary;
  summary.split_spec = {
      {/*dim_index=*/0, /*block_count=*/2},
      {/*dim_index=*/1, /*block_count=*/3},
      {/*dim_index=*/2, /*block_count=*/4},
  };
  for (int i = 0; i < 2; ++i) {
    for (int j = 0; j < 3; ++j) {
      for (int k = 0; k < 4; ++k) {
        summary.block_summaries.push_back(CreateBlockSummary(
            {(int64_t)i, (int64_t)j, (int64_t)k}, 1, 0.0, (float)k));
      }
    }
  }

  const std::vector<int64_t> current_shape = {2, 3, 4};
  const std::vector<int64_t> new_shape = {6, 4};
  FloatSummary result =
      ApplyReshapeToSummary(summary, current_shape, new_shape);

  FloatSummary expected_summary;
  expected_summary.split_spec = {{/*dim_index=*/1, /*block_count=*/4}};
  expected_summary.block_summaries = {
      CreateBlockSummary({0}, 6, 0.0, 0.0),
      CreateBlockSummary({1}, 6, 0.0, 1.0),
      CreateBlockSummary({2}, 6, 0.0, 2.0),
      CreateBlockSummary({3}, 6, 0.0, 3.0),
  };
  EXPECT_THAT(result, FloatSummaryEq(expected_summary));
}

TEST(ApplyTransposeToSummaryTest, NoSplits) {
  FloatSummary summary;
  summary.block_summaries.push_back(CreateBlockSummary({}, 120));
  const std::vector<int64_t> current_shape = {10, 12};
  const std::vector<int64_t> permutation = {1, 0};

  FloatSummary result =
      ApplyTransposeToSummary(summary, current_shape, permutation);
  EXPECT_THAT(result, FloatSummaryEq(summary));
}

TEST(ApplyTransposeToSummaryTest, TransposeExample) {
  FloatSummary summary;
  summary.split_spec = {
      {/*dim_index=*/0, /*block_count=*/2},
      {/*dim_index=*/1, /*block_count=*/4},
      {/*dim_index=*/3, /*block_count=*/5},
  };
  for (int64_t i = 0; i < 2; ++i) {
    for (int64_t j = 0; j < 4; ++j) {
      for (int64_t k = 0; k < 5; ++k) {
        summary.block_summaries.push_back(
            CreateBlockSummary({i, j, k}, 1, i, j));
      }
    }
  }
  const std::vector<int64_t> current_shape = {10, 20, 30, 40};
  const std::vector<int64_t> permutation = {1, 3, 0, 2};

  FloatSummary result =
      ApplyTransposeToSummary(summary, current_shape, permutation);

  FloatSummary expected_summary;
  expected_summary.split_spec = {{/*dim_index=*/0, /*block_count=*/4},
                                 {/*dim_index=*/1, /*block_count=*/5},
                                 {/*dim_index=*/2, /*block_count=*/2}};
  for (int i = 0; i < 2; ++i) {
    for (int j = 0; j < 4; ++j) {
      for (int k = 0; k < 5; ++k) {
        expected_summary.block_summaries.push_back(
            CreateBlockSummary({j, k, i}, 1, i, j));
      }
    }
  }
  EXPECT_THAT(result, FloatSummaryEq(expected_summary));
}

TEST(ApplyBroadcastToSummaryTest, NoSplits) {
  FloatSummary summary;
  summary.block_summaries.push_back(CreateBlockSummary({}, 120));
  const std::vector<int64_t> current_shape = {10, 12};
  const std::vector<int64_t> new_shape = {3, 10, 12};
  const std::vector<int64_t> broadcast_dimensions = {1, 2};

  FloatSummary result = ApplyBroadcastToSummary(
      summary, current_shape, new_shape, broadcast_dimensions);

  FloatSummary expected_summary;
  expected_summary.block_summaries.push_back(CreateBlockSummary({}, 360));
  EXPECT_THAT(result, FloatSummaryEq(expected_summary));
}

// Example 1 from ApplyBroadcastToSummary documentation.
TEST(ApplyBroadcastToSummaryTest, BroadcastExample1) {
  FloatSummary summary;
  summary.split_spec = {{/*dim_index=*/0, /*block_count=*/2}};
  summary.block_summaries = {
      CreateBlockSummary({0}, 3, 0.0, 1.0),
      CreateBlockSummary({1}, 3, 1.0, 2.0),
  };
  const std::vector<int64_t> current_shape = {2, 3};
  const std::vector<int64_t> new_shape = {4, 2, 3};
  const std::vector<int64_t> broadcast_dimensions = {1, 2};

  FloatSummary result = ApplyBroadcastToSummary(
      summary, current_shape, new_shape, broadcast_dimensions);

  FloatSummary expected_summary;
  expected_summary.split_spec = {{/*dim_index=*/1, /*block_count=*/2}};
  expected_summary.block_summaries = {
      CreateBlockSummary({0}, 12, 0.0, 1.0),
      CreateBlockSummary({1}, 12, 1.0, 2.0),
  };
  EXPECT_THAT(result, FloatSummaryEq(expected_summary));
}

// Example 2 from ApplyBroadcastToSummary documentation, with input split spec
// dimensions not in increasing order to test sorting.
TEST(ApplyBroadcastToSummaryTest, BroadcastExample2) {
  FloatSummary summary;
  summary.split_spec = {{/*dim_index=*/1, /*block_count=*/3},
                        {/*dim_index=*/0, /*block_count=*/2}};
  for (int j = 0; j < 3; ++j) {
    for (int i = 0; i < 2; ++i) {
      summary.block_summaries.push_back(
          CreateBlockSummary({(int64_t)j, (int64_t)i}, 1, i, j));
    }
  }

  const std::vector<int64_t> current_shape = {2, 3};
  const std::vector<int64_t> new_shape = {4, 2, 3};
  const std::vector<int64_t> broadcast_dimensions = {1, 2};

  FloatSummary result = ApplyBroadcastToSummary(
      summary, current_shape, new_shape, broadcast_dimensions);

  FloatSummary expected_summary;
  expected_summary.split_spec = {{/*dim_index=*/1, /*block_count=*/2},
                                 {/*dim_index=*/2, /*block_count=*/3}};
  for (int i = 0; i < 2; ++i) {
    for (int j = 0; j < 3; ++j) {
      expected_summary.block_summaries.push_back(
          CreateBlockSummary({(int64_t)i, (int64_t)j}, 4, i, j));
    }
  }
  EXPECT_THAT(result, FloatSummaryEq(expected_summary));
}

TEST(ApplyBroadcastToSummaryTest, BroadcastWithTranspose) {
  FloatSummary summary;
  summary.split_spec = {{/*dim_index=*/0, /*block_count=*/2},
                        {/*dim_index=*/1, /*block_count=*/3}};
  for (int i = 0; i < 2; ++i) {
    for (int j = 0; j < 3; ++j) {
      summary.block_summaries.push_back(
          CreateBlockSummary({(int64_t)i, (int64_t)j}, 1, i, j));
    }
  }
  const std::vector<int64_t> current_shape = {2, 3};
  const std::vector<int64_t> new_shape = {3, 4, 2};
  const std::vector<int64_t> broadcast_dimensions = {2, 0};

  FloatSummary result = ApplyBroadcastToSummary(
      summary, current_shape, new_shape, broadcast_dimensions);

  FloatSummary expected_summary;
  expected_summary.split_spec = {{/*dim_index=*/0, /*block_count=*/3},
                                 {/*dim_index=*/2, /*block_count=*/2}};
  for (int j = 0; j < 3; ++j) {
    for (int i = 0; i < 2; ++i) {
      expected_summary.block_summaries.push_back(
          CreateBlockSummary({(int64_t)j, (int64_t)i}, 4, i, j));
    }
  }
  EXPECT_THAT(result, FloatSummaryEq(expected_summary));
}

TEST(ApplyBroadcastToSummaryTest, BroadcastWithDimensionsInsertedBetween) {
  FloatSummary summary;
  summary.split_spec = {{/*dim_index=*/0, /*block_count=*/2},
                        {/*dim_index=*/1, /*block_count=*/3}};
  for (int i = 0; i < 2; ++i) {
    for (int j = 0; j < 3; ++j) {
      summary.block_summaries.push_back(
          CreateBlockSummary({(int64_t)i, (int64_t)j}, 1, i, j));
    }
  }
  const std::vector<int64_t> current_shape = {2, 3};
  const std::vector<int64_t> new_shape = {5, 2, 4, 3, 6};
  const std::vector<int64_t> broadcast_dimensions = {1, 3};

  FloatSummary result = ApplyBroadcastToSummary(
      summary, current_shape, new_shape, broadcast_dimensions);

  FloatSummary expected_summary;
  expected_summary.split_spec = {{/*dim_index=*/1, /*block_count=*/2},
                                 {/*dim_index=*/3, /*block_count=*/3}};
  for (int i = 0; i < 2; ++i) {
    for (int j = 0; j < 3; ++j) {
      expected_summary.block_summaries.push_back(
          CreateBlockSummary({(int64_t)i, (int64_t)j}, 120, i, j));
    }
  }
  EXPECT_THAT(result, FloatSummaryEq(expected_summary));
}

// Example 3 from ApplyBroadcastToSummary documentation.
TEST(ApplyBroadcastToSummaryTest, InverseBroadcastExample3) {
  FloatSummary summary;
  summary.split_spec = {{/*dim_index=*/0, /*block_count=*/2},
                        {/*dim_index=*/1, /*block_count=*/2}};
  summary.block_summaries = {
      CreateBlockSummary({0, 0}, 6, 0.0, 0.0),
      CreateBlockSummary({0, 1}, 6, 1.0, 1.0),
      CreateBlockSummary({1, 0}, 6, 2.0, 2.0),
      CreateBlockSummary({1, 1}, 6, 3.0, 3.0),
  };
  const std::vector<int64_t> current_shape = {4, 2, 3};
  const std::vector<int64_t> new_shape = {2, 3};
  const std::vector<int64_t> broadcast_dimensions = {-1, 0, 1};

  FloatSummary result = ApplyBroadcastToSummary(
      summary, current_shape, new_shape, broadcast_dimensions);

  FloatSummary expected_summary;
  expected_summary.split_spec = {{/*dim_index=*/0, /*block_count=*/2}};
  expected_summary.block_summaries = {
      CreateBlockSummary({0}, 3, 0.0, 2.0),
      CreateBlockSummary({1}, 3, 1.0, 3.0),
  };
  EXPECT_THAT(result, FloatSummaryEq(expected_summary));
}

// Inverse broadcast where a dropped dimension is not split.
TEST(ApplyBroadcastToSummaryTest, InverseBroadcastDropUnsplitDimension) {
  FloatSummary summary;
  summary.split_spec = {{/*dim_index=*/1, /*block_count=*/2}};
  summary.block_summaries = {
      CreateBlockSummary({0}, 12, 0.0, 0.0),
      CreateBlockSummary({1}, 12, 1.0, 1.0),
  };
  const std::vector<int64_t> current_shape = {4, 2, 3};
  const std::vector<int64_t> new_shape = {2, 3};
  const std::vector<int64_t> broadcast_dimensions = {-1, 0, 1};

  FloatSummary result = ApplyBroadcastToSummary(
      summary, current_shape, new_shape, broadcast_dimensions);

  FloatSummary expected_summary;
  expected_summary.split_spec = {{/*dim_index=*/0, /*block_count=*/2}};
  expected_summary.block_summaries = {
      CreateBlockSummary({0}, 3, 0.0, 0.0),
      CreateBlockSummary({1}, 3, 1.0, 1.0),
  };
  EXPECT_THAT(result, FloatSummaryEq(expected_summary));
}

// Inverse broadcast dropping multiple dimensions, one split and one not split.
TEST(ApplyBroadcastToSummaryTest, InverseBroadcastDropMultipleDimensions) {
  FloatSummary summary;
  summary.split_spec = {{/*dim_index=*/0, /*block_count=*/2},
                        {/*dim_index=*/1, /*block_count=*/2}};
  summary.block_summaries = {
      CreateBlockSummary({0, 0}, 15, 0.0, 0.0),
      CreateBlockSummary({0, 1}, 15, 1.0, 1.0),
      CreateBlockSummary({1, 0}, 15, 2.0, 2.0),
      CreateBlockSummary({1, 1}, 15, 3.0, 3.0),
  };
  const std::vector<int64_t> current_shape = {4, 2, 3, 5};
  const std::vector<int64_t> new_shape = {2, 3};
  const std::vector<int64_t> broadcast_dimensions = {-1, 0, 1, -1};

  FloatSummary result = ApplyBroadcastToSummary(
      summary, current_shape, new_shape, broadcast_dimensions);

  FloatSummary expected_summary;
  expected_summary.split_spec = {{/*dim_index=*/0, /*block_count=*/2}};
  expected_summary.block_summaries = {
      CreateBlockSummary({0}, 1.5, 0.0, 2.0),
      CreateBlockSummary({1}, 1.5, 1.0, 3.0),
  };
  EXPECT_THAT(result, FloatSummaryEq(expected_summary));
}

// Inverse broadcast resulting in a scalar.
TEST(ApplyBroadcastToSummaryTest, InverseBroadcastToScalar) {
  FloatSummary summary;
  summary.split_spec = {{/*dim_index=*/0, /*block_count=*/2}};
  summary.block_summaries = {
      CreateBlockSummary({0}, 2, 0.0, 0.0),
      CreateBlockSummary({1}, 2, 1.0, 1.0),
  };
  const std::vector<int64_t> current_shape = {4};
  const std::vector<int64_t> new_shape = {};
  const std::vector<int64_t> broadcast_dimensions = {-1};

  FloatSummary result = ApplyBroadcastToSummary(
      summary, current_shape, new_shape, broadcast_dimensions);

  FloatSummary expected_summary;
  expected_summary.split_spec = {};
  expected_summary.block_summaries = {
      CreateBlockSummary({}, 1, 0.0, 1.0),
  };
  EXPECT_THAT(result, FloatSummaryEq(expected_summary));
}

TEST(ApplyNonUnshardTensorTransformationToSummaryTest, NoTransformation) {
  OriginalTensorSummary original_summary;
  original_summary.summaries.emplace_back();
  original_summary.summaries[0].block_summaries.push_back(
      CreateBlockSummary({}, 120));
  original_summary.dimensions = {10, 12};

  ASSERT_OK_AND_ASSIGN(OriginalTensorSummary result,
                       ApplyNonUnshardTensorTransformationToSummary(
                           original_summary,
                           /*transformation=*/nullptr,
                           /*stopping_transformation=*/nullptr));
  ASSERT_EQ(result.summaries.size(), 1);
  EXPECT_THAT(result.summaries[0],
              FloatSummaryEq(original_summary.summaries[0]));
  EXPECT_EQ(result.dimensions, original_summary.dimensions);
}

TEST(ApplyNonUnshardTensorTransformationToSummaryTest, SingleReshape) {
  OriginalTensorSummary original_summary;
  original_summary.dimensions = {2, 6};
  original_summary.summaries.emplace_back();
  original_summary.summaries[0].split_spec = {
      {/*dim_index=*/0, /*block_count=*/2},
      {/*dim_index=*/1, /*block_count=*/2}};
  original_summary.summaries[0].block_summaries = {
      CreateBlockSummary({0, 0}, 3, 0.0, 1.0),
      CreateBlockSummary({0, 1}, 3, 0.0, 2.0),
      CreateBlockSummary({1, 0}, 3, 0.0, 3.0),
      CreateBlockSummary({1, 1}, 3, 0.0, 4.0),
  };
  TensorTransformation reshape =
      Reshape{/*continuation=*/nullptr, /*output_dimensions=*/{12}};

  ASSERT_OK_AND_ASSIGN(OriginalTensorSummary result,
                       ApplyNonUnshardTensorTransformationToSummary(
                           original_summary, &reshape,
                           /*stopping_transformation=*/nullptr));

  FloatSummary expected_summary;
  expected_summary.split_spec = {};
  expected_summary.block_summaries = {CreateBlockSummary({}, 12, 0.0, 4.0)};
  ASSERT_EQ(result.summaries.size(), 1);
  EXPECT_THAT(result.summaries[0], FloatSummaryEq(expected_summary));
  EXPECT_EQ(result.dimensions, std::vector<int64_t>({12}));
}

TEST(ApplyNonUnshardTensorTransformationToSummaryTest, MultipleReshapes) {
  OriginalTensorSummary original_summary;
  original_summary.dimensions = {2, 3, 4, 5};
  original_summary.summaries.emplace_back();
  original_summary.summaries[0].split_spec = {
      {/*dim_index=*/0, /*block_count=*/2},
      {/*dim_index=*/1, /*block_count=*/3},
      {/*dim_index=*/3, /*block_count=*/5}};
  for (int i = 0; i < 2; ++i) {
    for (int j = 0; j < 3; ++j) {
      for (int k = 0; k < 5; ++k) {
        original_summary.summaries[0].block_summaries.push_back(
            CreateBlockSummary({(int64_t)i, (int64_t)j, (int64_t)k}, 4, 0.0,
                               (float)j));
      }
    }
  }

  auto reshape2 = std::make_shared<const TensorTransformation>(
      Reshape{/*continuation=*/nullptr, /*output_dimensions=*/{120}});
  TensorTransformation reshape1 =
      Reshape{/*continuation=*/reshape2, /*output_dimensions=*/{2, 12, 5}};

  ASSERT_OK_AND_ASSIGN(OriginalTensorSummary result,
                       ApplyNonUnshardTensorTransformationToSummary(
                           original_summary, &reshape1,
                           /*stopping_transformation=*/nullptr));

  FloatSummary expected_summary;
  expected_summary.split_spec = {};
  expected_summary.block_summaries = {CreateBlockSummary({}, 120, 0.0, 2.0)};
  ASSERT_EQ(result.summaries.size(), 1);
  EXPECT_THAT(result.summaries[0], FloatSummaryEq(expected_summary));
  EXPECT_EQ(result.dimensions, std::vector<int64_t>({120}));
}

TEST(ApplyNonUnshardTensorTransformationToSummaryTest, StopTransformation) {
  OriginalTensorSummary original_summary;
  original_summary.dimensions = {2, 3, 4, 5};
  original_summary.summaries.emplace_back();
  original_summary.summaries[0].split_spec = {
      {/*dim_index=*/0, /*block_count=*/2},
      {/*dim_index=*/1, /*block_count=*/3},
      {/*dim_index=*/3, /*block_count=*/5}};
  for (int i = 0; i < 2; ++i) {
    for (int j = 0; j < 3; ++j) {
      for (int k = 0; k < 5; ++k) {
        original_summary.summaries[0].block_summaries.push_back(
            CreateBlockSummary({(int64_t)i, (int64_t)j, (int64_t)k}, 4, 0.0,
                               (float)j));
      }
    }
  }

  auto reshape2 = std::make_shared<const TensorTransformation>(
      Reshape{/*continuation=*/nullptr, /*output_dimensions=*/{120}});
  TensorTransformation reshape1 =
      Reshape{/*continuation=*/reshape2, /*output_dimensions=*/{2, 12, 5}};

  ASSERT_OK_AND_ASSIGN(OriginalTensorSummary result,
                       ApplyNonUnshardTensorTransformationToSummary(
                           original_summary, &reshape1,
                           /*stopping_transformation=*/reshape2.get()));

  FloatSummary expected_summary;
  expected_summary.split_spec = {{/*dim_index=*/0, /*block_count=*/2},
                                 {/*dim_index=*/2, /*block_count=*/5}};
  expected_summary.block_summaries = {
      CreateBlockSummary({0, 0}, 12, 0.0, 2.0),
      CreateBlockSummary({0, 1}, 12, 0.0, 2.0),
      CreateBlockSummary({0, 2}, 12, 0.0, 2.0),
      CreateBlockSummary({0, 3}, 12, 0.0, 2.0),
      CreateBlockSummary({0, 4}, 12, 0.0, 2.0),
      CreateBlockSummary({1, 0}, 12, 0.0, 2.0),
      CreateBlockSummary({1, 1}, 12, 0.0, 2.0),
      CreateBlockSummary({1, 2}, 12, 0.0, 2.0),
      CreateBlockSummary({1, 3}, 12, 0.0, 2.0),
      CreateBlockSummary({1, 4}, 12, 0.0, 2.0),
  };
  ASSERT_EQ(result.summaries.size(), 1);
  EXPECT_THAT(result.summaries[0], FloatSummaryEq(expected_summary));
  EXPECT_EQ(result.dimensions, std::vector<int64_t>({2, 12, 5}));
}

TEST(ApplyNonUnshardTensorTransformationToSummaryTest,
     StopTransformationByValue) {
  OriginalTensorSummary original_summary;
  original_summary.dimensions = {2, 3, 4, 5};
  original_summary.summaries.emplace_back();
  original_summary.summaries[0].split_spec = {
      {/*dim_index=*/0, /*block_count=*/2},
      {/*dim_index=*/1, /*block_count=*/3},
      {/*dim_index=*/3, /*block_count=*/5}};
  for (int i = 0; i < 2; ++i) {
    for (int j = 0; j < 3; ++j) {
      for (int k = 0; k < 5; ++k) {
        original_summary.summaries[0].block_summaries.push_back(
            CreateBlockSummary({(int64_t)i, (int64_t)j, (int64_t)k}, 4, 0.0,
                               (float)j));
      }
    }
  }

  auto reshape2 = std::make_shared<const TensorTransformation>(
      Reshape{/*continuation=*/nullptr, /*output_dimensions=*/{120}});
  TensorTransformation reshape1 =
      Reshape{/*continuation=*/reshape2, /*output_dimensions=*/{2, 12, 5}};

  // stopping_transformation has same value as *reshape2, but is a
  // different object.
  TensorTransformation stopping_transformation =
      Reshape{/*continuation=*/nullptr, /*output_dimensions=*/{120}};
  ASSERT_EQ(*reshape2, stopping_transformation);
  ASSERT_NE(reshape2.get(), &stopping_transformation);

  ASSERT_OK_AND_ASSIGN(
      OriginalTensorSummary result,
      ApplyNonUnshardTensorTransformationToSummary(
          original_summary, &reshape1,
          /*stopping_transformation=*/&stopping_transformation));

  FloatSummary expected_summary;
  expected_summary.split_spec = {{/*dim_index=*/0, /*block_count=*/2},
                                 {/*dim_index=*/2, /*block_count=*/5}};
  expected_summary.block_summaries = {
      CreateBlockSummary({0, 0}, 12, 0.0, 2.0),
      CreateBlockSummary({0, 1}, 12, 0.0, 2.0),
      CreateBlockSummary({0, 2}, 12, 0.0, 2.0),
      CreateBlockSummary({0, 3}, 12, 0.0, 2.0),
      CreateBlockSummary({0, 4}, 12, 0.0, 2.0),
      CreateBlockSummary({1, 0}, 12, 0.0, 2.0),
      CreateBlockSummary({1, 1}, 12, 0.0, 2.0),
      CreateBlockSummary({1, 2}, 12, 0.0, 2.0),
      CreateBlockSummary({1, 3}, 12, 0.0, 2.0),
      CreateBlockSummary({1, 4}, 12, 0.0, 2.0),
  };
  ASSERT_EQ(result.summaries.size(), 1);
  EXPECT_THAT(result.summaries[0], FloatSummaryEq(expected_summary));
  EXPECT_EQ(result.dimensions, std::vector<int64_t>({2, 12, 5}));
}

TEST(ApplyNonUnshardTensorTransformationToSummaryTest, UnshardNotSupported) {
  OriginalTensorSummary original_summary;
  original_summary.dimensions = {12};
  original_summary.summaries.emplace_back();
  original_summary.summaries[0].block_summaries.push_back(
      CreateBlockSummary({}, 12));

  auto unshard = std::make_shared<const TensorTransformation>(
      Unshard{/*continuation=*/nullptr,
              /*original_dimensions=*/{2, 6},
              /*sharding=*/HloSharding::Replicate()});
  TensorTransformation reshape =
      Reshape{/*continuation=*/unshard, /*output_dimensions=*/{2, 6}};

  EXPECT_THAT(ApplyNonUnshardTensorTransformationToSummary(
                  original_summary, &reshape,
                  /*stopping_transformation=*/nullptr),
              StatusIs(absl::StatusCode::kInvalidArgument));
}

TEST(TensorTransformationTest, Equality) {
  auto reshape1 =
      Reshape{/*continuation=*/nullptr, /*output_dimensions=*/{2, 2}};
  auto reshape2 =
      Reshape{/*continuation=*/nullptr, /*output_dimensions=*/{2, 2}};
  auto reshape3 =
      Reshape{/*continuation=*/nullptr, /*output_dimensions=*/{2, 3}};
  EXPECT_EQ(reshape1, reshape2);
  EXPECT_NE(reshape1, reshape3);

  auto unshard1 = Unshard{/*continuation=*/nullptr,
                          /*original_dimensions=*/{4},
                          /*sharding=*/HloSharding::Replicate()};
  auto unshard2 = Unshard{/*continuation=*/nullptr,
                          /*original_dimensions=*/{4},
                          /*sharding=*/HloSharding::Replicate()};
  auto unshard3 = Unshard{/*continuation=*/nullptr,
                          /*original_dimensions=*/{5},
                          /*sharding=*/HloSharding::Replicate()};
  Array<int64_t> tile_assignment({2, 1});
  tile_assignment(0, 0) = 0;
  tile_assignment(1, 0) = 1;
  auto unshard4 = Unshard{/*continuation=*/nullptr,
                          /*original_dimensions=*/{4},
                          /*sharding=*/HloSharding::Tile(tile_assignment)};
  EXPECT_EQ(unshard1, unshard2);
  EXPECT_NE(unshard1, unshard3);
  EXPECT_NE(unshard1, unshard4);

  auto broadcast1 = Broadcast{/*continuation=*/nullptr,
                              /*output_dimensions=*/{2, 2},
                              /*broadcast_dimensions=*/{0, 1}};
  auto broadcast2 = Broadcast{/*continuation=*/nullptr,
                              /*output_dimensions=*/{2, 2},
                              /*broadcast_dimensions=*/{0, 1}};
  auto broadcast3 = Broadcast{/*continuation=*/nullptr,
                              /*output_dimensions=*/{2, 3},
                              /*broadcast_dimensions=*/{0, 1}};
  auto broadcast4 = Broadcast{/*continuation=*/nullptr,
                              /*output_dimensions=*/{2, 2},
                              /*broadcast_dimensions=*/{1, 0}};
  EXPECT_EQ(broadcast1, broadcast2);
  EXPECT_NE(broadcast1, broadcast3);
  EXPECT_NE(broadcast1, broadcast4);

  auto unshard_cont = std::make_shared<const TensorTransformation>(unshard1);
  auto reshape_with_cont1 =
      Reshape{/*continuation=*/unshard_cont, /*output_dimensions=*/{2, 2}};
  auto reshape_with_cont2 =
      Reshape{/*continuation=*/unshard_cont, /*output_dimensions=*/{2, 2}};
  EXPECT_EQ(reshape_with_cont1, reshape_with_cont2);
  EXPECT_NE(reshape1, reshape_with_cont1);

  TensorTransformation rt_reshape1 = reshape1;
  TensorTransformation rt_reshape2 = reshape2;
  TensorTransformation rt_unshard1 = unshard1;
  TensorTransformation rt_reshape_with_cont1 = reshape_with_cont1;

  EXPECT_EQ(rt_reshape1, rt_reshape2);
  EXPECT_NE(rt_reshape1, rt_unshard1);
  EXPECT_NE(rt_reshape1, rt_reshape_with_cont1);
}

TEST(TensorTransformationTest, Hash) {
  auto reshape1 =
      Reshape{/*continuation=*/nullptr, /*output_dimensions=*/{2, 2}};
  auto unshard1 = Unshard{/*continuation=*/nullptr,
                          /*original_dimensions=*/{4},
                          /*sharding=*/HloSharding::Replicate()};
  auto unshard2 = Unshard{/*continuation=*/nullptr,
                          /*original_dimensions=*/{4},
                          /*sharding=*/HloSharding::Replicate()};
  Array<int64_t> tile_assignment({2, 1});
  tile_assignment(0, 0) = 0;
  tile_assignment(1, 0) = 1;
  auto unshard3 = Unshard{/*continuation=*/nullptr,
                          /*original_dimensions=*/{4},
                          /*sharding=*/HloSharding::Tile(tile_assignment)};
  auto broadcast1 = Broadcast{/*continuation=*/nullptr,
                              /*output_dimensions=*/{2, 2},
                              /*broadcast_dimensions=*/{0, 1}};

  auto unshard_cont = std::make_shared<const TensorTransformation>(unshard1);
  auto reshape_with_cont1 =
      Reshape{/*continuation=*/unshard_cont, /*output_dimensions=*/{2, 2}};
  auto reshape_with_cont2 =
      Reshape{/*continuation=*/unshard_cont, /*output_dimensions=*/{2, 2}};

  TensorTransformation rt1 = reshape1;
  TensorTransformation rt2 = unshard1;
  TensorTransformation rt3 = unshard2;
  TensorTransformation rt4 = unshard3;
  TensorTransformation rt5 = reshape_with_cont1;
  TensorTransformation rt6 = reshape_with_cont2;
  TensorTransformation rt7 =
      Reshape{/*continuation=*/nullptr, /*output_dimensions=*/{2, 2}};
  TensorTransformation rt8 = broadcast1;

  EXPECT_TRUE(absl::VerifyTypeImplementsAbslHashCorrectly(
      {rt1, rt2, rt3, rt4, rt5, rt6, rt7, rt8}));
}

TEST(TensorTransformationTest, AbslStringify) {
  auto reshape1 =
      Reshape{/*continuation=*/nullptr, /*output_dimensions=*/{2, 2}};
  TensorTransformation rt_reshape1 = reshape1;
  EXPECT_EQ(absl::StrCat(rt_reshape1),
            "Reshape{dimensions=[2, 2], continuation=nullptr}");

  auto unshard1 = Unshard{/*continuation=*/nullptr,
                          /*original_dimensions=*/{4},
                          /*sharding=*/HloSharding::Replicate()};
  TensorTransformation rt_unshard1 = unshard1;
  EXPECT_EQ(
      absl::StrCat(rt_unshard1),
      "Unshard{dimensions=[4], sharding={replicated}, continuation=nullptr}");

  auto broadcast1 = Broadcast{/*continuation=*/nullptr,
                              /*output_dimensions=*/{2, 2},
                              /*broadcast_dimensions=*/{0, 1}};
  TensorTransformation rt_broadcast1 = broadcast1;
  EXPECT_EQ(absl::StrCat(rt_broadcast1),
            "Broadcast{dimensions=[2, 2], broadcast_dimensions=[0, 1], "
            "continuation=nullptr}");

  auto unshard_cont = std::make_shared<const TensorTransformation>(unshard1);
  auto reshape_with_cont =
      Reshape{/*continuation=*/unshard_cont, /*output_dimensions=*/{2, 2}};
  TensorTransformation rt_reshape_with_cont = reshape_with_cont;
  EXPECT_EQ(absl::StrCat(rt_reshape_with_cont),
            "Reshape{dimensions=[2, 2], continuation=Unshard{dimensions=[4], "
            "sharding={replicated}, continuation=nullptr}}");
}

TEST(TensorTransformationTest, ToString) {
  EXPECT_EQ(tensor_transformation::ToString(nullptr), "nullptr");

  auto reshape = std::make_shared<const TensorTransformation>(
      Reshape{/*continuation=*/nullptr, /*output_dimensions=*/{2, 2}});
  EXPECT_EQ(tensor_transformation::ToString(reshape.get()),
            "Reshape{dimensions=[2, 2], continuation=nullptr}");

  auto unshard = std::make_shared<const TensorTransformation>(
      Unshard{/*continuation=*/nullptr,
              /*original_dimensions=*/{4},
              /*sharding=*/HloSharding::Replicate()});
  auto reshape_with_cont = std::make_shared<const TensorTransformation>(
      Reshape{/*continuation=*/unshard, /*output_dimensions=*/{2, 2}});
  EXPECT_EQ(tensor_transformation::ToString(reshape_with_cont.get()),
            "Reshape{dimensions=[2, 2], continuation=Unshard{dimensions=[4], "
            "sharding={replicated}, continuation=nullptr}}");
}

TEST(TensorTransformationTest, AppendContinuation) {
  auto reshape = std::make_shared<const TensorTransformation>(
      Reshape{/*continuation=*/nullptr, /*output_dimensions=*/{2, 2}});
  auto unshard = std::make_shared<const TensorTransformation>(
      Unshard{/*continuation=*/nullptr,
              /*original_dimensions=*/{4},
              /*sharding=*/HloSharding::Replicate()});

  // Append to nullptr.
  std::shared_ptr<const TensorTransformation> result1 =
      tensor_transformation::AppendContinuation(nullptr, reshape);
  EXPECT_EQ(*result1, *reshape);

  // Append nullptr.
  std::shared_ptr<const TensorTransformation> result2 =
      tensor_transformation::AppendContinuation(reshape, nullptr);
  EXPECT_EQ(*result2, *reshape);

  // Append to a single transformation.
  std::shared_ptr<const TensorTransformation> result3 =
      tensor_transformation::AppendContinuation(reshape, unshard);
  auto expected1 = std::make_shared<const TensorTransformation>(
      Reshape{/*continuation=*/unshard, /*output_dimensions=*/{2, 2}});
  EXPECT_EQ(*result3, *expected1);

  // Append to a chain of transformations.
  auto broadcast = std::make_shared<const TensorTransformation>(
      Broadcast{/*continuation=*/nullptr,
                /*output_dimensions=*/{2, 4},
                /*broadcast_dimensions=*/{0, 1}});
  std::shared_ptr<const TensorTransformation> result4 =
      tensor_transformation::AppendContinuation(result3, broadcast);

  auto unshard_with_broadcast = std::make_shared<const TensorTransformation>(
      Unshard{/*continuation=*/broadcast,
              /*original_dimensions=*/{4},
              /*sharding=*/HloSharding::Replicate()});
  auto expected2 = std::make_shared<const TensorTransformation>(Reshape{
      /*continuation=*/unshard_with_broadcast, /*output_dimensions=*/{2, 2}});
  EXPECT_EQ(*result4, *expected2);
}

// Example from AlignTensorSummaries documentation:
// baseline: split_spec = [{0: 4}, {2: 6}]
// target:   split_spec = [{0: 6}, {2: 4}]
// Aligned:  split_spec = [{0: 2}, {2: 2}]
TEST(AlignTensorSummariesTest, DocExample1) {
  OriginalTensorSummary baseline;
  baseline.dimensions = {10, 1, 10};
  baseline.summaries.emplace_back();
  baseline.summaries[0].split_spec = {
      {/*dim_index=*/0, /*block_count=*/4},
      {/*dim_index=*/2, /*block_count=*/6},
  };
  for (int i = 0; i < 4; ++i) {
    for (int j = 0; j < 6; ++j) {
      baseline.summaries[0].block_summaries.push_back(
          CreateBlockSummary({(int64_t)i, (int64_t)j}, 1, i, j));
    }
  }

  OriginalTensorSummary target;
  target.dimensions = {10, 1, 10};
  target.summaries.emplace_back();
  target.summaries[0].split_spec = {
      {/*dim_index=*/0, /*block_count=*/6},
      {/*dim_index=*/2, /*block_count=*/4},
  };
  for (int i = 0; i < 6; ++i) {
    for (int j = 0; j < 4; ++j) {
      target.summaries[0].block_summaries.push_back(
          CreateBlockSummary({(int64_t)i, (int64_t)j}, 1, i, j));
    }
  }

  OriginalTensorSummary baseline_aligned, target_aligned;
  ASSERT_OK_AND_ASSIGN(std::tie(baseline_aligned, target_aligned),
                       AlignTensorSummaries(baseline, target));

  FloatSummary expected_baseline;
  expected_baseline.split_spec = {{/*dim_index=*/0, /*block_count=*/2},
                                  {/*dim_index=*/2, /*block_count=*/2}};
  // Baseline: 4 blocks on dim 0 -> 2 blocks. 0,1->0, 2,3->1
  // 6 blocks on dim 2 -> 2 blocks. 0,1,2->0, 3,4,5->1
  // For new block {0,0}:
  // i in 0,1, j in 0,1,2.
  // min_val for this group: min(i)=0. max_val for this group: max(j)=2.
  // count = 2*3=6
  expected_baseline.block_summaries = {
      CreateBlockSummary({0, 0}, 6, 0, 2),
      CreateBlockSummary({0, 1}, 6, 0, 5),
      CreateBlockSummary({1, 0}, 6, 2, 2),
      CreateBlockSummary({1, 1}, 6, 2, 5),
  };
  ASSERT_EQ(baseline_aligned.summaries.size(), 1);
  EXPECT_THAT(baseline_aligned.summaries[0], FloatSummaryEq(expected_baseline));
  EXPECT_EQ(baseline_aligned.dimensions, baseline.dimensions);

  FloatSummary expected_target;
  expected_target.split_spec = {{/*dim_index=*/0, /*block_count=*/2},
                                {/*dim_index=*/2, /*block_count=*/2}};
  // Target: 6 blocks on dim 0 -> 2 blocks. 0,1,2->0, 3,4,5->1
  // 4 blocks on dim 2 -> 2 blocks. 0,1->0, 2,3->1
  // For new block {0,0}:
  // i in 0,1,2, j in 0,1
  // blocks are merged using CombineBlockSummaries which sums counts, takes min
  // of mins and max of maxes. block {0,0}: old blocks with i=0..2, j=0..1
  // min_val for this group: min(i)=0. max_val for this group: max(j)=1.
  // count = 3*2=6
  expected_target.block_summaries = {
      CreateBlockSummary({0, 0}, 6, 0, 1),
      CreateBlockSummary({0, 1}, 6, 0, 3),
      CreateBlockSummary({1, 0}, 6, 3, 1),
      CreateBlockSummary({1, 1}, 6, 3, 3),
  };
  ASSERT_EQ(target_aligned.summaries.size(), 1);
  EXPECT_THAT(target_aligned.summaries[0], FloatSummaryEq(expected_target));
  EXPECT_EQ(target_aligned.dimensions, target.dimensions);
}

// Example 2 from AlignTensorSummaries documentation:
// baseline: split_spec = [{1: 8}]
// target:   split_spec = [{1: 8}, {3: 5}]
// Aligned:  split_spec = [{1: 8}]
TEST(AlignTensorSummariesTest, DocExample2) {
  OriginalTensorSummary baseline;
  baseline.dimensions = {1, 10, 1, 10};
  baseline.summaries.emplace_back();
  baseline.summaries[0].split_spec = {{/*dim_index=*/1, /*block_count=*/8}};
  for (int i = 0; i < 8; ++i) {
    baseline.summaries[0].block_summaries.push_back(
        CreateBlockSummary({(int64_t)i}, 1, i, i));
  }

  OriginalTensorSummary target;
  target.dimensions = {1, 10, 1, 10};
  target.summaries.emplace_back();
  target.summaries[0].split_spec = {{/*dim_index=*/1, /*block_count=*/8},
                                    {/*dim_index=*/3, /*block_count=*/5}};
  for (int i = 0; i < 8; ++i) {
    for (int j = 0; j < 5; ++j) {
      target.summaries[0].block_summaries.push_back(
          CreateBlockSummary({(int64_t)i, (int64_t)j}, 1, i, j));
    }
  }

  OriginalTensorSummary baseline_aligned, target_aligned;
  ASSERT_OK_AND_ASSIGN(std::tie(baseline_aligned, target_aligned),
                       AlignTensorSummaries(baseline, target));

  FloatSummary expected_baseline;
  expected_baseline.split_spec = {{/*dim_index=*/1, /*block_count=*/8}};
  expected_baseline.block_summaries = baseline.summaries[0].block_summaries;
  ASSERT_EQ(baseline_aligned.summaries.size(), 1);
  EXPECT_THAT(baseline_aligned.summaries[0], FloatSummaryEq(expected_baseline));
  EXPECT_EQ(baseline_aligned.dimensions, baseline.dimensions);

  FloatSummary expected_target;
  expected_target.split_spec = {{/*dim_index=*/1, /*block_count=*/8}};
  // Dim 3 is merged.
  for (int i = 0; i < 8; ++i) {
    expected_target.block_summaries.push_back(
        CreateBlockSummary({(int64_t)i}, 5, i, 4));
  }
  ASSERT_EQ(target_aligned.summaries.size(), 1);
  EXPECT_THAT(target_aligned.summaries[0], FloatSummaryEq(expected_target));
  EXPECT_EQ(target_aligned.dimensions, target.dimensions);
}

// Example 3 from AlignTensorSummaries documentation:
// baseline: split_spec = [{0: 10}, {1: 4}]
// target:   split_spec = [{0: 4}, {2: 5}]
// Aligned:  split_spec = [{0: 2}]
TEST(AlignTensorSummariesTest, DocExample3) {
  OriginalTensorSummary baseline;
  baseline.dimensions = {10, 10, 10};
  baseline.summaries.emplace_back();
  baseline.summaries[0].split_spec = {{/*dim_index=*/0, /*block_count=*/10},
                                      {/*dim_index=*/1, /*block_count=*/4}};
  for (int i = 0; i < 10; ++i) {
    for (int j = 0; j < 4; ++j) {
      baseline.summaries[0].block_summaries.push_back(
          CreateBlockSummary({(int64_t)i, (int64_t)j}, 1, i, j));
    }
  }

  OriginalTensorSummary target;
  target.dimensions = {10, 10, 10};
  target.summaries.emplace_back();
  target.summaries[0].split_spec = {{/*dim_index=*/0, /*block_count=*/4},
                                    {/*dim_index=*/2, /*block_count=*/5}};
  for (int i = 0; i < 4; ++i) {
    for (int j = 0; j < 5; ++j) {
      target.summaries[0].block_summaries.push_back(
          CreateBlockSummary({(int64_t)i, (int64_t)j}, 1, i, j));
    }
  }

  OriginalTensorSummary baseline_aligned, target_aligned;
  ASSERT_OK_AND_ASSIGN(std::tie(baseline_aligned, target_aligned),
                       AlignTensorSummaries(baseline, target));

  FloatSummary expected_baseline;
  expected_baseline.split_spec = {{/*dim_index=*/0, /*block_count=*/2}};
  // Dim 0: 10 blocks -> 2 blocks. 0..4->0, 5..9->1
  // Dim 1 is merged.
  // Block 0: i=0..4, j=0..3. min=0, max=3, count=5*4=20
  expected_baseline.block_summaries.push_back(
      CreateBlockSummary({0}, 20, 0, 3));
  // Block 1: i=5..9, j=0..3. min=5, max=3, count=5*4=20
  expected_baseline.block_summaries.push_back(
      CreateBlockSummary({1}, 20, 5, 3));
  ASSERT_EQ(baseline_aligned.summaries.size(), 1);
  EXPECT_THAT(baseline_aligned.summaries[0], FloatSummaryEq(expected_baseline));
  EXPECT_EQ(baseline_aligned.dimensions, baseline.dimensions);

  FloatSummary expected_target;
  expected_target.split_spec = {{/*dim_index=*/0, /*block_count=*/2}};
  // Dim 0: 4 blocks -> 2 blocks. 0,1->0, 2,3->1
  // Dim 2 is merged.
  // Block 0: i=0..1, j=0..4. min=0, max=4, count=2*5=10
  expected_target.block_summaries.push_back(CreateBlockSummary({0}, 10, 0, 4));
  // Block 1: i=2..3, j=0..4. min=2, max=4, count=2*5=10
  expected_target.block_summaries.push_back(CreateBlockSummary({1}, 10, 2, 4));
  ASSERT_EQ(target_aligned.summaries.size(), 1);
  EXPECT_THAT(target_aligned.summaries[0], FloatSummaryEq(expected_target));
  EXPECT_EQ(target_aligned.dimensions, target.dimensions);
}

TEST(AlignTensorSummariesTest, NoSplits) {
  OriginalTensorSummary baseline;
  baseline.dimensions = {10};
  baseline.summaries.emplace_back();
  baseline.summaries[0].split_spec = {};
  baseline.summaries[0].block_summaries.push_back(
      CreateBlockSummary({}, 10, 0, 0));

  OriginalTensorSummary target;
  target.dimensions = {10};
  target.summaries.emplace_back();
  target.summaries[0].split_spec = {};
  target.summaries[0].block_summaries.push_back(
      CreateBlockSummary({}, 10, 1, 1));

  OriginalTensorSummary baseline_aligned, target_aligned;
  ASSERT_OK_AND_ASSIGN(std::tie(baseline_aligned, target_aligned),
                       AlignTensorSummaries(baseline, target));

  ASSERT_EQ(baseline_aligned.summaries.size(), 1);
  EXPECT_THAT(baseline_aligned.summaries[0],
              FloatSummaryEq(baseline.summaries[0]));
  EXPECT_EQ(baseline_aligned.dimensions, baseline.dimensions);
  ASSERT_EQ(target_aligned.summaries.size(), 1);
  EXPECT_THAT(target_aligned.summaries[0], FloatSummaryEq(target.summaries[0]));
  EXPECT_EQ(target_aligned.dimensions, target.dimensions);
}

TEST(AlignTensorSummariesTest, SplitInOneSummaryOnly) {
  OriginalTensorSummary baseline;
  baseline.dimensions = {10};
  baseline.summaries.emplace_back();
  baseline.summaries[0].split_spec = {{/*dim_index=*/0, /*block_count=*/2}};
  baseline.summaries[0].block_summaries = {CreateBlockSummary({0}, 5, 0, 1),
                                           CreateBlockSummary({1}, 5, 1, 2)};

  OriginalTensorSummary target;
  target.dimensions = {10};
  target.summaries.emplace_back();
  target.summaries[0].split_spec = {};
  target.summaries[0].block_summaries.push_back(
      CreateBlockSummary({}, 10, 0, 2));

  OriginalTensorSummary baseline_aligned, target_aligned;
  ASSERT_OK_AND_ASSIGN(std::tie(baseline_aligned, target_aligned),
                       AlignTensorSummaries(baseline, target));

  FloatSummary expected_baseline;
  expected_baseline.split_spec = {};
  expected_baseline.block_summaries = {CreateBlockSummary({}, 10, 0, 2)};

  ASSERT_EQ(baseline_aligned.summaries.size(), 1);
  EXPECT_THAT(baseline_aligned.summaries[0], FloatSummaryEq(expected_baseline));
  EXPECT_EQ(baseline_aligned.dimensions, baseline.dimensions);
  ASSERT_EQ(target_aligned.summaries.size(), 1);
  EXPECT_THAT(target_aligned.summaries[0], FloatSummaryEq(target.summaries[0]));
  EXPECT_EQ(target_aligned.dimensions, target.dimensions);
}

TEST(AlignTensorSummariesTest, GcdIsOne) {
  OriginalTensorSummary baseline;
  baseline.dimensions = {6};
  baseline.summaries.emplace_back();
  baseline.summaries[0].split_spec = {{/*dim_index=*/0, /*block_count=*/2}};
  baseline.summaries[0].block_summaries = {CreateBlockSummary({0}, 3, 0, 1),
                                           CreateBlockSummary({1}, 3, 1, 2)};

  OriginalTensorSummary target;
  target.dimensions = {6};
  target.summaries.emplace_back();
  target.summaries[0].split_spec = {{/*dim_index=*/0, /*block_count=*/3}};
  target.summaries[0].block_summaries = {CreateBlockSummary({0}, 2, 0, 1),
                                         CreateBlockSummary({1}, 2, 1, 2),
                                         CreateBlockSummary({2}, 2, 2, 3)};

  OriginalTensorSummary baseline_aligned, target_aligned;
  ASSERT_OK_AND_ASSIGN(std::tie(baseline_aligned, target_aligned),
                       AlignTensorSummaries(baseline, target));

  FloatSummary expected_baseline;
  expected_baseline.split_spec = {};
  expected_baseline.block_summaries = {CreateBlockSummary({}, 6, 0, 2)};
  ASSERT_EQ(baseline_aligned.summaries.size(), 1);
  EXPECT_THAT(baseline_aligned.summaries[0], FloatSummaryEq(expected_baseline));
  EXPECT_EQ(baseline_aligned.dimensions, baseline.dimensions);

  FloatSummary expected_target;
  expected_target.split_spec = {};
  expected_target.block_summaries = {CreateBlockSummary({}, 6, 0, 3)};
  ASSERT_EQ(target_aligned.summaries.size(), 1);
  EXPECT_THAT(target_aligned.summaries[0], FloatSummaryEq(expected_target));
  EXPECT_EQ(target_aligned.dimensions, target.dimensions);
}

TEST(ScopeInstructionTest, ToString) {
  ScopeInstruction scope1 = {/*instruction_name=*/"loop",
                             /*iteration_index=*/3};
  EXPECT_EQ(scope1.ToString(), "loop#3");
  ScopeInstruction scope2 = {/*instruction_name=*/"call",
                             /*iteration_index=*/0};
  EXPECT_EQ(scope2.ToString(), "call");
  ScopeInstruction scope3 = {/*instruction_name=*/"wild",
                             /*iteration_index=*/-1};
  EXPECT_EQ(scope3.ToString(), "wild#*");
  ScopeInstruction scope4 = {/*instruction_name=*/"while",
                             /*iteration_index=*/-2};
  EXPECT_EQ(scope4.ToString(), "while#$");
}

TEST(ScopeInstructionTest, FromString) {
  EXPECT_EQ(
      ScopeInstruction::FromString("loop#3"),
      (ScopeInstruction{/*instruction_name=*/"loop", /*iteration_index=*/3}));
  EXPECT_EQ(
      ScopeInstruction::FromString("call"),
      (ScopeInstruction{/*instruction_name=*/"call", /*iteration_index=*/0}));
  EXPECT_EQ(
      ScopeInstruction::FromString("wild#*"),
      (ScopeInstruction{/*instruction_name=*/"wild", /*iteration_index=*/-1}));
  EXPECT_EQ(
      ScopeInstruction::FromString("while#$"),
      (ScopeInstruction{/*instruction_name=*/"while", /*iteration_index=*/-2}));
}

TEST(TensorKeyTest, ToString) {
  TensorKey key1 = {/*instruction_name=*/"hlo", /*shape_index=*/{0, 1}};
  EXPECT_EQ(key1.ToString(), "hlo@{0,1}");
  TensorKey key2 = {/*instruction_name=*/"hlo", /*shape_index=*/{}};
  EXPECT_EQ(key2.ToString(), "hlo");
}

TEST(ScopedTensorKeyTest, ToString) {
  ScopedTensorKey key1 = {
      /*scope_instructions=*/{{/*instruction_name=*/"loop",
                               /*iteration_index=*/3}},
      /*tensor_key=*/{/*instruction_name=*/"hlo", /*shape_index=*/{0, 1}}};
  EXPECT_EQ(key1.ToString(), "loop#3/hlo@{0,1}");

  ScopedTensorKey key2 = {
      /*scope_instructions=*/{{/*instruction_name=*/"call",
                               /*iteration_index=*/0}},
      /*tensor_key=*/{/*instruction_name=*/"hlo", /*shape_index=*/{}}};
  EXPECT_EQ(key2.ToString(), "call/hlo");

  ScopedTensorKey key3 = {
      /*scope_instructions=*/{},
      /*tensor_key=*/{/*instruction_name=*/"hlo", /*shape_index=*/{2}}};
  EXPECT_EQ(key3.ToString(), "hlo@{2}");

  ScopedTensorKey key4 = {
      /*scope_instructions=*/{{/*instruction_name=*/"outer",
                               /*iteration_index=*/1},
                              {/*instruction_name=*/"inner",
                               /*iteration_index=*/2}},
      /*tensor_key=*/{/*instruction_name=*/"hlo", /*shape_index=*/{}}};
  EXPECT_EQ(key4.ToString(), "outer#1/inner#2/hlo");
}

TEST(ScopedTensorKeyTest, FromString) {
  EXPECT_EQ(
      ScopedTensorKey::FromString("loop#3/hlo", {0, 1}),
      (ScopedTensorKey{/*scope_instructions=*/{{/*instruction_name=*/"loop",
                                                /*iteration_index=*/3}},
                       /*tensor_key=*/{/*instruction_name=*/"hlo",
                                       /*shape_index=*/{0, 1}}}));

  EXPECT_EQ(
      ScopedTensorKey::FromString("call/hlo"),
      (ScopedTensorKey{
          /*scope_instructions=*/{{/*instruction_name=*/"call",
                                   /*iteration_index=*/0}},
          /*tensor_key=*/{/*instruction_name=*/"hlo", /*shape_index=*/{}}}));

  EXPECT_EQ(ScopedTensorKey::FromString("hlo", {2}),
            (ScopedTensorKey{/*scope_instructions=*/{},
                             /*tensor_key=*/{/*instruction_name=*/"hlo",
                                             /*shape_index=*/{2}}}));

  EXPECT_EQ(
      ScopedTensorKey::FromString("outer#1/inner#2/hlo"),
      (ScopedTensorKey{
          /*scope_instructions=*/
          {{/*instruction_name=*/"outer", /*iteration_index=*/1},
           {/*instruction_name=*/"inner", /*iteration_index=*/2}},
          /*tensor_key=*/{/*instruction_name=*/"hlo", /*shape_index=*/{}}}));
}

TEST(OriginalTensorSummaryTest, ToDebugString) {
  OriginalTensorSummary summary;
  summary.dimensions = {2, 2};
  summary.summaries.emplace_back();
  summary.summaries[0].split_spec = {{/*dim_index=*/0, /*block_count=*/2}};
  summary.summaries[0].block_summaries = {
      CreateBlockSummary({0}, 2, 0.0, 1.0),
      CreateBlockSummary({1}, 2, 1.0, 2.0),
  };
  EXPECT_EQ(summary.ToDebugString(),
            "OriginalTensorSummary{\n"
            "  dimensions: [2, 2]\n"
            "  summaries:\n"
            "    split_spec:\n"
            "      {dim_index: 0, block_count: 2}\n"
            "    block_summaries:\n"
            "      {block_indices: [0], min: 0, max: 1, mean: 0, stddev: 0, "
            "count: 2, nan_count: 0, pos_inf_count: 0, neg_inf_count: 0, "
            "zero_count: 0}\n"
            "      {block_indices: [1], min: 1, max: 2, mean: 0, stddev: 0, "
            "count: 2, nan_count: 0, pos_inf_count: 0, neg_inf_count: 0, "
            "zero_count: 0}\n"
            "}\n");
}

TEST(GetAbsoluteScopedTensorKeyTest, Conversion) {
  LogHloOutputMetadata metadata;
  metadata.set_instruction_name("hlo");
  metadata.add_shape_index(0);
  metadata.add_shape_index(1);
  auto* scope1 = metadata.add_scopes();
  scope1->set_instruction_name("loop");
  scope1->set_it_count(3);
  auto* scope2 = metadata.add_scopes();
  scope2->set_instruction_name("call");
  scope2->set_it_count(0);

  AbsoluteScopedTensorKey expected_key = {
      /*scope_instructions=*/{
          {/*instruction_name=*/"loop", /*iteration_index=*/3},
          {/*instruction_name=*/"call",
           /*iteration_index=*/0}},
      /*tensor_key=*/{/*instruction_name=*/"hlo", /*shape_index=*/{0, 1}}};

  EXPECT_EQ(GetAbsoluteScopedTensorKey(metadata), expected_key);
}

TEST(GetAbsoluteScopedTensorKeyTest, EmptyScopesAndShapeIndex) {
  LogHloOutputMetadata metadata;
  metadata.set_instruction_name("hlo");

  AbsoluteScopedTensorKey expected_key = {
      /*scope_instructions=*/{},
      /*tensor_key=*/{/*instruction_name=*/"hlo", /*shape_index=*/{}}};

  EXPECT_EQ(GetAbsoluteScopedTensorKey(metadata), expected_key);
}

TEST(ScopeInstructionTest, ProtoConversion) {
  ScopeInstruction scope = {/*instruction_name=*/"loop", /*iteration_index=*/3};
  ScopeInstructionProto proto = scope.ToProto();
  EXPECT_EQ(proto.instruction_name(), "loop");
  EXPECT_EQ(proto.iteration_index(), 3);
  EXPECT_EQ(ScopeInstruction::FromProto(proto), scope);
}

TEST(TensorKeyTest, ProtoConversion) {
  TensorKey key = {/*instruction_name=*/"hlo", /*shape_index=*/{0, 1}};
  TensorKeyProto proto = key.ToProto();
  EXPECT_EQ(proto.instruction_name(), "hlo");
  EXPECT_THAT(proto.shape_index(), testing::ElementsAre(0, 1));
  EXPECT_EQ(TensorKey::FromProto(proto), key);
}

TEST(ScopedTensorKeyTest, ProtoConversion) {
  ScopedTensorKey key = {
      /*scope_instructions=*/{{/*instruction_name=*/"loop",
                               /*iteration_index=*/3}},
      /*tensor_key=*/{/*instruction_name=*/"hlo", /*shape_index=*/{0, 1}}};
  AbsoluteScopedTensorKeyProto proto = key.ToProto();
  EXPECT_EQ(proto.tensor_key().instruction_name(), "hlo");
  EXPECT_THAT(proto.tensor_key().shape_index(), testing::ElementsAre(0, 1));
  ASSERT_EQ(proto.scope_instructions_size(), 1);
  EXPECT_EQ(proto.scope_instructions(0).instruction_name(), "loop");
  EXPECT_EQ(proto.scope_instructions(0).iteration_index(), 3);
  EXPECT_EQ(ScopedTensorKey::FromProto(proto), key);
}

TEST(OriginalTensorSummaryTest, ProtoConversion) {
  OriginalTensorSummary summary;
  summary.dimensions = {2, 2};
  summary.summaries.emplace_back();
  summary.summaries[0].split_spec = {{/*dim_index=*/0, /*block_count=*/2}};
  summary.summaries[0].block_summaries = {
      CreateBlockSummary({0}, 2, 0.0, 1.0),
      CreateBlockSummary({1}, 2, 1.0, 2.0),
  };
  RecoveredTensorSummaryProto::OriginalTensorSummaryProto proto =
      summary.ToProto();
  OriginalTensorSummary summary_from_proto =
      OriginalTensorSummary::FromProto(proto);
  EXPECT_EQ(summary_from_proto.dimensions, summary.dimensions);
  ASSERT_EQ(summary_from_proto.summaries.size(), 1);
  EXPECT_THAT(summary_from_proto.summaries[0],
              FloatSummaryEq(summary.summaries[0]));
}

TEST(TensorTransformationTest, ProtoConversion) {
  auto unshard = std::make_shared<const TensorTransformation>(
      Unshard{/*continuation=*/nullptr,
              /*original_dimensions=*/{4},
              /*sharding=*/HloSharding::Replicate()});
  auto broadcast = std::make_shared<const TensorTransformation>(
      Broadcast{/*continuation=*/unshard,
                /*output_dimensions=*/{2, 2},
                /*broadcast_dimensions=*/{0, 1}});
  auto reshape = std::make_shared<const TensorTransformation>(
      Reshape{/*continuation=*/broadcast, /*output_dimensions=*/{4}});

  google::protobuf::RepeatedPtrField<TensorTransformationProto> proto_field;
  tensor_transformation::ToProto(reshape.get(), &proto_field);

  ASSERT_EQ(proto_field.size(), 3);
  EXPECT_TRUE(proto_field.Get(0).has_reshape());
  EXPECT_THAT(proto_field.Get(0).reshape().output_dimensions(),
              testing::ElementsAre(4));
  EXPECT_TRUE(proto_field.Get(1).has_broadcast());
  EXPECT_THAT(proto_field.Get(1).broadcast().output_dimensions(),
              testing::ElementsAre(2, 2));
  EXPECT_THAT(proto_field.Get(1).broadcast().broadcast_dimensions(),
              testing::ElementsAre(0, 1));
  EXPECT_TRUE(proto_field.Get(2).has_unshard());
  EXPECT_THAT(proto_field.Get(2).unshard().original_dimensions(),
              testing::ElementsAre(4));
  EXPECT_THAT(proto_field.Get(2).unshard().sharding(),
              EqualsProto(HloSharding::Replicate().ToProto()));

  ASSERT_OK_AND_ASSIGN(
      std::shared_ptr<const TensorTransformation> transformation_from_proto,
      tensor_transformation::FromProto(proto_field));

  EXPECT_EQ(*transformation_from_proto, *reshape);
}

TEST(RecoveredTensorSummaryTest, ProtoConversion) {
  AbsoluteScopedTensorKey key = {
      /*scope_instructions=*/{{/*instruction_name=*/"loop",
                               /*iteration_index=*/3}},
      /*tensor_key=*/{/*instruction_name=*/"hlo", /*shape_index=*/{0, 1}}};
  auto reshape = std::make_shared<const TensorTransformation>(
      Reshape{/*continuation=*/nullptr, /*output_dimensions=*/{4}});
  OriginalTensorSummary summary;
  summary.dimensions = {2, 2};
  summary.summaries.emplace_back();
  summary.summaries[0].split_spec = {{/*dim_index=*/0, /*block_count=*/2}};
  summary.summaries[0].block_summaries = {
      CreateBlockSummary({0}, 2, 0.0, 1.0),
      CreateBlockSummary({1}, 2, 1.0, 2.0),
  };

  RecoveredTensorSummaryProto proto =
      CreateRecoveredTensorSummaryProto(key, reshape, summary);

  EXPECT_THAT(proto.tensor_key(), EqualsProto(key.ToProto()));
  ASSERT_EQ(proto.pending_transformation_size(), 1);
  EXPECT_TRUE(proto.pending_transformation(0).has_reshape());
  EXPECT_THAT(proto.original_tensor_summary(), EqualsProto(summary.ToProto()));

  ASSERT_OK_AND_ASSIGN(RecoveredTensorSummary recovered_summary,
                       RecoveredTensorSummaryFromProto(proto));
  EXPECT_EQ(recovered_summary.original_tensor_key, key);
  EXPECT_EQ(*recovered_summary.pending_transformation, *reshape);
  EXPECT_EQ(recovered_summary.original_tensor_summary.dimensions,
            summary.dimensions);
  ASSERT_EQ(recovered_summary.original_tensor_summary.summaries.size(), 1);
  EXPECT_THAT(recovered_summary.original_tensor_summary.summaries[0],
              FloatSummaryEq(summary.summaries[0]));
}

TEST(ScopedTensorKeyTest, ComparisonAndSorting) {
  auto k1 = ScopedTensorKey::Create(TensorKey::Create("abc", {0}));
  auto k2 = ScopedTensorKey::Create(TensorKey::Create("def", {0}));
  auto k3 = ScopedTensorKey::Create(TensorKey::Create("def", {1}));

  EXPECT_TRUE(k1 < k2);
  EXPECT_TRUE(k2 < k3);
  EXPECT_TRUE(k1 < k3);

  std::vector<ScopedTensorKey> keys = {k3, k1, k2};
  std::sort(keys.begin(), keys.end());
  EXPECT_THAT(keys, ::testing::ElementsAre(k1, k2, k3));
}

}  // namespace
}  // namespace xla::numerics::comparison
