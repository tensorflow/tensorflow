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

#include "xla/hlo/tools/comparison/comparison_result_utils.h"

#include "xla/hlo/tools/comparison/comparison_result.pb.h"
#include "xla/tsl/platform/test.h"
#include "xla/tsl/util/proto/parse_text_proto.h"

namespace xla::numerics::comparison {
namespace {

using ::testing::DoubleEq;
using ::testing::DoubleNear;
using ::tsl::proto_testing::ParseTextProtoOrDie;

TEST(ComputeDiffScoreTest, EmptyProto) {
  ComparisonResultProto result;
  EXPECT_THAT(ComputeDiffScore(result), DoubleEq(-1.0));
}

TEST(ComputeDiffScoreTest, MissingBaselineSummary) {
  ComparisonResultProto result =
      ParseTextProtoOrDie<ComparisonResultProto>(R"pb(
        target_tensor_summaries {}
      )pb");
  EXPECT_THAT(ComputeDiffScore(result), DoubleEq(-1.0));
}

TEST(ComputeDiffScoreTest, MissingTargetSummary) {
  ComparisonResultProto result =
      ParseTextProtoOrDie<ComparisonResultProto>(R"pb(
        baseline_tensor_summaries {}
      )pb");
  EXPECT_THAT(ComputeDiffScore(result), DoubleEq(-1.0));
}

TEST(ComputeDiffScoreTest, BlockSummariesSizeMismatch) {
  ComparisonResultProto result =
      ParseTextProtoOrDie<ComparisonResultProto>(R"pb(
        baseline_tensor_summaries { block_summaries {} }
        target_tensor_summaries {}
      )pb");
  EXPECT_THAT(ComputeDiffScore(result), DoubleEq(-1.0));
}

TEST(ComputeDiffScoreTest, NoBlocks) {
  ComparisonResultProto result =
      ParseTextProtoOrDie<ComparisonResultProto>(R"pb(
        baseline_tensor_summaries {}
        target_tensor_summaries {}
      )pb");
  EXPECT_THAT(ComputeDiffScore(result), DoubleEq(0.0));
}

TEST(ComputeDiffScoreTest, BlockCountMismatch) {
  ComparisonResultProto result =
      ParseTextProtoOrDie<ComparisonResultProto>(R"pb(
        baseline_tensor_summaries { block_summaries { count: 1 } }
        target_tensor_summaries { block_summaries { count: 2 } }
      )pb");
  EXPECT_THAT(ComputeDiffScore(result), DoubleEq(-1.0));
}

TEST(ComputeDiffScoreTest, IdenticalSummaries) {
  ComparisonResultProto result =
      ParseTextProtoOrDie<ComparisonResultProto>(R"pb(
        baseline_tensor_summaries {
          block_summaries { min: 1 max: 2 mean: 1.5 stddev: 0.5 count: 4 }
        }
        target_tensor_summaries {
          block_summaries { min: 1 max: 2 mean: 1.5 stddev: 0.5 count: 4 }
        }
      )pb");
  EXPECT_THAT(ComputeDiffScore(result), DoubleEq(0.0));
}

TEST(ComputeDiffScoreTest, SingleBlockDifference) {
  ComparisonResultProto result =
      ParseTextProtoOrDie<ComparisonResultProto>(R"pb(
        baseline_tensor_summaries {
          block_summaries { min: 1 max: 1 mean: 1 stddev: 0 count: 1 }
        }
        target_tensor_summaries {
          block_summaries { min: 3 max: 3 mean: 3 stddev: 0 count: 1 }
        }
      )pb");
  EXPECT_THAT(ComputeDiffScore(result), DoubleNear(35.27, 1e-2));
}

TEST(ComputeDiffScoreTest, MultipleBlocksOneDifferent) {
  ComparisonResultProto result =
      ParseTextProtoOrDie<ComparisonResultProto>(R"pb(
        baseline_tensor_summaries {
          block_summaries { min: 1 max: 1 mean: 1 stddev: 0 count: 1 }
          block_summaries { min: 1 max: 1 mean: 1 stddev: 0 count: 1 }
        }
        target_tensor_summaries {
          block_summaries { min: 1 max: 1 mean: 1 stddev: 0 count: 1 }
          block_summaries { min: 3 max: 3 mean: 3 stddev: 0 count: 1 }
        }
      )pb");
  EXPECT_THAT(ComputeDiffScore(result), DoubleNear(29.66, 1e-2));
}

TEST(ComputeDiffScoreTest, ZeroValue) {
  ComparisonResultProto result = ParseTextProtoOrDie<
      ComparisonResultProto>(R"pb(
    baseline_tensor_summaries {
      block_summaries {
        min: -1
        max: 1
        mean: 0
        stddev: 1
        count: 3
        zero_count: 1
      }
    }
    target_tensor_summaries {
      block_summaries { min: 1 max: 1 mean: 1 stddev: 1 count: 3 zero_count: 0 }
    }
  )pb");
  EXPECT_THAT(ComputeDiffScore(result), DoubleNear(67.81, 1e-2));
}

TEST(ComputeDiffScoreTest, ZeroValueSame) {
  ComparisonResultProto result =
      ParseTextProtoOrDie<ComparisonResultProto>(R"pb(
        baseline_tensor_summaries {
          block_summaries {
            min: -1
            max: 1
            mean: 0
            stddev: 1
            count: 3
            zero_count: 1
          }
        }
        target_tensor_summaries {
          block_summaries {
            min: -1
            max: 1
            mean: 0
            stddev: 1
            count: 3
            zero_count: 1
          }
        }
      )pb");
  EXPECT_THAT(ComputeDiffScore(result), DoubleEq(0.0));
}

TEST(ComputeDiffScoreTest, NanValue) {
  ComparisonResultProto result =
      ParseTextProtoOrDie<ComparisonResultProto>(R"pb(
        baseline_tensor_summaries {
          block_summaries {
            min: 1
            max: 1
            mean: nan
            stddev: nan
            count: 1
            nan_count: 1
          }
        }
        target_tensor_summaries {
          block_summaries { min: 1 max: 1 mean: 1 stddev: 0 count: 1 }
        }
      )pb");
  EXPECT_THAT(ComputeDiffScore(result), DoubleNear(72.74, 1e-2));
}

TEST(ComputeDiffScoreTest, NanValueSame) {
  ComparisonResultProto result =
      ParseTextProtoOrDie<ComparisonResultProto>(R"pb(
        baseline_tensor_summaries {
          block_summaries {
            min: 1
            max: 1
            mean: nan
            stddev: nan
            count: 1
            nan_count: 1
          }
        }
        target_tensor_summaries {
          block_summaries {
            min: 1
            max: 1
            mean: nan
            stddev: nan
            count: 1
            nan_count: 1
          }
        }
      )pb");
  EXPECT_THAT(ComputeDiffScore(result), DoubleEq(0.0));
}

TEST(ComputeDiffScoreTest, InfValue) {
  ComparisonResultProto result =
      ParseTextProtoOrDie<ComparisonResultProto>(R"pb(
        baseline_tensor_summaries {
          block_summaries {
            min: 1
            max: 1
            mean: inf
            stddev: 0
            count: 1
            pos_inf_count: 1
          }
        }
        target_tensor_summaries {
          block_summaries { min: 1 max: 1 mean: 1 stddev: 0 count: 1 }
        }
      )pb");
  EXPECT_THAT(ComputeDiffScore(result), DoubleNear(62.75, 1e-2));
}

TEST(ComputeDiffScoreTest, InfValueSame) {
  ComparisonResultProto result =
      ParseTextProtoOrDie<ComparisonResultProto>(R"pb(
        baseline_tensor_summaries {
          block_summaries {
            min: 1
            max: 1
            mean: inf
            stddev: 0
            count: 1
            pos_inf_count: 1
          }
        }
        target_tensor_summaries {
          block_summaries {
            min: 1
            max: 1
            mean: inf
            stddev: 0
            count: 1
            pos_inf_count: 1
          }
        }
      )pb");
  EXPECT_THAT(ComputeDiffScore(result), DoubleEq(0.0));
}

TEST(ComputeDiffScoreTest, InfValueDifferentSign) {
  ComparisonResultProto result =
      ParseTextProtoOrDie<ComparisonResultProto>(R"pb(
        baseline_tensor_summaries {
          block_summaries {
            min: 1
            max: 1
            mean: inf
            stddev: 0
            count: 1
            pos_inf_count: 1
          }
        }
        target_tensor_summaries {
          block_summaries {
            min: 1
            max: 1
            mean: -inf
            stddev: 0
            count: 1
            neg_inf_count: 1
          }
        }
      )pb");
  EXPECT_THAT(ComputeDiffScore(result), DoubleNear(65.59, 1e-2));
}

TEST(ComputeDiffScoreTest, InfValueWithFloat) {
  ComparisonResultProto result =
      ParseTextProtoOrDie<ComparisonResultProto>(R"pb(
        baseline_tensor_summaries {
          block_summaries {
            min: 1
            max: 1
            mean: inf
            stddev: 0
            count: 1
            pos_inf_count: 1
          }
        }
        target_tensor_summaries {
          block_summaries { min: 1 max: 1 mean: 1.0 stddev: 0 count: 1 }
        }
      )pb");
  EXPECT_THAT(ComputeDiffScore(result), DoubleNear(62.75, 1e-2));
}

TEST(ComputeDiffScoreTest, SmallFloatingPointDifferenceNearZero) {
  ComparisonResultProto result =
      ParseTextProtoOrDie<ComparisonResultProto>(R"pb(
        baseline_tensor_summaries {
          block_summaries { min: 0 max: 0 mean: 0 stddev: 0 count: 1 }
        }
        target_tensor_summaries {
          block_summaries { min: 0 max: 0 mean: 1e-7 stddev: 0 count: 1 }
        }
      )pb");
  EXPECT_THAT(ComputeDiffScore(result), DoubleNear(0.000595, 1e-5));
}

TEST(ComputeDiffScoreTest, LargeFloatingPointDifferenceNearZero) {
  ComparisonResultProto result =
      ParseTextProtoOrDie<ComparisonResultProto>(R"pb(
        baseline_tensor_summaries {
          block_summaries { min: 0 max: 0 mean: 0 stddev: 0 count: 1 }
        }
        target_tensor_summaries {
          block_summaries { min: 0 max: 0 mean: 0.1 stddev: 0 count: 1 }
        }
      )pb");
  EXPECT_THAT(ComputeDiffScore(result), DoubleNear(54.05, 1e-2));
}

TEST(GetColorForScoreTest, ThresholdValues) {
  EXPECT_EQ(GetColorForScore(-1.0), "#d3d3d3");
  EXPECT_EQ(GetColorForScore(0.0), "#99ff99");
  EXPECT_EQ(GetColorForScore(1.0), "#c0f580");
  EXPECT_EQ(GetColorForScore(5.0), "#e0ee40");
  EXPECT_EQ(GetColorForScore(10.0), "#eeee00");
  EXPECT_EQ(GetColorForScore(30.0), "#ffc000");
  EXPECT_EQ(GetColorForScore(60.0), "#ff8000");
  EXPECT_EQ(GetColorForScore(100.0), "#ff1717");
}

TEST(GetColorForScoreTest, Interpolation) {
  // Between 0.0 (#99ff99) and 1.0 (#c0f580)
  // R: 0x99 (153) and 0xc0 (192) -> middle is 172.5 -> 0xac
  // G: 0xff (255) and 0xf5 (245) -> middle is 250 -> 0xfa
  // B: 0x99 (153) and 0x80 (128) -> middle is 140.5 -> 0x8c
  EXPECT_EQ(GetColorForScore(0.5), "#acfa8c");

  // Out of bounds
  EXPECT_EQ(GetColorForScore(-2.0), "#d3d3d3");
  EXPECT_EQ(GetColorForScore(200.0), "#ff1717");
}

}  // namespace
}  // namespace xla::numerics::comparison
