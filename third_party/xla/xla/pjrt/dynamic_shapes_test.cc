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

#include "xla/pjrt/dynamic_shapes.h"

#include <cstdint>
#include <cstring>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/status/status.h"
#include "absl/status/status_matchers.h"
#include "absl/types/span.h"
#include "xla/shape.h"
#include "xla/shape_util.h"
#include "xla/xla_data.pb.h"

namespace xla {
namespace {

using ::absl_testing::IsOkAndHolds;
using ::absl_testing::StatusIs;

// Serializes one int32 metadata slot per entry of `sizes`.
std::vector<uint8_t> MakeMetadata(absl::Span<const int32_t> sizes) {
  std::vector<uint8_t> bytes(sizes.size() * sizeof(int32_t));
  std::memcpy(bytes.data(), sizes.data(), bytes.size());
  return bytes;
}

MATCHER_P(ShapeEqualTo, expected, "") {
  return ShapeUtil::Equal(arg, expected);
}

TEST(ReadDynamicShapeMetadataTest, Rank1DynamicDimension) {
  Shape shape = ShapeUtil::MakeShape(F32, {64}, {true});
  EXPECT_THAT(ReadDynamicShapeMetadata(MakeMetadata({24}), shape,
                                       PjRtDynamicShapeKind::kPrefix),
              IsOkAndHolds(ShapeEqualTo(ShapeUtil::MakeShape(F32, {24}))));
}

TEST(ReadDynamicShapeMetadataTest, StaticDimensionIgnoresUnwrittenSlot) {
  // Only dimension 0 is dynamic. A producer may leave the metadata slot of the
  // static dimension unwritten (zero); the static extent must be used instead.
  Shape shape = ShapeUtil::MakeShape(F32, {64, 128}, {true, false});
  EXPECT_THAT(ReadDynamicShapeMetadata(MakeMetadata({24, 0}), shape,
                                       PjRtDynamicShapeKind::kPrefix),
              IsOkAndHolds(ShapeEqualTo(ShapeUtil::MakeShape(F32, {24, 128}))));
}

TEST(ReadDynamicShapeMetadataTest, StaticDimensionIgnoresWrittenSlot) {
  // Producers that write all slots store the static size in the static slot;
  // the result is the same either way.
  Shape shape = ShapeUtil::MakeShape(F32, {64, 128}, {true, false});
  EXPECT_THAT(ReadDynamicShapeMetadata(MakeMetadata({24, 128}), shape,
                                       PjRtDynamicShapeKind::kSuffix),
              IsOkAndHolds(ShapeEqualTo(ShapeUtil::MakeShape(F32, {24, 128}))));
}

TEST(ReadDynamicShapeMetadataTest, StaticDimensionIgnoresGarbageSlot) {
  // A stale value in a static slot that would exceed the static extent must not
  // affect the result.
  Shape shape = ShapeUtil::MakeShape(F32, {64, 128}, {true, false});
  EXPECT_THAT(ReadDynamicShapeMetadata(MakeMetadata({24, 4096}), shape,
                                       PjRtDynamicShapeKind::kPrefix),
              IsOkAndHolds(ShapeEqualTo(ShapeUtil::MakeShape(F32, {24, 128}))));
}

TEST(ReadDynamicShapeMetadataTest, MultipleDynamicDimensions) {
  Shape shape = ShapeUtil::MakeShape(F32, {64, 32, 128}, {true, true, false});
  EXPECT_THAT(
      ReadDynamicShapeMetadata(MakeMetadata({24, 7, 0}), shape,
                               PjRtDynamicShapeKind::kPrefix),
      IsOkAndHolds(ShapeEqualTo(ShapeUtil::MakeShape(F32, {24, 7, 128}))));
}

TEST(ReadDynamicShapeMetadataTest, DynamicSizeExceedingBoundIsInvalid) {
  Shape shape = ShapeUtil::MakeShape(F32, {64, 128}, {true, false});
  EXPECT_THAT(ReadDynamicShapeMetadata(MakeMetadata({65, 128}), shape,
                                       PjRtDynamicShapeKind::kPrefix),
              StatusIs(absl::StatusCode::kInvalidArgument));
}

TEST(ReadDynamicShapeMetadataTest, InsufficientMetadataIsInvalid) {
  Shape shape = ShapeUtil::MakeShape(F32, {64, 128}, {true, false});
  EXPECT_THAT(ReadDynamicShapeMetadata(MakeMetadata({24}), shape,
                                       PjRtDynamicShapeKind::kPrefix),
              StatusIs(absl::StatusCode::kInvalidArgument));
}

TEST(ReadDynamicShapeMetadataTest, NotSupportedKindIsInvalid) {
  Shape shape = ShapeUtil::MakeShape(F32, {64}, {true});
  EXPECT_THAT(ReadDynamicShapeMetadata(MakeMetadata({24}), shape,
                                       PjRtDynamicShapeKind::kNotSupported),
              StatusIs(absl::StatusCode::kInvalidArgument));
}

}  // namespace
}  // namespace xla
