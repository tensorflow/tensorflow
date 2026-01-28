/* Copyright 2025 The OpenXLA Authors.

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

#include "xla/service/shaped_slice.h"

#include <cstddef>
#include <cstdint>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/status/status_matchers.h"
#include "absl/strings/str_cat.h"
#include "xla/primitive_util.h"
#include "xla/service/buffer_assignment.h"
#include "xla/service/shaped_slice.pb.h"
#include "xla/shape_util.h"
#include "xla/tsl/util/proto/parse_text_proto.h"
#include "xla/tsl/util/proto/proto_matchers.h"
#include "xla/xla_data.pb.h"

namespace xla {
namespace {
using absl_testing::IsOkAndHolds;
using ::testing::HasSubstr;
using tsl::proto_testing::EqualsProto;
using tsl::proto_testing::ParseTextProtoOrDie;

TEST(ShapedSliceTest, Stringify) {
  constexpr int64_t kNumElements = 1024;
  const size_t kSizeInBytes = kNumElements * primitive_util::ByteWidth(F32);
  BufferAllocation alloc(/*index=*/0,
                         /*size=*/kSizeInBytes,
                         /*color=*/0);
  ShapedSlice shaped_slice;
  shaped_slice.slice =
      BufferAllocation::Slice(&alloc, /*offset=*/primitive_util::ByteWidth(F32),
                              /*size=*/kSizeInBytes);
  shaped_slice.shape = ShapeUtil::MakeShape(F32, {kNumElements - 1});
  EXPECT_THAT(absl::StrCat(shaped_slice), HasSubstr("ShapedSlice"));
  EXPECT_THAT(absl::StrCat(shaped_slice),
              HasSubstr(absl::StrCat(shaped_slice.slice)));
  EXPECT_THAT(absl::StrCat(shaped_slice),
              HasSubstr(shaped_slice.shape.ToString(/*print_layout=*/true)));
}

TEST(ShapedSliceTest, ToProto) {
  constexpr int64_t kNumElements = 1024;
  const size_t kSizeInBytes = kNumElements * primitive_util::ByteWidth(F32);
  BufferAllocation alloc(/*index=*/0,
                         /*size=*/kSizeInBytes,
                         /*color=*/0);
  ShapedSlice shaped_slice;
  shaped_slice.slice = BufferAllocation::Slice(
      &alloc, /*offset=*/primitive_util::ByteWidth(F32),
      /*size=*/kSizeInBytes - primitive_util::ByteWidth(F32));
  shaped_slice.shape = ShapeUtil::MakeShape(F32, {kNumElements - 1});

  EXPECT_THAT(
      shaped_slice.ToProto(), IsOkAndHolds(EqualsProto(R"pb(
        slice { buffer_allocation_index: 0 offset: 4 size: 4092 }
        shape {
          element_type: F32
          dimensions: 1023
          layout { minor_to_major: 0 tail_padding_alignment_in_elements: 1 }
          is_dynamic_dimension: false
        }
      )pb")));
}

TEST(ShapedSliceTest, FromProto) {
  ShapedSliceProto proto = ParseTextProtoOrDie<ShapedSliceProto>(R"pb(
    slice { buffer_allocation_index: 0 offset: 4 size: 4092 }
    shape {
      element_type: F32
      dimensions: 1023
      layout { minor_to_major: 0 tail_padding_alignment_in_elements: 1 }
      is_dynamic_dimension: false
    }
  )pb");

  constexpr int64_t kNumElementsInBuffer = 1024;
  const size_t kSizeInBytes =
      kNumElementsInBuffer * primitive_util::ByteWidth(F32);

  BufferAllocation alloc(/*index=*/0,
                         /*size=*/kSizeInBytes,
                         /*color=*/0);

  ShapedSlice expected_shaped_slice;
  expected_shaped_slice.slice = BufferAllocation::Slice(
      &alloc, /*offset=*/primitive_util::ByteWidth(F32),
      /*size=*/kSizeInBytes - primitive_util::ByteWidth(F32));

  // The slice starts with an offset of one element, therefore it can only hold
  // kNumElementsInBuffer - 1 elements.
  constexpr int64_t kNumElementsInSlice = kNumElementsInBuffer - 1;

  expected_shaped_slice.shape =
      ShapeUtil::MakeShape(F32, {kNumElementsInSlice});

  std::vector<BufferAllocation> buffer_allocations = {alloc};
  EXPECT_THAT(ShapedSlice::FromProto(proto, buffer_allocations),
              IsOkAndHolds(expected_shaped_slice));
}

TEST(NullableShapedSliceTest, StringifyNonEmptySlice) {
  constexpr int64_t kNumElementsInBuffer = 1024;
  const size_t kSizeInBytes =
      kNumElementsInBuffer * primitive_util::ByteWidth(F32);
  BufferAllocation alloc(/*index=*/0,
                         /*size=*/kSizeInBytes,
                         /*color=*/0);
  ShapedSlice shaped_slice;
  shaped_slice.slice =
      BufferAllocation::Slice(&alloc, /*offset=*/primitive_util::ByteWidth(F32),
                              /*size=*/kSizeInBytes);

  // The slice starts with an offset of one element, therefore it can only hold
  // kNumElementsInBuffer - 1 elements.
  constexpr int64_t kNumElementsInSlice = kNumElementsInBuffer - 1;
  shaped_slice.shape = ShapeUtil::MakeShape(F32, {kNumElementsInSlice});
  NullableShapedSlice non_empty_slice(shaped_slice);
  EXPECT_THAT(absl::StrCat(non_empty_slice),
              HasSubstr(absl::StrCat(shaped_slice)));
}

TEST(NullableShapedSliceTest, StringifyEmptySlice) {
  NullableShapedSlice empty_slice;
  EXPECT_THAT(absl::StrCat(empty_slice), HasSubstr("null"));
}

TEST(NullableShapedSliceTest, ToProtoNonEmptySlice) {
  constexpr int64_t kNumElementsInBuffer = 1024;
  const size_t kSizeInBytes =
      kNumElementsInBuffer * primitive_util::ByteWidth(F32);
  BufferAllocation alloc(/*index=*/0,
                         /*size=*/kSizeInBytes,
                         /*color=*/0);
  ShapedSlice shaped_slice;
  shaped_slice.slice = BufferAllocation::Slice(
      &alloc, /*offset=*/primitive_util::ByteWidth(F32),
      /*size=*/kSizeInBytes - primitive_util::ByteWidth(F32));

  // The slice starts with an offset of one element, therefore it can only hold
  // kNumElementsInBuffer - 1 elements.
  constexpr int64_t kNumElementsInSlice = kNumElementsInBuffer - 1;
  shaped_slice.shape = ShapeUtil::MakeShape(F32, {kNumElementsInSlice});
  NullableShapedSlice non_empty_slice(shaped_slice);
  EXPECT_THAT(
      non_empty_slice.ToProto(), IsOkAndHolds(EqualsProto(R"pb(
        shaped_slice {
          slice { buffer_allocation_index: 0 offset: 4 size: 4092 }
          shape {
            element_type: F32
            dimensions: 1023
            layout { minor_to_major: 0 tail_padding_alignment_in_elements: 1 }
            is_dynamic_dimension: false
          }
        }
      )pb")));
}

TEST(NullableShapedSliceTest, ToProtoEmptySlice) {
  NullableShapedSlice empty_slice;
  EXPECT_THAT(empty_slice.ToProto(), IsOkAndHolds(EqualsProto(R"pb()pb")));
}

TEST(NullableShapedSliceTest, FromProtoNonEmptySlice) {
  NullableShapedSliceProto proto =
      ParseTextProtoOrDie<NullableShapedSliceProto>(R"pb(
        shaped_slice {
          slice { buffer_allocation_index: 0 offset: 4 size: 4092 }
          shape {
            element_type: F32
            dimensions: 1023
            layout { minor_to_major: 0 tail_padding_alignment_in_elements: 1 }
            is_dynamic_dimension: false
          }
        }
      )pb");

  constexpr int64_t kNumElementsInBuffer = 1024;
  const size_t kSizeInBytes =
      kNumElementsInBuffer * primitive_util::ByteWidth(F32);
  BufferAllocation alloc(/*index=*/0,
                         /*size=*/kSizeInBytes,
                         /*color=*/0);
  std::vector<BufferAllocation> buffer_allocations = {alloc};
  ShapedSlice expected_shaped_slice;
  expected_shaped_slice.slice = BufferAllocation::Slice(
      &buffer_allocations[0], /*offset=*/primitive_util::ByteWidth(F32),
      /*size=*/kSizeInBytes - primitive_util::ByteWidth(F32));

  // The slice starts with an offset of one element, therefore it can only hold
  // kNumElementsInBuffer - 1 elements.
  constexpr size_t kNumElementsInSlice = kNumElementsInBuffer - 1;
  expected_shaped_slice.shape =
      ShapeUtil::MakeShape(F32, {kNumElementsInSlice});

  EXPECT_THAT(NullableShapedSlice::FromProto(
                  proto, /*buffer_allocations=*/buffer_allocations),
              IsOkAndHolds(NullableShapedSlice(expected_shaped_slice)));
}

TEST(NullableShapedSliceTest, FromProtoEmptySlice) {
  NullableShapedSliceProto proto;
  EXPECT_THAT(NullableShapedSlice::FromProto(proto, /*buffer_allocations=*/{}),
              IsOkAndHolds(NullableShapedSlice()));
}

}  // namespace
}  // namespace xla
