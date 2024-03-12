// Copyright 2023 The OpenXLA Authors.
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

#include "xla/python/ifrt_proxy/common/types.h"

#include <cstdint>
#include <utility>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "xla/pjrt/pjrt_common.h"
#include "xla/python/ifrt/shape.h"
#include "xla/python/ifrt_proxy/common/types.pb.h"
#include "tsl/platform/status_matchers.h"
#include "tsl/platform/statusor.h"
#include "tsl/platform/test.h"

namespace xla {
namespace ifrt {
namespace proxy {
namespace {

using ::tsl::testing::IsOkAndHolds;

#if defined(PLATFORM_GOOGLE)
using ::testing::EquivToProto;
#endif

TEST(DTypeTest, ToFromProto) {
  for (int i = 0; i < proto::DType_descriptor()->value_count(); ++i) {
    const proto::DType dtype = static_cast<proto::DType>(
        proto::DType_descriptor()->value(i)->number());
    EXPECT_EQ(ToDTypeProto(FromDTypeProto(dtype)), dtype);
  }
}

// TODO(b/315809436): Test needs rewrite because protobuf matchers are not OSS
#if defined(PLATFORM_GOOGLE)
class ShapeTest
    : public testing::TestWithParam<std::pair<Shape, proto::Shape>> {};

TEST_P(ShapeTest, FromShapeProto) {
  const auto& [shape, shape_proto] = GetParam();
  EXPECT_EQ(FromShapeProto(shape_proto), shape);
}

TEST_P(ShapeTest, ToShapeProto) {
  const auto& [shape, shape_proto] = GetParam();
  EXPECT_THAT(ToShapeProto(shape), EquivToProto(shape_proto));
}

proto::Shape MakeProtoShape(absl::Span<const int64_t> dims) {
  auto shape = proto::Shape();
  for (auto dim : dims) {
    shape.add_dimensions(dim);
  }
  return shape;
}

INSTANTIATE_TEST_SUITE_P(Shape, ShapeTest,
                         testing::ValuesIn({
                             std::make_pair(Shape({}), MakeProtoShape({})),
                             std::make_pair(Shape({1, 2}),
                                            MakeProtoShape({1, 2})),
                         }));
#endif

// TODO(b/315809436): Test needs rewrite because protobuf matchers are not OSS
#if defined(PLATFORM_GOOGLE)
class VariantTest : public testing::TestWithParam<
                        std::pair<xla::PjRtValueType, proto::Variant>> {};

TEST_P(VariantTest, FromVariantProto) {
  const auto& [variant, variant_proto] = GetParam();
  EXPECT_THAT(FromVariantProto(variant_proto), IsOkAndHolds(variant));
}

TEST_P(VariantTest, ToVariantProto) {
  const auto& [variant, variant_proto] = GetParam();
  EXPECT_THAT(ToVariantProto(variant),
              IsOkAndHolds(EquivToProto(variant_proto)));
}

proto::Variant MakeProtoVariantString(absl::string_view arg) {
  auto variant = proto::Variant();
  variant.set_string_value(arg);
  return variant;
}

proto::Variant MakeProtoVariantInt64(int64_t arg) {
  auto variant = proto::Variant();
  variant.set_int64_value(arg);
  return variant;
}

proto::Variant MakeProtoVariantInt64List(absl::Span<const int64_t> arg) {
  auto variant = proto::Variant();
  for (auto arg : arg) {
    variant.mutable_int64_list()->add_values(arg);
  }
  return variant;
}

proto::Variant MakeProtoVariantFloat(float arg) {
  auto variant = proto::Variant();
  variant.set_float_value(arg);
  return variant;
}

INSTANTIATE_TEST_SUITE_P(
    Variant, VariantTest,
    testing::ValuesIn({
        std::make_pair(xla::PjRtValueType("foo"),
                       MakeProtoVariantString("foo")),
        std::make_pair(xla::PjRtValueType(static_cast<int64_t>(1234)),
                       MakeProtoVariantInt64(1234)),
        std::make_pair(xla::PjRtValueType(std::vector<int64_t>{1, 2}),
                       MakeProtoVariantInt64List({1, 2})),
        std::make_pair(xla::PjRtValueType(3.14f), MakeProtoVariantFloat(3.14f)),
    }));
#endif

class ByteStridesTest : public testing::TestWithParam<std::vector<int64_t>> {};

TEST_P(ByteStridesTest, ToFromProto) {
  std::vector<int64_t> strides = GetParam();
  EXPECT_EQ(FromByteStridesProto(ToByteStridesProto(strides)), strides);
}

INSTANTIATE_TEST_SUITE_P(
    ByteStrides, ByteStridesTest,
    testing::ValuesIn(std::vector<std::vector<int64_t>>{
        {}, {1}, {0}, {4, 8}, {8, 4}, {1, 2, 3, 4}, {0, 4}, {4, 0}}));

TEST(ArrayCopySemantics, ToFromProtoTest) {
  // NOLINTNEXTLINE readability-proto-enum-for-loop
  for (int proto_enum_int = proto::ArrayCopySemantics_MIN;
       proto_enum_int <= proto::ArrayCopySemantics_MAX; ++proto_enum_int) {
    const auto proto_enum =
        static_cast<proto::ArrayCopySemantics>(proto_enum_int);
    if (proto_enum == proto::ARRAY_COPY_SEMANTICS_UNSPECIFIED) {
      continue;
    }
    TF_ASSERT_OK_AND_ASSIGN(const auto cpp_enum,
                            FromArrayCopySemanticsProto(proto_enum));
    EXPECT_EQ(proto_enum, ToArrayCopySemanticsProto(cpp_enum));
  }
}

}  // namespace
}  // namespace proxy
}  // namespace ifrt
}  // namespace xla
