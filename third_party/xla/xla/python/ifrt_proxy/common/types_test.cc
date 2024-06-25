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
#include <string>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "xla/pjrt/pjrt_common.h"
#include "xla/python/ifrt_proxy/common/types.pb.h"
#include "tsl/platform/status_matchers.h"
#include "tsl/platform/statusor.h"
#include "tsl/platform/test.h"

namespace xla {
namespace ifrt {
namespace proxy {
namespace {

using ::tsl::testing::IsOkAndHolds;

class VariantTest : public testing::TestWithParam<xla::PjRtValueType> {};

TEST_P(VariantTest, ToFromVariantProto) {
  const auto& variant = GetParam();
  TF_ASSERT_OK_AND_ASSIGN(proto::Variant variant_proto,
                          ToVariantProto(variant));
  EXPECT_THAT(FromVariantProto(variant_proto), IsOkAndHolds(variant));
}

INSTANTIATE_TEST_SUITE_P(
    Variant, VariantTest,
    testing::Values(xla::PjRtValueType(std::string("foo")),
                    xla::PjRtValueType(static_cast<int64_t>(1234)),
                    xla::PjRtValueType(std::vector<int64_t>{1, 2}),
                    xla::PjRtValueType(3.14f)));

class ByteStridesTest : public testing::TestWithParam<std::vector<int64_t>> {};

TEST_P(ByteStridesTest, ToFromProto) {
  std::vector<int64_t> strides = GetParam();
  EXPECT_EQ(FromByteStridesProto(ToByteStridesProto(strides)), strides);
}

INSTANTIATE_TEST_SUITE_P(
    ByteStrides, ByteStridesTest,
    testing::ValuesIn(std::vector<std::vector<int64_t>>{
        {}, {1}, {0}, {4, 8}, {8, 4}, {1, 2, 3, 4}, {0, 4}, {4, 0}}));

TEST(ArrayCopySemanticsTest, FromToFromProto) {
  for (int i = 0; i < proto::ArrayCopySemantics_descriptor()->value_count();
       ++i) {
    const auto proto_enum = static_cast<proto::ArrayCopySemantics>(
        proto::ArrayCopySemantics_descriptor()->value(i)->number());
    if (proto_enum == proto::ARRAY_COPY_SEMANTICS_UNSPECIFIED) {
      continue;
    }
    TF_ASSERT_OK_AND_ASSIGN(const auto cpp_enum,
                            FromArrayCopySemanticsProto(proto_enum));
    TF_ASSERT_OK_AND_ASSIGN(
        const auto cpp_enum_copy,
        FromArrayCopySemanticsProto(ToArrayCopySemanticsProto(cpp_enum)));
    EXPECT_EQ(cpp_enum_copy, cpp_enum);
  }
}

}  // namespace
}  // namespace proxy
}  // namespace ifrt
}  // namespace xla
