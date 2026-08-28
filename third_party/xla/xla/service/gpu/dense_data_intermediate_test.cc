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

#include "xla/service/gpu/dense_data_intermediate.h"

#include <array>
#include <cstdint>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "xla/literal.h"
#include "xla/literal_util.h"
#include "xla/service/gpu/dense_data_intermediate.pb.h"
#include "xla/types.h"

namespace xla {
namespace gpu {
namespace {

using ::testing::ElementsAreArray;

TEST(DenseDataIntermediateTest, LiteralToAttrToXlaFormat) {
  // int16, should be aliased.
  {
    Literal literal = LiteralUtil::CreateR2<int16_t>({{0, 1, 2}, {3, 4, 5}});

    ASSERT_OK_AND_ASSIGN(DenseDataIntermediate data,
                         LiteralToXlaFormat(literal));
    EXPECT_EQ(data.span().size(), literal.size_bytes());
    EXPECT_EQ(reinterpret_cast<const char*>(data.span().data()),
              literal.untyped_data());
  }

  // int4, even, should be a new (unaliased) packed array.
  {
    Literal literal = LiteralUtil::CreateR2<s4>(
        {{s4(0), s4(1), s4(2)}, {s4(3), s4(4), s4(5)}});

    ASSERT_OK_AND_ASSIGN(DenseDataIntermediate data,
                         LiteralToXlaFormat(literal));
    EXPECT_EQ(data.span(), std::vector<uint8_t>({0x10, 0x32, 0x54}));
    EXPECT_NE(reinterpret_cast<const void*>(data.span().data()),
              literal.untyped_data());
  }

  // int4, odd, should be a new (unaliased) packed array.
  {
    Literal literal = LiteralUtil::CreateR2<u4>(
        {{u4(0), u4(1), u4(2)}, {u4(3), u4(4), u4(5)}, {u4(6), u4(7), u4(8)}});

    ASSERT_OK_AND_ASSIGN(DenseDataIntermediate data,
                         LiteralToXlaFormat(literal));
    EXPECT_EQ(data.span(),
              std::vector<uint8_t>({0x10, 0x32, 0x54, 0x76, 0x08}));
    EXPECT_NE(reinterpret_cast<const void*>(data.span().data()),
              literal.untyped_data());
  }
}

TEST(DenseDataIntermediateTest, OwnedDataToProto) {
  const std::vector<uint8_t> data = {1, 2, 3, 4};
  DenseDataIntermediate constant = DenseDataIntermediate::Own(data);

  DenseDataIntermediateProto proto = constant.ToProto();
  EXPECT_THAT(proto.data(), ElementsAreArray(data));
}

TEST(DenseDataIntermediateTest, BorrowedDataToProto) {
  constexpr std::array<uint8_t, 4> kData = {5, 6, 7, 8};
  DenseDataIntermediate constant = DenseDataIntermediate::Alias(kData);
  DenseDataIntermediateProto proto = constant.ToProto();
  EXPECT_THAT(proto.data(), ElementsAreArray(kData));
}

TEST(DenseDataIntermediateTest, FromProto) {
  constexpr std::array<uint8_t, 4> kData = {1, 2, 3, 4};
  DenseDataIntermediateProto proto;
  proto.mutable_data()->assign(kData.begin(), kData.end());

  DenseDataIntermediate constant = DenseDataIntermediate::FromProto(proto);
  EXPECT_THAT(constant.span(), ElementsAreArray(kData));
}

}  // namespace
}  // namespace gpu
}  // namespace xla
