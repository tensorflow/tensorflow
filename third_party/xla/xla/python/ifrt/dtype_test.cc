// Copyright 2024 The OpenXLA Authors.
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

#include "xla/python/ifrt/dtype.h"

#include <optional>
#include <tuple>
#include <vector>

#include <gtest/gtest.h>
#include "xla/python/ifrt/dtype.pb.h"
#include "xla/tsl/platform/statusor.h"
#include "xla/tsl/platform/test.h"

namespace xla {
namespace ifrt {
namespace {

TEST(DTypeTest, FromToFromProto) {
  for (int i = 0; i < DTypeProto::Kind_descriptor()->value_count(); ++i) {
    DTypeProto proto;
    proto.set_kind(static_cast<DTypeProto::Kind>(
        DTypeProto::Kind_descriptor()->value(i)->number()));
    TF_ASSERT_OK_AND_ASSIGN(DType dtype, DType::FromProto(proto));
    TF_ASSERT_OK_AND_ASSIGN(DType dtype_copy,
                            DType::FromProto(dtype.ToProto()));
    EXPECT_EQ(dtype_copy, dtype);
  }
}

TEST(DTypeTest, ByteSize) {
  for (const auto& [kind, byte_size] :
       std::vector<std::tuple<DType::Kind, int>>({
           {DType::kS2, -1},        {DType::kU2, -1},
           {DType::kS4, -1},        {DType::kU4, -1},
           {DType::kPred, 1},       {DType::kS8, 1},
           {DType::kU8, 1},         {DType::kF4E2M1FN, -1},
           {DType::kF8E3M4, 1},     {DType::kF8E4M3, 1},
           {DType::kF8E4M3FN, 1},   {DType::kF8E4M3B11FNUZ, 1},
           {DType::kF8E4M3FNUZ, 1}, {DType::kF8E5M2, 1},
           {DType::kF8E5M2FNUZ, 1}, {DType::kF8E8M0FNU, 1},
           {DType::kS16, 2},        {DType::kU16, 2},
           {DType::kF16, 2},        {DType::kBF16, 2},
           {DType::kS32, 4},        {DType::kU32, 4},
           {DType::kF32, 4},        {DType::kS64, 8},
           {DType::kU64, 8},        {DType::kF64, 8},
           {DType::kC64, 8},        {DType::kC128, 16},
           {DType::kToken, -1},     {DType::kInvalid, -1},
           {DType::kString, -1},
       })) {
    EXPECT_EQ(DType(kind).byte_size(),
              byte_size == -1 ? std::nullopt : std::make_optional(byte_size));
  }
}

TEST(DTypeTest, BitSize) {
  for (const auto& [kind, bit_size] :
       std::vector<std::tuple<DType::Kind, int>>({
           {DType::kS2, 2},         {DType::kU2, 2},
           {DType::kS4, 4},         {DType::kU4, 4},
           {DType::kPred, 8},       {DType::kS8, 8},
           {DType::kU8, 8},         {DType::kF4E2M1FN, 4},
           {DType::kF8E3M4, 8},     {DType::kF8E4M3, 8},
           {DType::kF8E4M3FN, 8},   {DType::kF8E4M3B11FNUZ, 8},
           {DType::kF8E4M3FNUZ, 8}, {DType::kF8E5M2, 8},
           {DType::kF8E5M2FNUZ, 8}, {DType::kF8E8M0FNU, 8},
           {DType::kS16, 16},       {DType::kU16, 16},
           {DType::kF16, 16},       {DType::kBF16, 16},
           {DType::kS32, 32},       {DType::kU32, 32},
           {DType::kF32, 32},       {DType::kS64, 64},
           {DType::kU64, 64},       {DType::kF64, 64},
           {DType::kC64, 64},       {DType::kC128, 128},
           {DType::kToken, -1},     {DType::kInvalid, -1},
           {DType::kString, -1},
       })) {
    EXPECT_EQ(DType(kind).bit_size(),
              bit_size == -1 ? std::nullopt : std::make_optional(bit_size));
  }
}

}  // namespace
}  // namespace ifrt
}  // namespace xla
