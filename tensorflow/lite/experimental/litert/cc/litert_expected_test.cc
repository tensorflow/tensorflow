// Copyright 2024 Google LLC.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "tensorflow/lite/experimental/litert/cc/litert_expected.h"

#include <cstdint>
#include <initializer_list>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "tensorflow/lite/experimental/litert/c/litert_common.h"
#include "tensorflow/lite/experimental/litert/cc/litert_buffer_ref.h"

namespace litert {

namespace {

static constexpr LiteRtStatus kErrorStatus = kLiteRtStatusErrorInvalidArgument;

struct TypeWithAllocation {
  TypeWithAllocation(std::initializer_list<int> il) : allocated(il) {}
  std::vector<int> allocated;
};

struct TypeWithFields {
  TypeWithFields(int i_, int j_) : i(i_), j(j_) {}
  int i;
  int j;
};

TEST(ExpectedTest, PrimitiveExplicit) {
  Expected<float> exp(1.0);
  ASSERT_TRUE(exp.HasValue());
}

TEST(ExpectedTest, PrimitiveImplicit) {
  Expected<float> exp = 1.0;
  ASSERT_TRUE(exp.HasValue());
}

TEST(ExpectedTest, ClassWithAllocation) {
  Expected<TypeWithAllocation> exp(TypeWithAllocation({1, 2, 3}));
  ASSERT_TRUE(exp.HasValue());
}

TEST(ExpectedTest, ClassWithFields) {
  Expected<TypeWithFields> exp(TypeWithFields(1, 2));
  ASSERT_TRUE(exp.HasValue());
}

TEST(ExpectedTest, FromErrorExplicit) {
  Expected<TypeWithAllocation> exp((Unexpected(kErrorStatus, "MESSAGE")));
  ASSERT_FALSE(exp.HasValue());
}

TEST(ExpectedTest, FromErrorImplicit) {
  Expected<TypeWithAllocation> exp = Unexpected(kErrorStatus);
  ASSERT_FALSE(exp.HasValue());
}

TEST(ExpectedTest, CopyCstorError) {
  const Expected<int> exp = Unexpected(kErrorStatus);
  Expected<int> other(exp);
  ASSERT_FALSE(other.HasValue());
  EXPECT_EQ(other.Error().Status(), kErrorStatus);
}

TEST(ExpectedTest, CopyCstorVal) {
  const Expected<int> exp = 2;
  Expected<int> other(exp);
  ASSERT_TRUE(other.HasValue());
  EXPECT_EQ(other.Value(), 2);
}

TEST(ExpectedTest, CopyAssignError) {
  const Expected<int> exp = Unexpected(kErrorStatus);
  ASSERT_FALSE(exp.HasValue());
  Expected<int> other = exp;
  ASSERT_FALSE(other.HasValue());
  EXPECT_EQ(other.Error().Status(), kErrorStatus);
}

TEST(ExpectedTest, CopyAssignVal) {
  const Expected<int> exp = 2;
  Expected<int> other = exp;
  ASSERT_TRUE(other.HasValue());
  EXPECT_EQ(other.Value(), 2);
}

TEST(ExpectedTest, MoveCstorError) {
  Expected<int> exp = Unexpected(kErrorStatus);
  Expected<int> other(std::move(exp));
  ASSERT_FALSE(other.HasValue());
  EXPECT_EQ(other.Error().Status(), kErrorStatus);
}

TEST(ExpectedTest, MoveCstorVal) {
  Expected<int> exp = 2;
  Expected<int> other(std::move(exp));
  ASSERT_TRUE(other.HasValue());
  EXPECT_EQ(other.Value(), 2);
}

TEST(ExpectedTest, MoveAssignError) {
  Expected<int> exp = Unexpected(kErrorStatus);
  Expected<int> other = std::move(exp);
  ASSERT_FALSE(other.HasValue());
  EXPECT_EQ(other.Error().Status(), kErrorStatus);
}

TEST(ExpectedTest, MoveAssignVal) {
  Expected<int> exp = 2;
  Expected<int> other = std::move(exp);
  ASSERT_TRUE(other.HasValue());
  EXPECT_EQ(other.Value(), 2);
}

TEST(ExpectedTest, Indirection) {
  Expected<TypeWithFields> exp(TypeWithFields(1, 2));
  EXPECT_EQ(exp->i, 1);
  EXPECT_EQ(exp->j, 2);
}

TEST(ExpectedTest, Dereference) {
  Expected<TypeWithFields> exp(TypeWithFields(1, 2));
  const auto& val = *exp;
  EXPECT_EQ(val.i, 1);
  EXPECT_EQ(val.j, 2);
}

TEST(UnexpectedTest, WithStatus) {
  Unexpected err(kErrorStatus);
  EXPECT_EQ(err.Error().Status(), kErrorStatus);
  EXPECT_TRUE(err.Error().Message().empty());
}

TEST(UnexpectedTest, WithMessage) {
  Unexpected err(kErrorStatus, "MESSAGE");
  EXPECT_EQ(err.Error().Status(), kErrorStatus);
  EXPECT_EQ(err.Error().Message(), "MESSAGE");
}

Expected<OwningBufferRef<uint8_t>> Go() {
  std::string data = "21234";
  OwningBufferRef<uint8_t> buf(data.c_str());
  return buf;
}

Expected<OwningBufferRef<uint8_t>> Forward() {
  auto thing = Go();
  if (!thing.HasValue()) {
    return thing.Error();
  }
  // No copy ellision here.
  return thing;
}

TEST(ExpectedTest, ForwardBufThroughFuncs) {
  auto res = Forward();
  EXPECT_TRUE(res.HasValue());
  EXPECT_EQ(res->StrView(), "21234");
}

TEST(ExpectedWithNoValue, WithoutError) {
  Expected<void> expected = {};
  EXPECT_TRUE(expected.HasValue());
}

TEST(ExpectedWithNoValue, WithError) {
  Expected<void> expected(Unexpected(kErrorStatus, "MESSAGE"));
  EXPECT_FALSE(expected.HasValue());
  EXPECT_EQ(expected.Error().Status(), kErrorStatus);
  EXPECT_EQ(expected.Error().Message(), "MESSAGE");
}

TEST(ExpectedWithNoValue, OStreamOutput) {
  Expected<void> expected(Unexpected(kErrorStatus, "MESSAGE"));
  std::ostringstream oss;
  oss << expected.Error();
  EXPECT_THAT(oss.str(), testing::HasSubstr("MESSAGE"));
}

}  // namespace

}  // namespace litert
