/* Copyright 2022 The OpenXLA Authors.

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

#include "xla/python/ifrt/index.h"

#include <cstdint>
#include <limits>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/hash/hash_testing.h"
#include "absl/status/status.h"
#include "absl/status/status_matchers.h"
#include "xla/python/ifrt/index.pb.h"
#include "xla/python/ifrt/serdes_test_util.h"
#include "xla/python/ifrt/serdes_version.h"
#include "xla/python/ifrt/shape.h"

namespace xla {
namespace ifrt {
namespace {

using ::absl_testing::IsOkAndHolds;
using ::absl_testing::StatusIs;
using ::testing::ElementsAre;
using ::testing::HasSubstr;
using ::testing::IsEmpty;

constexpr int64_t kSqrtMaxNumElements = int64_t{1} << 28;
static_assert(kSqrtMaxNumElements * kSqrtMaxNumElements ==
              Shape::kMaxNumElements + 1);

TEST(IndexTest, Construction) {
  EXPECT_THAT(Index({1, 2}).elements(), ElementsAre(1, 2));
  EXPECT_THAT(Index({0, Shape::kMaxNumElements}).elements(),
              ElementsAre(0, Shape::kMaxNumElements));
  EXPECT_THAT(Index::Zeros(0).elements(), IsEmpty());
  EXPECT_THAT(Index::Zeros(2).elements(), ElementsAre(0, 0));

  EXPECT_THAT(Index::Create({}), IsOkAndHolds(Index({})));
  ASSERT_OK_AND_ASSIGN(Index created,
                       Index::Create({0, Shape::kMaxNumElements}));
  EXPECT_THAT(created.elements(), ElementsAre(0, Shape::kMaxNumElements));
  EXPECT_OK(Index::Create({kSqrtMaxNumElements - 1, 0, kSqrtMaxNumElements}));
  EXPECT_THAT(Index::Create({1, -1}),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("Index expects non-negative element values")));
  EXPECT_THAT(Index::Create({0, Shape::kMaxNumElements + 1}),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("exceeds Shape::kMaxNumElements")));
  EXPECT_THAT(Index::Create({kSqrtMaxNumElements, 0, kSqrtMaxNumElements}),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("exceeds Shape::kMaxNumElements")));
  EXPECT_THAT(Index::Create({Shape::kMaxNumElements, Shape::kMaxNumElements}),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("exceeds Shape::kMaxNumElements")));
}

TEST(IndexDeathTest, InvalidConstructionAndArithmetic) {
  EXPECT_DEATH((void)Index({1, -1}),
               "Index expects non-negative element values");
  EXPECT_DEATH((void)Index({Shape::kMaxNumElements + 1}),
               "exceeds Shape::kMaxNumElements");
  EXPECT_DEATH((void)Index({kSqrtMaxNumElements, kSqrtMaxNumElements}),
               "exceeds Shape::kMaxNumElements");
  EXPECT_DEATH((void)Index::Zeros(-1), "");
  EXPECT_DEATH((void)(Index({1, 2}) - Index({2, 1})), "");
  EXPECT_DEATH((void)(Index({Shape::kMaxNumElements}) + Index({1})), "");
  EXPECT_DEATH(
      (void)(Index({kSqrtMaxNumElements, 0}) + Index({0, kSqrtMaxNumElements})),
      "");
  EXPECT_DEATH((void)(Index({Shape::kMaxNumElements, 0}) +
                      Index({0, Shape::kMaxNumElements})),
               "");
  EXPECT_DEATH((void)(Index({2}) * std::vector<int64_t>{-1}), "");
  EXPECT_DEATH((void)(Index({0}) * std::vector<int64_t>{-1}), "");
  EXPECT_DEATH(
      (void)(Index({Shape::kMaxNumElements}) * std::vector<int64_t>{2}), "");
  EXPECT_DEATH(
      (void)(Index({Shape::kMaxNumElements}) *
             std::vector<int64_t>{std::numeric_limits<int64_t>::max()}),
      "");
  EXPECT_DEATH(
      (void)(Index({Shape::kMaxNumElements / 2 + 1}) * std::vector<int64_t>{2}),
      "");
  EXPECT_DEATH(
      (void)(Index({1}) * std::vector<int64_t>{Shape::kMaxNumElements + 1}),
      "");
  EXPECT_DEATH((void)(Index({kSqrtMaxNumElements, 1}) *
                      std::vector<int64_t>{1, kSqrtMaxNumElements}),
               "");
  EXPECT_DEATH((void)(Index({Shape::kMaxNumElements, 1}) *
                      std::vector<int64_t>{1, Shape::kMaxNumElements}),
               "");
}

TEST(IndexTest, Operations) {
  EXPECT_EQ(Index({1, 2}), Index({1, 2}));
  EXPECT_NE(Index({1, 2}), Index({1, 3}));

  Index a({11, 22});
  Index b({2, 3});

  EXPECT_EQ(a + b, Index({13, 25}));
  {
    Index c = a;
    EXPECT_EQ(c += b, Index({13, 25}));
  }
  EXPECT_EQ(Index({Shape::kMaxNumElements}) + Index({0}),
            Index({Shape::kMaxNumElements}));
  EXPECT_EQ(
      Index({kSqrtMaxNumElements - 1, 0}) + Index({0, kSqrtMaxNumElements}),
      Index({kSqrtMaxNumElements - 1, kSqrtMaxNumElements}));

  EXPECT_EQ(a - b, Index({9, 19}));
  {
    Index c = a;
    EXPECT_EQ(c -= b, Index({9, 19}));
  }
  EXPECT_EQ(Index({2}) - Index({2}), Index({0}));

  EXPECT_EQ(a * std::vector<int64_t>({1, 2}), Index({11, 44}));
  {
    Index c = a;
    EXPECT_EQ(c *= std::vector<int64_t>({1, 2}), Index({11, 44}));
  }
  EXPECT_EQ(Index({Shape::kMaxNumElements}) * std::vector<int64_t>({0}),
            Index({0}));
  EXPECT_EQ(Index({0}) * std::vector<int64_t>({Shape::kMaxNumElements + 1}),
            Index({0}));
  EXPECT_EQ(
      Index({0}) * std::vector<int64_t>({std::numeric_limits<int64_t>::max()}),
      Index({0}));
  EXPECT_EQ(Index({Shape::kMaxNumElements / 2}) * std::vector<int64_t>({2}),
            Index({(Shape::kMaxNumElements / 2) * 2}));
  EXPECT_EQ(Index({kSqrtMaxNumElements - 1, 1}) *
                std::vector<int64_t>({1, kSqrtMaxNumElements}),
            Index({kSqrtMaxNumElements - 1, kSqrtMaxNumElements}));
}

TEST(IndexTest, Hash) {
  EXPECT_TRUE(absl::VerifyTypeImplementsAbslHashCorrectly({
      Index({}),
      Index({1}),
      Index({2}),
      Index({1, 2}),
      Index({1, 3}),
      Index({2, 1}),
      Index({1, 2, 3}),
      Index({1, 2, 4}),
  }));
}

class IndexSerDesTest : public testing::TestWithParam<SerDesVersion> {
 public:
  IndexSerDesTest() : version_(GetParam()) {}

  SerDesVersion version() const { return version_; }

 private:
  SerDesVersion version_;
};

TEST_P(IndexSerDesTest, ToFromProto) {
  {
    Index index({});
    IndexProto proto = index.ToProto(version());
    ASSERT_OK_AND_ASSIGN(Index index_copy, Index::FromProto(proto));
    EXPECT_EQ(index_copy, index);
  }
  {
    Index index({1, 2});
    IndexProto proto = index.ToProto(version());
    ASSERT_OK_AND_ASSIGN(Index index_copy, Index::FromProto(proto));
    EXPECT_EQ(index_copy, index);
  }
}

TEST(IndexTest, FromProtoRejectsInvalidElements) {
  {
    IndexProto proto;
    proto.set_version_number(SerDesVersionNumber(0).value());
    proto.add_elements(3);
    proto.add_elements(-1);
    EXPECT_THAT(
        Index::FromProto(proto),
        StatusIs(absl::StatusCode::kInvalidArgument,
                 HasSubstr("Index expects non-negative element values")));
  }
  {
    IndexProto proto;
    proto.set_version_number(SerDesVersionNumber(0).value());
    proto.add_elements(3);
    proto.add_elements(Shape::kMaxNumElements + 1);
    EXPECT_THAT(Index::FromProto(proto),
                StatusIs(absl::StatusCode::kInvalidArgument,
                         HasSubstr("exceeds Shape::kMaxNumElements")));
  }
  {
    IndexProto proto;
    proto.set_version_number(SerDesVersionNumber(0).value());
    proto.add_elements(kSqrtMaxNumElements);
    proto.add_elements(kSqrtMaxNumElements);
    EXPECT_THAT(Index::FromProto(proto),
                StatusIs(absl::StatusCode::kInvalidArgument,
                         HasSubstr("exceeds Shape::kMaxNumElements")));
  }
}

INSTANTIATE_TEST_SUITE_P(
    SerDesVersion, IndexSerDesTest,
    testing::ValuesIn(test_util::AllSupportedSerDesVersions()));

}  // namespace
}  // namespace ifrt
}  // namespace xla
