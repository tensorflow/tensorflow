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

#include "xla/python/ifrt/index_domain.h"

#include <cstdint>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/hash/hash_testing.h"
#include "absl/status/status.h"
#include "absl/status/status_matchers.h"
#include "xla/python/ifrt/index.h"
#include "xla/python/ifrt/index_domain.pb.h"
#include "xla/python/ifrt/serdes_test_util.h"
#include "xla/python/ifrt/serdes_version.h"
#include "xla/python/ifrt/shape.h"

namespace xla {
namespace ifrt {
namespace {

using ::absl_testing::IsOkAndHolds;
using ::absl_testing::StatusIs;
using ::testing::HasSubstr;

constexpr int64_t kSqrtMaxNumElements = int64_t{1} << 28;
static_assert(kSqrtMaxNumElements * kSqrtMaxNumElements ==
              Shape::kMaxNumElements + 1);

TEST(IndexDomainTest, Construction) {
  IndexDomain a(Index({1, 2}), Shape({3, 4}));
  EXPECT_EQ(a.origin(), Index({1, 2}));
  EXPECT_EQ(a.shape(), Shape({3, 4}));

  IndexDomain b(Shape({3, 4}));
  EXPECT_EQ(b.origin(), Index({0, 0}));
  EXPECT_EQ(b.shape(), Shape({3, 4}));

  EXPECT_THAT(IndexDomain::Create(Index({1, 2}), Shape({3, 4})),
              IsOkAndHolds(a));

  EXPECT_OK(IndexDomain::Create(Index({}), Shape({})));
  EXPECT_OK(
      IndexDomain::Create(Index({Shape::kMaxNumElements, 0}), Shape({0, 0})));
  EXPECT_OK(
      IndexDomain::Create(Index({0, 0}), Shape({0, Shape::kMaxNumElements})));
  EXPECT_OK(
      IndexDomain::Create(Index({Shape::kMaxNumElements - 5}), Shape({5})));
  EXPECT_OK(IndexDomain::Create(Index({kSqrtMaxNumElements - 1, 0}),
                                Shape({0, kSqrtMaxNumElements})));
  EXPECT_OK(IndexDomain::Create(Index({kSqrtMaxNumElements - 2, 0}),
                                Shape({1, kSqrtMaxNumElements})));
  EXPECT_THAT(IndexDomain::Create(Index({1}), Shape({3, 4})),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("same number of dimensions")));
  EXPECT_THAT(IndexDomain::Create(Index({1, 2}), Shape({3})),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("same number of dimensions")));
  EXPECT_THAT(IndexDomain::Create(Index({}), Shape({1})),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("same number of dimensions")));
  EXPECT_THAT(IndexDomain::Create(Index({1}), Shape({})),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("same number of dimensions")));
  EXPECT_THAT(
      IndexDomain::Create(Index({Shape::kMaxNumElements, 0}), Shape({1, 2})),
      StatusIs(absl::StatusCode::kInvalidArgument,
               HasSubstr("exceeds Shape::kMaxNumElements")));
  EXPECT_THAT(
      IndexDomain::Create(Index({Shape::kMaxNumElements - 5}), Shape({6})),
      StatusIs(absl::StatusCode::kInvalidArgument,
               HasSubstr("exceeds Shape::kMaxNumElements")));
  EXPECT_THAT(IndexDomain::Create(Index({Shape::kMaxNumElements, 0}),
                                  Shape({0, Shape::kMaxNumElements})),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("exceeds Shape::kMaxNumElements")));
  EXPECT_THAT(IndexDomain::Create(Index({kSqrtMaxNumElements, 0}),
                                  Shape({0, kSqrtMaxNumElements})),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("exceeds Shape::kMaxNumElements")));
  EXPECT_THAT(IndexDomain::Create(Index({kSqrtMaxNumElements - 1, 0}),
                                  Shape({1, kSqrtMaxNumElements})),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("exceeds Shape::kMaxNumElements")));
  EXPECT_THAT(IndexDomain::Create(Index({kSqrtMaxNumElements, 0, 0}),
                                  Shape({0, 0, kSqrtMaxNumElements})),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("exceeds Shape::kMaxNumElements")));
}

TEST(IndexDomainDeathTest, InvalidConstructionAndOffsetOverflow) {
  EXPECT_DEATH((void)IndexDomain(Index({1}), Shape({3, 4})),
               "same number of dimensions");
  EXPECT_DEATH((void)IndexDomain(Index({1, 2}), Shape({3})),
               "same number of dimensions");
  EXPECT_DEATH((void)IndexDomain(Index({Shape::kMaxNumElements}), Shape({1})),
               "exceeds Shape::kMaxNumElements");
  EXPECT_DEATH(
      (void)IndexDomain(Index({Shape::kMaxNumElements - 5}), Shape({6})),
      "exceeds Shape::kMaxNumElements");
  EXPECT_DEATH((void)IndexDomain(Index({kSqrtMaxNumElements, 0}),
                                 Shape({0, kSqrtMaxNumElements})),
               "exceeds Shape::kMaxNumElements");
  IndexDomain domain(Index({10}), Shape({5}));
  EXPECT_DEATH((void)(domain + Index({Shape::kMaxNumElements - 12})),
               "exceeds Shape::kMaxNumElements");
  EXPECT_DEATH((void)(domain += Index({Shape::kMaxNumElements - 12})),
               "exceeds Shape::kMaxNumElements");
  IndexDomain multi_dim_domain(Index({0, 0}), Shape({0, kSqrtMaxNumElements}));
  EXPECT_DEATH((void)(multi_dim_domain + Index({kSqrtMaxNumElements, 0})),
               "exceeds Shape::kMaxNumElements");
  EXPECT_DEATH((void)(multi_dim_domain += Index({kSqrtMaxNumElements, 0})),
               "exceeds Shape::kMaxNumElements");
  EXPECT_DEATH((void)(domain + Index({1, 2})), "");
  EXPECT_DEATH((void)(domain += Index({1, 2})), "");
  EXPECT_DEATH((void)(domain - Index({1, 2})), "");
  EXPECT_DEATH((void)(domain -= Index({1, 2})), "");
  EXPECT_DEATH((void)(domain - Index({11})), "");
  EXPECT_DEATH((void)(domain -= Index({11})), "");
}

TEST(IndexDomainTest, Operations) {
  IndexDomain a(Index({1, 2}), Shape({3, 4}));
  Index b({1, 2});

  EXPECT_EQ(a + b, IndexDomain(Index({2, 4}), Shape({3, 4})));
  {
    IndexDomain c = a;
    EXPECT_EQ(c += b, IndexDomain(Index({2, 4}), Shape({3, 4})));
  }

  EXPECT_EQ(a - b, IndexDomain(Index({0, 0}), Shape({3, 4})));
  {
    IndexDomain c = a;
    EXPECT_EQ(c -= b, IndexDomain(Index({0, 0}), Shape({3, 4})));
  }
}

TEST(IndexDomainTest, Intersect) {
  // Scalar domains.
  {
    IndexDomain a(Index({}), Shape({}));
    IndexDomain b(Index({}), Shape({}));
    EXPECT_THAT(a.Intersect(b),
                IsOkAndHolds(IndexDomain(Index({}), Shape({}))));
  }

  // Identical domains.
  {
    IndexDomain a(Index({1, 2}), Shape({3, 4}));
    EXPECT_THAT(a.Intersect(a), IsOkAndHolds(a));
  }

  // Partially overlapping domains.
  {
    IndexDomain a(Index({1, 2}), Shape({3, 4}));
    IndexDomain b(Index({2, 0}), Shape({4, 5}));
    EXPECT_THAT(a.Intersect(b),
                IsOkAndHolds(IndexDomain(Index({2, 2}), Shape({2, 3}))));
    EXPECT_THAT(b.Intersect(a),
                IsOkAndHolds(IndexDomain(Index({2, 2}), Shape({2, 3}))));
  }

  // Contained domain.
  {
    IndexDomain a(Index({0, 0}), Shape({6, 8}));
    IndexDomain b(Index({2, 3}), Shape({2, 4}));
    EXPECT_THAT(a.Intersect(b), IsOkAndHolds(b));
    EXPECT_THAT(b.Intersect(a), IsOkAndHolds(b));
  }

  // Touching at boundary (disjoint).
  {
    IndexDomain a(Index({0, 0}), Shape({2, 4}));
    IndexDomain b(Index({2, 1}), Shape({2, 2}));
    EXPECT_THAT(a.Intersect(b),
                IsOkAndHolds(IndexDomain(Index({0, 0}), Shape({0, 0}))));
  }

  // Separated in one dimension (disjoint).
  {
    IndexDomain a(Index({0, 0}), Shape({2, 4}));
    IndexDomain b(Index({3, 1}), Shape({2, 2}));
    EXPECT_THAT(a.Intersect(b),
                IsOkAndHolds(IndexDomain(Index({0, 0}), Shape({0, 0}))));
  }

  // Zero-sized input domain.
  {
    IndexDomain a(Index({1, 2}), Shape({0, 4}));
    IndexDomain b(Index({0, 0}), Shape({4, 4}));
    EXPECT_THAT(a.Intersect(b),
                IsOkAndHolds(IndexDomain(Index({0, 0}), Shape({0, 0}))));
  }

  // Mismatched numbers of dimensions.
  {
    IndexDomain a(Index({1, 2}), Shape({3, 4}));
    IndexDomain b(Index({1}), Shape({3}));
    EXPECT_THAT(a.Intersect(b), StatusIs(absl::StatusCode::kInvalidArgument));
  }

  // Domains at the `Shape::kMaxNumElements` upper-bound boundary.
  {
    IndexDomain a(Index({Shape::kMaxNumElements - 10}), Shape({10}));
    IndexDomain b(Index({Shape::kMaxNumElements - 5}), Shape({5}));
    EXPECT_THAT(a.Intersect(b), IsOkAndHolds(b));
    EXPECT_THAT(b.Intersect(a), IsOkAndHolds(b));
  }
  {
    IndexDomain a(Index({kSqrtMaxNumElements - 2, 0}),
                  Shape({1, kSqrtMaxNumElements}));
    IndexDomain b(Index({kSqrtMaxNumElements - 2, 10}),
                  Shape({1, kSqrtMaxNumElements - 10}));
    EXPECT_THAT(a.Intersect(b), IsOkAndHolds(b));
    EXPECT_THAT(b.Intersect(a), IsOkAndHolds(b));
  }
}

TEST(IndexDomainTest, Contains) {
  // Scalar domains.
  {
    IndexDomain a(Index({}), Shape({}));
    IndexDomain b(Index({}), Shape({}));
    EXPECT_THAT(a.Contains(b), IsOkAndHolds(true));
  }

  // Identical domains.
  {
    IndexDomain a(Index({1, 2}), Shape({3, 4}));
    EXPECT_THAT(a.Contains(a), IsOkAndHolds(true));
  }

  // Contained domain.
  {
    IndexDomain a(Index({0, 0}), Shape({6, 8}));
    IndexDomain b(Index({2, 3}), Shape({2, 4}));
    EXPECT_THAT(a.Contains(b), IsOkAndHolds(true));
    EXPECT_THAT(b.Contains(a), IsOkAndHolds(false));
  }

  // Partially overlapping domains.
  {
    IndexDomain a(Index({1, 2}), Shape({3, 4}));
    IndexDomain b(Index({2, 0}), Shape({4, 5}));
    EXPECT_THAT(a.Contains(b), IsOkAndHolds(false));
    EXPECT_THAT(b.Contains(a), IsOkAndHolds(false));
  }

  // Disjoint domains.
  {
    IndexDomain a(Index({2, 0}), Shape({4, 4}));
    IndexDomain b(Index({0, 0}), Shape({2, 2}));
    EXPECT_THAT(a.Contains(b), IsOkAndHolds(false));

    IndexDomain c(Index({7, 5}), Shape({2, 2}));
    EXPECT_THAT(a.Contains(c), IsOkAndHolds(false));
  }

  // Zero-sized domain.
  {
    IndexDomain a(Shape({4, 4}));
    IndexDomain zero(Shape({0, 0}));
    EXPECT_THAT(a.Contains(zero), IsOkAndHolds(true));
  }

  // Mismatched numbers of dimensions.
  {
    IndexDomain a(Index({1, 2}), Shape({3, 4}));
    IndexDomain b(Index({1}), Shape({3}));
    EXPECT_THAT(a.Contains(b), StatusIs(absl::StatusCode::kInvalidArgument));
  }

  // Domains at the `Shape::kMaxNumElements` upper-bound boundary.
  {
    IndexDomain a(Index({Shape::kMaxNumElements - 10}), Shape({10}));
    IndexDomain b(Index({Shape::kMaxNumElements - 5}), Shape({5}));
    EXPECT_THAT(a.Contains(b), IsOkAndHolds(true));
    EXPECT_THAT(b.Contains(a), IsOkAndHolds(false));
  }
  {
    IndexDomain a(Index({kSqrtMaxNumElements - 2, 0}),
                  Shape({1, kSqrtMaxNumElements}));
    IndexDomain b(Index({kSqrtMaxNumElements - 2, 10}),
                  Shape({1, kSqrtMaxNumElements - 10}));
    EXPECT_THAT(a.Contains(b), IsOkAndHolds(true));
    EXPECT_THAT(b.Contains(a), IsOkAndHolds(false));
  }
}

TEST(IndexDomainTest, Hash) {
  EXPECT_TRUE(absl::VerifyTypeImplementsAbslHashCorrectly(
      {IndexDomain(Index({1, 2}), Shape({3, 4})),
       IndexDomain(Index({1, 2}), Shape({4, 3})),
       IndexDomain(Index({2, 1}), Shape({3, 4})),
       IndexDomain(Index({2, 1}), Shape({4, 3}))}));
}

class IndexDomainSerDesTest : public testing::TestWithParam<SerDesVersion> {
 public:
  IndexDomainSerDesTest() : version_(GetParam()) {}

  SerDesVersion version() const { return version_; }

 private:
  SerDesVersion version_;
};

TEST_P(IndexDomainSerDesTest, ToFromProto) {
  {
    IndexDomain index_domain(Index({}), Shape({}));
    IndexDomainProto proto = index_domain.ToProto(version());
    ASSERT_OK_AND_ASSIGN(IndexDomain index_domain_copy,
                         IndexDomain::FromProto(proto));
    EXPECT_EQ(index_domain_copy, index_domain);
  }
  {
    IndexDomain index_domain(Index({1, 2}), Shape({3, 4}));
    IndexDomainProto proto = index_domain.ToProto(version());
    ASSERT_OK_AND_ASSIGN(IndexDomain index_domain_copy,
                         IndexDomain::FromProto(proto));
    EXPECT_EQ(index_domain_copy, index_domain);
  }
}

TEST(IndexDomainTest,
     FromProtoRejectsMismatchedNumDimensionsAndUpperBoundOverflow) {
  {
    IndexDomainProto proto;
    proto.set_version_number(SerDesVersionNumber(0).value());
    *proto.mutable_origin() = Index({1}).ToProto();
    *proto.mutable_shape() = Shape({3, 4}).ToProto();
    EXPECT_THAT(IndexDomain::FromProto(proto),
                StatusIs(absl::StatusCode::kInvalidArgument,
                         HasSubstr("same number of dimensions")));
  }
  {
    IndexDomainProto proto;
    proto.set_version_number(SerDesVersionNumber(0).value());
    *proto.mutable_origin() = Index({1, 2}).ToProto();
    *proto.mutable_shape() = Shape({3}).ToProto();
    EXPECT_THAT(IndexDomain::FromProto(proto),
                StatusIs(absl::StatusCode::kInvalidArgument,
                         HasSubstr("same number of dimensions")));
  }
  {
    IndexDomainProto proto;
    proto.set_version_number(SerDesVersionNumber(0).value());
    *proto.mutable_origin() = Index({Shape::kMaxNumElements}).ToProto();
    *proto.mutable_shape() = Shape({1}).ToProto();
    EXPECT_THAT(IndexDomain::FromProto(proto),
                StatusIs(absl::StatusCode::kInvalidArgument,
                         HasSubstr("exceeds Shape::kMaxNumElements")));
  }
  {
    IndexDomainProto proto;
    proto.set_version_number(SerDesVersionNumber(0).value());
    *proto.mutable_origin() = Index({Shape::kMaxNumElements - 5}).ToProto();
    *proto.mutable_shape() = Shape({6}).ToProto();
    EXPECT_THAT(IndexDomain::FromProto(proto),
                StatusIs(absl::StatusCode::kInvalidArgument,
                         HasSubstr("exceeds Shape::kMaxNumElements")));
  }
  {
    IndexDomainProto proto;
    proto.set_version_number(SerDesVersionNumber(0).value());
    *proto.mutable_origin() = Index({Shape::kMaxNumElements, 0}).ToProto();
    *proto.mutable_shape() = Shape({0, Shape::kMaxNumElements}).ToProto();
    EXPECT_THAT(IndexDomain::FromProto(proto),
                StatusIs(absl::StatusCode::kInvalidArgument,
                         HasSubstr("exceeds Shape::kMaxNumElements")));
  }
  {
    IndexDomainProto proto;
    proto.set_version_number(SerDesVersionNumber(0).value());
    *proto.mutable_origin() = Index({kSqrtMaxNumElements, 0}).ToProto();
    *proto.mutable_shape() = Shape({0, kSqrtMaxNumElements}).ToProto();
    EXPECT_THAT(IndexDomain::FromProto(proto),
                StatusIs(absl::StatusCode::kInvalidArgument,
                         HasSubstr("exceeds Shape::kMaxNumElements")));
  }
}

INSTANTIATE_TEST_SUITE_P(
    SerDesVersion, IndexDomainSerDesTest,
    testing::ValuesIn(test_util::AllSupportedSerDesVersions()));

}  // namespace
}  // namespace ifrt
}  // namespace xla
