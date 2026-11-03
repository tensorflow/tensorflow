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

TEST(IndexDomainTest, Construction) {
  IndexDomain a(Index({1, 2}), Shape({3, 4}));
  EXPECT_EQ(a.origin(), Index({1, 2}));
  EXPECT_EQ(a.shape(), Shape({3, 4}));

  IndexDomain b(Shape({3, 4}));
  EXPECT_EQ(b.origin(), Index({0, 0}));
  EXPECT_EQ(b.shape(), Shape({3, 4}));
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

  // Disjoint / negative-origin domains.
  {
    IndexDomain a(Index({0, 0}), Shape({4, 4}));
    IndexDomain b(Index({-2, 0}), Shape({2, 2}));
    EXPECT_THAT(a.Contains(b), IsOkAndHolds(false));

    IndexDomain c(Index({5, 5}), Shape({2, 2}));
    EXPECT_THAT(a.Contains(c), IsOkAndHolds(false));
  }

  // Zero-sized domain.
  {
    IndexDomain a(Shape({4, 4}));
    IndexDomain zero(Shape({0, 0}));
    EXPECT_THAT(a.Contains(zero), IsOkAndHolds(true));
  }

  // Negative shape dimensions.
  {
    IndexDomain a(Index({0, 0}), Shape({-1, 4}));
    IndexDomain b(Index({0, 0}), Shape({2, 4}));
    EXPECT_THAT(a.Contains(b), StatusIs(absl::StatusCode::kInvalidArgument));
    EXPECT_THAT(b.Contains(a), StatusIs(absl::StatusCode::kInvalidArgument));
  }

  // Mismatched rank.
  {
    IndexDomain a(Index({1, 2}), Shape({3, 4}));
    IndexDomain b(Index({1}), Shape({3}));
    EXPECT_THAT(a.Contains(b), StatusIs(absl::StatusCode::kInvalidArgument));
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

INSTANTIATE_TEST_SUITE_P(
    SerDesVersion, IndexDomainSerDesTest,
    testing::ValuesIn(test_util::AllSupportedSerDesVersions()));

}  // namespace
}  // namespace ifrt
}  // namespace xla
