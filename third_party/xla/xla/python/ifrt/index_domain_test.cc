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

#include <memory>
#include <utility>
#include <vector>

#include <gtest/gtest.h>
#include "absl/hash/hash_testing.h"
#include "xla/python/ifrt/index.h"
#include "xla/python/ifrt/shape.h"

namespace xla {
namespace ifrt {
namespace {

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

TEST(IndexDomainTest, Hash) {
  EXPECT_TRUE(absl::VerifyTypeImplementsAbslHashCorrectly(
      {IndexDomain(Index({1, 2}), Shape({3, 4})),
       IndexDomain(Index({1, 2}), Shape({4, 3})),
       IndexDomain(Index({2, 1}), Shape({3, 4})),
       IndexDomain(Index({2, 1}), Shape({4, 3}))}));
}

}  // namespace
}  // namespace ifrt
}  // namespace xla
