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

#include <memory>
#include <utility>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>

namespace xla {
namespace ifrt {
namespace {

using ::testing::ElementsAre;

TEST(IndexTest, Construction) {
  EXPECT_THAT(Index({1, 2}).elements(), ElementsAre(1, 2));
  EXPECT_THAT(Index::Zeros(2).elements(), ElementsAre(0, 0));
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

  EXPECT_EQ(a - b, Index({9, 19}));
  {
    Index c = a;
    EXPECT_EQ(c -= b, Index({9, 19}));
  }

  EXPECT_EQ(a * std::vector<int64_t>({1, 2}), Index({11, 44}));
  {
    Index c = a;
    EXPECT_EQ(c *= std::vector<int64_t>({1, 2}), Index({11, 44}));
  }
}

}  // namespace
}  // namespace ifrt
}  // namespace xla
