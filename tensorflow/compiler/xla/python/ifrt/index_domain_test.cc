/* Copyright 2022 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/compiler/xla/python/ifrt/index_domain.h"

#include <memory>
#include <utility>
#include <vector>

#include <gtest/gtest.h>

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

}  // namespace
}  // namespace ifrt
}  // namespace xla
