/* Copyright 2015 The TensorFlow Authors. All Rights Reserved.

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

#include "tsl/lib/gtl/iterator_range.h"

#include <vector>

#include "tsl/platform/macros.h"
#include "tsl/platform/test.h"
#include "tsl/platform/types.h"

namespace tsl {
namespace gtl {
namespace {

TEST(IteratorRange, WholeVector) {
  std::vector<int> v = {2, 3, 5, 7, 11, 13};
  iterator_range<std::vector<int>::iterator> range(v.begin(), v.end());
  int index = 0;
  for (int prime : range) {
    ASSERT_LT(index, v.size());
    EXPECT_EQ(v[index], prime);
    ++index;
  }
  EXPECT_EQ(v.size(), index);
}

TEST(IteratorRange, VectorMakeRange) {
  std::vector<int> v = {2, 3, 5, 7, 11, 13};
  auto range = make_range(v.begin(), v.end());
  int index = 0;
  for (int prime : range) {
    ASSERT_LT(index, v.size());
    EXPECT_EQ(v[index], prime);
    ++index;
  }
  EXPECT_EQ(v.size(), index);
}

TEST(IteratorRange, PartArray) {
  int v[] = {2, 3, 5, 7, 11, 13};
  iterator_range<int*> range(&v[1], &v[4]);  // 3, 5, 7
  int index = 1;
  for (int prime : range) {
    ASSERT_LT(index, TF_ARRAYSIZE(v));
    EXPECT_EQ(v[index], prime);
    ++index;
  }
  EXPECT_EQ(4, index);
}

TEST(IteratorRange, ArrayMakeRange) {
  int v[] = {2, 3, 5, 7, 11, 13};
  auto range = make_range(&v[1], &v[4]);  // 3, 5, 7
  int index = 1;
  for (int prime : range) {
    ASSERT_LT(index, TF_ARRAYSIZE(v));
    EXPECT_EQ(v[index], prime);
    ++index;
  }
  EXPECT_EQ(4, index);
}
}  // namespace
}  // namespace gtl
}  // namespace tsl
