/* Copyright 2017 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/compiler/xla/iterator_util.h"

#include <algorithm>
#include <list>

#include "tensorflow/compiler/xla/ptr_util.h"
#include "tensorflow/compiler/xla/test.h"

namespace xla {
namespace {

TEST(UnwrappingIteratorTest, Simple) {
  std::vector<std::unique_ptr<int>> v;
  for (int i = 0; i < 3; ++i) {
    v.push_back(MakeUnique<int>(i));
  }
  int i = 0;
  for (auto iter = MakeUnwrappingIterator(v.begin());
       iter != MakeUnwrappingIterator(v.end()); ++iter) {
    EXPECT_EQ(*iter, v[i].get());
    ++i;
  }
}

TEST(UnwrappingIteratorTest, PostincrementOperator) {
  std::vector<std::shared_ptr<int>> v;
  for (int i = 0; i < 3; ++i) {
    v.push_back(std::make_shared<int>(i));
  }
  auto iter = MakeUnwrappingIterator(v.begin());
  EXPECT_EQ(*(iter++), v[0].get());
  EXPECT_EQ(*iter, v[1].get());
}

// std::find relies on various iterator traits being properly defined.
TEST(UnwrappingIteratorTest, StdFind) {
  std::list<std::unique_ptr<int>> l;
  for (int i = 0; i < 3; ++i) {
    l.push_back(MakeUnique<int>(i));
  }
  EXPECT_EQ(l.begin()->get(),
            *std::find(MakeUnwrappingIterator(l.begin()),
                       MakeUnwrappingIterator(l.end()), l.begin()->get()));
}

}  // namespace
}  // namespace xla
