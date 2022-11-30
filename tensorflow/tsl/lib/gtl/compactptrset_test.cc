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

#include "tensorflow/tsl/lib/gtl/compactptrset.h"

#include "tensorflow/tsl/platform/hash.h"
#include "tensorflow/tsl/platform/test.h"
#include "tensorflow/tsl/platform/types.h"

namespace tsl {
namespace gtl {
namespace {

typedef CompactPointerSet<const char*> StringSet;

static std::vector<const char*> SortedContents(const StringSet& set) {
  std::vector<const char*> contents(set.begin(), set.end());
  std::sort(contents.begin(), contents.end());
  return contents;
}

TEST(CompactPointerSetTest, Simple) {
  // Make some aligned and some unaligned pointers.
  string data = "ABCDEFG";
  const char* a = &data[0];
  const char* b = &data[1];
  const char* c = &data[2];
  const char* d = &data[3];
  const char* e = &data[4];
  const char* f = &data[5];
  const char* g = &data[6];
  for (const auto& list : std::vector<std::vector<const char*>>({{
           {},                     // Empty
           {a},                    // Aligned singleton
           {b},                    // Unaligned singleton
           {nullptr},              // Test insertion of nullptr
           {a, b, c, d, e, f, g},  // Many
       }})) {
    LOG(INFO) << list.size();

    // Test insert along with accessors.
    StringSet set;
    ASSERT_TRUE(set.empty());
    for (auto p : list) {
      ASSERT_EQ(set.count(p), 0);
      ASSERT_TRUE(set.insert(p).second);
      ASSERT_EQ(set.count(p), 1);
      ASSERT_TRUE(set.find(p) != set.end());
    }
    ASSERT_EQ(set.size(), list.size());

    ASSERT_EQ(SortedContents(set), list);

    // Test copy constructor.
    {
      StringSet set2(set);
      ASSERT_EQ(SortedContents(set2), list);
    }

    // Test assignment/copying into a destination with different
    // initial elements.
    for (const auto& initial : std::vector<std::vector<const char*>>({{
             {},            // Empty
             {a},           // Aligned singleton
             {b},           // Unaligned singleton
             {nullptr},     // Test insertion of nullptr
             {a, b, c, d},  // Many
         }})) {
      StringSet dst;
      for (auto p : initial) {
        dst.insert(p);
      }
      ASSERT_EQ(dst.size(), initial.size());
      dst = set;
      ASSERT_EQ(SortedContents(dst), list);
      dst.clear();
      ASSERT_EQ(dst.size(), 0);
    }

    // Test erase along with accessors.
    for (auto p : list) {
      ASSERT_EQ(set.erase(p), 1);
      ASSERT_EQ(set.erase(p), 0);
    }
    ASSERT_TRUE(set.empty());
    ASSERT_EQ(set.size(), 0);
  }
}

}  // namespace
}  // namespace gtl
}  // namespace tsl
