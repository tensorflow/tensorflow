/* Copyright 2019 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/compiler/tf2tensorrt/utils/trt_lru_cache.h"

#include "tensorflow/core/platform/test.h"

namespace tensorflow {
namespace tensorrt {

TEST(LRUCacheTest, Basic) {
  LRUCache<int, int, std::hash<int>> cache;
  cache.reserve(2);
  // Insert 10
  cache.emplace(10, 100);
  EXPECT_EQ(cache.size(), 1);
  EXPECT_EQ(cache.count(10), 1);
  EXPECT_EQ(cache.at(10), 100);
  EXPECT_EQ(cache.count(100), 0);
  // Insert 20
  cache.emplace(20, 200);
  EXPECT_EQ(cache.size(), 2);
  EXPECT_EQ(cache.count(10), 1);
  EXPECT_EQ(cache.count(20), 1);
  EXPECT_EQ(cache.at(10), 100);
  EXPECT_EQ(cache.at(20), 200);
  EXPECT_EQ(cache.count(100), 0);
  EXPECT_EQ(cache.count(200), 0);
  // Insert 30, Evicting 10
  cache.emplace(30, 300);
  EXPECT_EQ(cache.count(10), 0);
  EXPECT_EQ(cache.count(20), 1);
  EXPECT_EQ(cache.count(30), 1);
  // Touch 20
  cache.at(20);
  // Insert 40, Evicting 30
  cache.emplace(40, 400);
  EXPECT_EQ(cache.count(10), 0);
  EXPECT_EQ(cache.count(20), 1);
  EXPECT_EQ(cache.count(30), 0);
  EXPECT_EQ(cache.count(40), 1);
}

}  // namespace tensorrt
}  // namespace tensorflow
