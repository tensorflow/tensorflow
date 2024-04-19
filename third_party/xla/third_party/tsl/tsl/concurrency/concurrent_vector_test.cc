/* Copyright 2022 Google LLC. All Rights Reserved.

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

#include "tsl/concurrency/concurrent_vector.h"

#include <algorithm>
#include <vector>

#include "tsl/platform/test.h"
#include "tsl/platform/threadpool.h"

namespace tsl {

using ::tsl::internal::ConcurrentVector;

TEST(ConcurrentVectorTest, SingleThreaded) {
  ConcurrentVector<int> vec(1);

  constexpr int kCount = 1000;

  for (int i = 0; i < kCount; ++i) {
    ASSERT_EQ(i, vec.emplace_back(i));
  }

  for (int i = 0; i < kCount; ++i) {
    EXPECT_EQ(i, vec[i]);
  }
}

TEST(ConcurrentVectorTest, OneWriterOneReader) {
  ConcurrentVector<int> vec(1);

  thread::ThreadPool pool(Env::Default(), "concurrent-vector", 4);
  constexpr int kCount = 1000;

  pool.Schedule([&] {
    for (int i = 0; i < kCount; ++i) {
      ASSERT_EQ(i, vec.emplace_back(i));
    }
  });

  pool.Schedule([&] {
    for (int i = 0; i < kCount; ++i) {
      while (i >= vec.size()) {
        // spin loop
      }
      EXPECT_EQ(i, vec[i]);
    }
  });
}

TEST(ConcurrentVectorTest, TwoWritersTwoReaders) {
  ConcurrentVector<int> vec(1);

  thread::ThreadPool pool(Env::Default(), "concurrent-vector", 4);
  constexpr int kCount = 1000;

  // Each writer stores from 0 to kCount/2 - 1 to the vector.
  auto writer = [&] {
    for (int i = 0; i < kCount / 2; ++i) {
      vec.emplace_back(i);
    }
  };

  pool.Schedule(writer);
  pool.Schedule(writer);

  // Reader reads all the data from the vector and verifies its content.
  auto reader = [&] {
    std::vector<int> stored;

    for (int i = 0; i < kCount; ++i) {
      while (i >= vec.size()) {
        // spin loop
      }
      stored.emplace_back(vec[i]);
    }

    std::sort(stored.begin(), stored.end());

    for (int i = 0; i < kCount / 2; ++i) {
      ASSERT_EQ(stored[2 * i], i);
      ASSERT_EQ(stored[2 * i + 1], i);
    }
  };

  pool.Schedule(reader);
  pool.Schedule(reader);
}

}  // namespace tsl
