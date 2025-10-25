/* Copyright 2025 The OpenXLA Authors.

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

#include "xla/backends/cpu/ynn_threadpool.h"

#include <array>
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <vector>

#include <gtest/gtest.h>
#include "xla/tsl/platform/env.h"
#include "xla/tsl/platform/threadpool.h"

namespace Eigen {
class ThreadPoolInterface;
}  // namespace Eigen

namespace xla::cpu {

TEST(YnnThreadpool, inline_scheduling) {
  YnnThreadpool thread_pool(static_cast<Eigen::ThreadPoolInterface*>(nullptr));

  static constexpr size_t size = 10000;

  std::vector<int32_t> data(size, 0);
  auto inc = [&](size_t i) { data[i]++; };

  thread_pool.parallel_for(size, inc);

  std::vector<int32_t> expected(size, 1);
  EXPECT_EQ(data, expected);
}

TEST(YnnThreadpool, single_loop) {
  tsl::thread::ThreadPool test_thread_pool(tsl::Env::Default(), "test", 4);
  YnnThreadpool thread_pool(test_thread_pool.AsEigenThreadPool());

  static constexpr size_t size = 10000;

  std::vector<int32_t> data(size, 0);
  auto inc = [&](size_t i) { data[i]++; };

  thread_pool.parallel_for(size, inc);

  std::vector<int32_t> expected(size, 1);
  EXPECT_EQ(data, expected);
}

TEST(YnnThreadpool, loop_chain) {
  tsl::thread::ThreadPool test_thread_pool(tsl::Env::Default(), "test", 4);
  YnnThreadpool thread_pool(test_thread_pool.AsEigenThreadPool());

  static constexpr size_t size = 10000;

  std::vector<int32_t> data(size, 0);
  auto inc = [&](size_t i) { data[i]++; };

  thread_pool.parallel_for(size, inc);
  thread_pool.parallel_for(size, inc);
  thread_pool.parallel_for(size, inc);
  thread_pool.parallel_for(size, inc);
  thread_pool.parallel_for(size, inc);

  std::vector<int32_t> expected(size, 5);
  EXPECT_EQ(data, expected);
}

TEST(YnnThreadpool, nested_loops) {
  tsl::thread::ThreadPool test_thread_pool(tsl::Env::Default(), "test", 4);
  YnnThreadpool thread_pool(test_thread_pool.AsEigenThreadPool());

  static constexpr size_t size = 100;

  std::array<std::atomic<int32_t>, size> data = {{0}};
  auto inc = [&](size_t i) { data[i]++; };

  thread_pool.parallel_for(
      size, [&](size_t i) { thread_pool.parallel_for(size, inc); });

  for (size_t i = 0; i < size; ++i) {
    EXPECT_EQ(data[i], size);
  }
}

}  // namespace xla::cpu
