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

#include "xla/backends/cpu/runtime/ynnpack/ynn_threadpool.h"

#include <array>
#include <atomic>
#include <cstddef>
#include <cstdint>

#include <gtest/gtest.h>
#include "slinky/base/thread_pool.h"
#include "xla/tsl/platform/env.h"
#include "xla/tsl/platform/threadpool.h"

namespace Eigen {
class ThreadPoolInterface;
}  // namespace Eigen

namespace xla::cpu {

TEST(YnnThreadpoolImpl, inline_scheduling) {
  auto ynn_threadpool =
      CreateYnnThreadpool(static_cast<Eigen::ThreadPoolInterface*>(nullptr));
  auto thread_pool =
      reinterpret_cast<slinky::thread_pool*>(ynn_threadpool->get());

  static constexpr size_t size = 10000;

  std::vector<int32_t> data(size, 0);
  auto inc = [&](size_t i) { data[i]++; };

  thread_pool->parallel_for(size, inc);

  std::vector<int32_t> expected(size, 1);
  EXPECT_EQ(data, expected);
}

TEST(YnnThreadpoolImpl, single_loop) {
  tsl::thread::ThreadPool test_thread_pool(tsl::Env::Default(), "test", 4);
  auto ynn_threadpool =
      CreateYnnThreadpool(test_thread_pool.AsEigenThreadPool());
  auto thread_pool =
      reinterpret_cast<slinky::thread_pool*>(ynn_threadpool->get());

  static constexpr size_t size = 10000;

  std::vector<int32_t> data(size, 0);
  auto inc = [&](size_t i) { data[i]++; };

  thread_pool->parallel_for(size, inc);

  std::vector<int32_t> expected(size, 1);
  EXPECT_EQ(data, expected);
}

TEST(YnnThreadpoolImpl, loop_chain) {
  tsl::thread::ThreadPool test_thread_pool(tsl::Env::Default(), "test", 4);
  auto ynn_threadpool =
      CreateYnnThreadpool(test_thread_pool.AsEigenThreadPool());
  auto thread_pool =
      reinterpret_cast<slinky::thread_pool*>(ynn_threadpool->get());

  static constexpr size_t size = 10000;

  std::vector<int32_t> data(size, 0);
  auto inc = [&](size_t i) { data[i]++; };

  thread_pool->parallel_for(size, inc);
  thread_pool->parallel_for(size, inc);
  thread_pool->parallel_for(size, inc);
  thread_pool->parallel_for(size, inc);
  thread_pool->parallel_for(size, inc);

  std::vector<int32_t> expected(size, 5);
  EXPECT_EQ(data, expected);
}

TEST(YnnThreadpoolImpl, nested_loops) {
  tsl::thread::ThreadPool test_thread_pool(tsl::Env::Default(), "test", 4);
  auto ynn_threadpool =
      CreateYnnThreadpool(test_thread_pool.AsEigenThreadPool());
  auto thread_pool =
      reinterpret_cast<slinky::thread_pool*>(ynn_threadpool->get());

  static constexpr size_t size = 100;

  std::array<std::atomic<int32_t>, size> data = {{0}};
  auto inc = [&](size_t i) { data[i]++; };

  thread_pool->parallel_for(
      size, [&](size_t i) { thread_pool->parallel_for(size, inc); });

  for (size_t i = 0; i < size; ++i) {
    EXPECT_EQ(data[i], size);
  }
}

}  // namespace xla::cpu
