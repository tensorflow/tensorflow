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

#include "xla/backends/gpu/runtime/lock_free_kernel_cache.h"

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <vector>

#include <gtest/gtest.h>
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "xla/stream_executor/kernel.h"
#include "xla/stream_executor/kernel_args.h"
#include "xla/stream_executor/launch_dim.h"
#include "xla/stream_executor/stream.h"
#include "xla/stream_executor/stream_executor.h"
#include "xla/tsl/platform/env.h"
#include "xla/tsl/platform/test.h"
#include "xla/tsl/platform/threadpool.h"

namespace xla::gpu {
namespace {

struct DummyKernel : public se::Kernel {
  explicit DummyKernel(int id) : id(id) {}
  int id;

  unsigned Arity() const override { return 0; }
  absl::StatusOr<int32_t> GetMaxOccupiedBlocksPerCore(se::ThreadDim,
                                                      size_t) const override {
    return 1;
  }
  absl::Status Launch(const se::ThreadDim&, const se::BlockDim&,
                      const std::optional<se::ClusterDim>&, se::Stream*,
                      const se::KernelArgs&) override {
    return absl::OkStatus();
  }
};

TEST(LockFreeKernelCacheTest, EmptyCache) {
  LockFreeKernelCache cache;
  EXPECT_TRUE(cache.empty());
  EXPECT_EQ(cache.size(), 0);

  auto* exec1 = reinterpret_cast<se::StreamExecutor*>(0x1000);
  EXPECT_EQ(cache.Find(exec1), nullptr);
  EXPECT_FALSE(cache.Contains(exec1));
}

TEST(LockFreeKernelCacheTest, SingleExecutorGetOrCreateAndFind) {
  LockFreeKernelCache cache;
  auto* exec1 = reinterpret_cast<se::StreamExecutor*>(0x1000);

  int load_count = 0;
  auto loader = [&]() -> absl::StatusOr<std::unique_ptr<se::Kernel>> {
    ++load_count;
    return std::make_unique<DummyKernel>(42);
  };

  ASSERT_OK_AND_ASSIGN(se::Kernel * kernel1, cache.GetOrCreate(exec1, loader));
  ASSERT_NE(kernel1, nullptr);
  EXPECT_EQ(static_cast<DummyKernel*>(kernel1)->id, 42);
  EXPECT_EQ(load_count, 1);
  EXPECT_EQ(cache.size(), 1);
  EXPECT_FALSE(cache.empty());

  // Second lookup via Find
  EXPECT_EQ(cache.Find(exec1), kernel1);
  EXPECT_TRUE(cache.Contains(exec1));

  // Second GetOrCreate should return cached kernel and not invoke loader
  ASSERT_OK_AND_ASSIGN(se::Kernel * cached_kernel,
                       cache.GetOrCreate(exec1, loader));
  EXPECT_EQ(cached_kernel, kernel1);
  EXPECT_EQ(load_count, 1);
}

TEST(LockFreeKernelCacheTest, MultipleExecutorsBeyondInlineCapacity) {
  LockFreeKernelCache cache;
  constexpr int kNumExecutors = 16;
  std::vector<se::Kernel*> loaded_kernels;

  for (uintptr_t i = 1; i <= kNumExecutors; ++i) {
    auto* exec = reinterpret_cast<se::StreamExecutor*>(i * 0x1000);
    ASSERT_OK_AND_ASSIGN(
        se::Kernel * kernel,
        cache.GetOrCreate(
            exec, [i]() -> absl::StatusOr<std::unique_ptr<se::Kernel>> {
              return std::make_unique<DummyKernel>(static_cast<int>(i));
            }));
    loaded_kernels.push_back(kernel);
  }

  EXPECT_EQ(cache.size(), kNumExecutors);

  // Verify all are present and correct via Find
  for (uintptr_t i = 1; i <= kNumExecutors; ++i) {
    auto* exec = reinterpret_cast<se::StreamExecutor*>(i * 0x1000);
    se::Kernel* k = cache.Find(exec);
    ASSERT_NE(k, nullptr);
    EXPECT_EQ(k, loaded_kernels[i - 1]);
    EXPECT_EQ(static_cast<DummyKernel*>(k)->id, static_cast<int>(i));
  }
}

TEST(LockFreeKernelCacheTest, LoaderFailureDoesNotCorruptCache) {
  LockFreeKernelCache cache;
  auto* exec1 = reinterpret_cast<se::StreamExecutor*>(0x1000);

  auto failing_loader = []() -> absl::StatusOr<std::unique_ptr<se::Kernel>> {
    return absl::InternalError("Load failed");
  };

  auto result = cache.GetOrCreate(exec1, failing_loader);
  EXPECT_FALSE(result.ok());
  EXPECT_EQ(cache.size(), 0);
  EXPECT_EQ(cache.Find(exec1), nullptr);

  // Subsequent successful load succeeds
  auto succeeding_loader = []() -> absl::StatusOr<std::unique_ptr<se::Kernel>> {
    return std::make_unique<DummyKernel>(100);
  };
  ASSERT_OK_AND_ASSIGN(se::Kernel * good_kernel,
                       cache.GetOrCreate(exec1, succeeding_loader));
  EXPECT_EQ(cache.size(), 1);
  EXPECT_EQ(cache.Find(exec1), good_kernel);
}

TEST(LockFreeKernelCacheTest, ConcurrentReadsAndWrites) {
  LockFreeKernelCache cache;
  constexpr int kNumExecutors = 8;
  std::atomic<bool> start{false};
  std::atomic<bool> done{false};

  {
    // Launch reader threads
    tsl::thread::ThreadPool reader_pool(tsl::Env::Default(), "readers", 4);
    for (int t = 0; t < 4; ++t) {
      reader_pool.Schedule([&]() {
        while (!start.load(std::memory_order_acquire)) {
        }
        while (!done.load(std::memory_order_relaxed)) {
          for (uintptr_t i = 1; i <= kNumExecutors; ++i) {
            auto* exec = reinterpret_cast<se::StreamExecutor*>(i * 0x1000);
            se::Kernel* k = cache.Find(exec);
            if (k != nullptr) {
              EXPECT_EQ(static_cast<DummyKernel*>(k)->id, static_cast<int>(i));
            }
          }
        }
      });
    }

    // Launch writer threads that initialize the executors
    {
      tsl::thread::ThreadPool writer_pool(tsl::Env::Default(), "writers",
                                          kNumExecutors);
      for (uintptr_t i = 1; i <= kNumExecutors; ++i) {
        writer_pool.Schedule([&, i]() {
          while (!start.load(std::memory_order_acquire)) {
          }
          auto* exec = reinterpret_cast<se::StreamExecutor*>(i * 0x1000);
          auto res = cache.GetOrCreate(
              exec, [i]() -> absl::StatusOr<std::unique_ptr<se::Kernel>> {
                return std::make_unique<DummyKernel>(static_cast<int>(i));
              });
          EXPECT_TRUE(res.ok());
        });
      }

      start.store(true, std::memory_order_release);
      // writer_pool destructor waits for all writers to complete.
    }

    done.store(true, std::memory_order_relaxed);
    // reader_pool destructor waits for all readers to complete.
  }

  EXPECT_EQ(cache.size(), kNumExecutors);
}

TEST(LockFreeKernelCacheTest, ConcurrentReadsAndWritesBeyondInlineCapacity) {
  LockFreeKernelCache cache;
  constexpr int kNumExecutors = 16;
  std::atomic<bool> start{false};
  std::atomic<bool> done{false};

  {
    // Launch reader threads
    tsl::thread::ThreadPool reader_pool(tsl::Env::Default(), "readers", 4);
    for (int t = 0; t < 4; ++t) {
      reader_pool.Schedule([&]() {
        while (!start.load(std::memory_order_acquire)) {
        }
        while (!done.load(std::memory_order_relaxed)) {
          for (uintptr_t i = 1; i <= kNumExecutors; ++i) {
            auto* exec = reinterpret_cast<se::StreamExecutor*>(i * 0x1000);
            se::Kernel* k = cache.Find(exec);
            if (k != nullptr) {
              EXPECT_EQ(static_cast<DummyKernel*>(k)->id, static_cast<int>(i));
            }
          }
        }
      });
    }

    // Launch writer threads that initialize the executors beyond inline
    // capacity
    {
      tsl::thread::ThreadPool writer_pool(tsl::Env::Default(), "writers",
                                          kNumExecutors);
      for (uintptr_t i = 1; i <= kNumExecutors; ++i) {
        writer_pool.Schedule([&, i]() {
          while (!start.load(std::memory_order_acquire)) {
          }
          auto* exec = reinterpret_cast<se::StreamExecutor*>(i * 0x1000);
          auto res = cache.GetOrCreate(
              exec, [i]() -> absl::StatusOr<std::unique_ptr<se::Kernel>> {
                return std::make_unique<DummyKernel>(static_cast<int>(i));
              });
          EXPECT_TRUE(res.ok());
        });
      }

      start.store(true, std::memory_order_release);
      // writer_pool destructor waits for all writers to complete.
    }

    done.store(true, std::memory_order_relaxed);
    // reader_pool destructor waits for all readers to complete.
  }

  EXPECT_EQ(cache.size(), kNumExecutors);
}

TEST(LockFreeKernelCacheTest, NullExecutorHandling) {
  LockFreeKernelCache cache;
  EXPECT_EQ(cache.Find(nullptr), nullptr);
  EXPECT_FALSE(cache.Contains(nullptr));
  EXPECT_FALSE(cache.contains(nullptr));

  auto loader = []() -> absl::StatusOr<std::unique_ptr<se::Kernel>> {
    return std::make_unique<DummyKernel>(1);
  };
  auto result = cache.GetOrCreate(nullptr, loader);
  EXPECT_FALSE(result.ok());
  EXPECT_EQ(result.status().code(), absl::StatusCode::kInvalidArgument);
  EXPECT_EQ(cache.size(), 0);
  EXPECT_TRUE(cache.empty());
}

TEST(LockFreeKernelCacheTest, ConcurrentSameExecutorContention) {
  LockFreeKernelCache cache;
  auto* exec1 = reinterpret_cast<se::StreamExecutor*>(0x1000);
  constexpr int kNumThreads = 10;
  std::atomic<int> load_count{0};
  std::atomic<bool> start{false};

  std::vector<se::Kernel*> returned_kernels(kNumThreads, nullptr);
  {
    tsl::thread::ThreadPool pool(tsl::Env::Default(), "contention_test",
                                 kNumThreads);
    for (int i = 0; i < kNumThreads; ++i) {
      pool.Schedule([&, i]() {
        while (!start.load(std::memory_order_acquire)) {
        }
        ASSERT_OK_AND_ASSIGN(
            se::Kernel * kernel,
            cache.GetOrCreate(
                exec1, [&]() -> absl::StatusOr<std::unique_ptr<se::Kernel>> {
                  load_count.fetch_add(1, std::memory_order_relaxed);
                  return std::make_unique<DummyKernel>(999);
                }));
        returned_kernels[i] = kernel;
      });
    }

    start.store(true, std::memory_order_release);
    // pool destructor waits for all threads to complete.
  }

  EXPECT_EQ(load_count.load(), 1);
  EXPECT_EQ(cache.size(), 1);
  ASSERT_NE(returned_kernels[0], nullptr);
  for (int i = 1; i < kNumThreads; ++i) {
    EXPECT_EQ(returned_kernels[i], returned_kernels[0]);
  }
}

TEST(LockFreeKernelCacheTest, ContainsAndContainsAlias) {
  LockFreeKernelCache cache;
  auto* exec1 = reinterpret_cast<se::StreamExecutor*>(0x1000);
  EXPECT_FALSE(cache.Contains(exec1));
  EXPECT_FALSE(cache.contains(exec1));

  auto res = cache.GetOrCreate(
      exec1, []() -> absl::StatusOr<std::unique_ptr<se::Kernel>> {
        return std::make_unique<DummyKernel>(1);
      });
  ASSERT_TRUE(res.ok());
  EXPECT_TRUE(cache.Contains(exec1));
  EXPECT_TRUE(cache.contains(exec1));
}

}  // namespace
}  // namespace xla::gpu
