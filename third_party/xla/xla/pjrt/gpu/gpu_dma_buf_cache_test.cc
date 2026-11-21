/* Copyright 2026 The OpenXLA Authors.

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

#include "xla/pjrt/gpu/gpu_dma_buf_cache.h"

#include <fcntl.h>
#include <unistd.h>

#include <atomic>
#include <cerrno>
#include <cstdint>
#include <memory>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/log/check.h"
#include "absl/status/status.h"
#include "absl/status/status_matchers.h"
#include "absl/status/statusor.h"
#include "absl/synchronization/notification.h"
#include "xla/stream_executor/device_address.h"
#include "xla/stream_executor/gpu/dma_buf.h"
#include "xla/tsl/platform/env.h"
#include "xla/tsl/platform/threadpool.h"

namespace xla {
namespace {

using ::absl_testing::StatusIs;

// Creates a real OS file descriptor (pipe read end) and closes the write end
// so that `fcntl(fd, F_GETFD)` can verify when `DmaBuf` closes it.
int MakeTestFd() {
  int fds[2] = {-1, -1};
  CHECK_EQ(::pipe(fds), 0);
  CHECK_EQ(::close(fds[1]), 0);
  return fds[0];
}

bool IsFdOpen(int fd) { return ::fcntl(fd, F_GETFD) != -1; }

void* FakeAddr(uintptr_t addr) { return reinterpret_cast<void*>(addr); }

TEST(GpuDmaBufCacheTest, CachesPerChunkAndComputesOffsets) {
  auto cache = std::make_shared<GpuDmaBufCache>();
  cache->alloc_visitor()(FakeAddr(0x10000), /*device_ordinal=*/0, 0x4000);

  int export_count = 0;
  stream_executor::DeviceAddressBase exported_range;
  auto export_fn = [&](const stream_executor::DeviceAddressBase& range)
      -> absl::StatusOr<stream_executor::gpu::DmaBuf> {
    ++export_count;
    exported_range = range;
    return stream_executor::gpu::DmaBuf(MakeTestFd(), /*offset=*/128);
  };

  ASSERT_OK_AND_ASSIGN(GpuDmaBufCache::ExportedDmaBuf first,
                       cache->GetOrExport(/*device_ordinal=*/0,
                                          stream_executor::DeviceAddressBase(
                                              FakeAddr(0x10000), 0x1000),
                                          export_fn));
  ASSERT_OK_AND_ASSIGN(GpuDmaBufCache::ExportedDmaBuf second,
                       cache->GetOrExport(/*device_ordinal=*/0,
                                          stream_executor::DeviceAddressBase(
                                              FakeAddr(0x10200), 0x400),
                                          export_fn));

  EXPECT_EQ(export_count, 1);
  EXPECT_EQ(exported_range.opaque(), FakeAddr(0x10000));
  EXPECT_EQ(exported_range.size(), 0x4000);
  EXPECT_EQ(first.dma_buf->fd(), second.dma_buf->fd());
  EXPECT_EQ(first.offset, 128);
  EXPECT_EQ(second.offset, 128 + 0x200);
  EXPECT_TRUE(IsFdOpen(first.dma_buf->fd()));
}

TEST(GpuDmaBufCacheTest, FreeVisitorClosesCachedFdWhenNoRefsRemain) {
  auto cache = std::make_shared<GpuDmaBufCache>();
  cache->alloc_visitor()(FakeAddr(0x20000), /*device_ordinal=*/0, 0x4000);

  int fd = -1;
  {
    ASSERT_OK_AND_ASSIGN(
        GpuDmaBufCache::ExportedDmaBuf entry,
        cache->GetOrExport(
            /*device_ordinal=*/0,
            stream_executor::DeviceAddressBase(FakeAddr(0x20000), 0x1000),
            [&](const stream_executor::DeviceAddressBase&)
                -> absl::StatusOr<stream_executor::gpu::DmaBuf> {
              fd = MakeTestFd();
              return stream_executor::gpu::DmaBuf(fd, /*offset=*/0);
            }));
    EXPECT_TRUE(IsFdOpen(fd));
  }
  // Cache still holds the fd after the caller drops its reference.
  EXPECT_TRUE(IsFdOpen(fd));

  cache->free_visitor()(FakeAddr(0x20000), /*device_ordinal=*/0, 0x4000);
  EXPECT_FALSE(IsFdOpen(fd));
  EXPECT_EQ(errno, EBADF);
}

TEST(GpuDmaBufCacheTest, ActiveReferenceKeepsFdAlivePastFreeVisitor) {
  auto cache = std::make_shared<GpuDmaBufCache>();
  cache->alloc_visitor()(FakeAddr(0x30000), /*device_ordinal=*/0, 0x4000);

  ASSERT_OK_AND_ASSIGN(
      GpuDmaBufCache::ExportedDmaBuf entry,
      cache->GetOrExport(
          /*device_ordinal=*/0,
          stream_executor::DeviceAddressBase(FakeAddr(0x30000), 0x1000),
          [](const stream_executor::DeviceAddressBase&)
              -> absl::StatusOr<stream_executor::gpu::DmaBuf> {
            return stream_executor::gpu::DmaBuf(MakeTestFd(), /*offset=*/0);
          }));
  const int fd = entry.dma_buf->fd();
  ASSERT_TRUE(IsFdOpen(fd));

  cache->free_visitor()(FakeAddr(0x30000), /*device_ordinal=*/0, 0x4000);
  EXPECT_TRUE(IsFdOpen(fd));

  entry.dma_buf.reset();
  EXPECT_FALSE(IsFdOpen(fd));
}

TEST(GpuDmaBufCacheTest, UntrackedAndMultiChunkRangesUseUncachedFallback) {
  auto cache = std::make_shared<GpuDmaBufCache>();
  cache->alloc_visitor()(FakeAddr(0x40000), /*device_ordinal=*/0, 0x2000);
  cache->alloc_visitor()(FakeAddr(0x42000), /*device_ordinal=*/0, 0x2000);

  int export_count = 0;
  auto export_fn = [&](const stream_executor::DeviceAddressBase&)
      -> absl::StatusOr<stream_executor::gpu::DmaBuf> {
    ++export_count;
    return stream_executor::gpu::DmaBuf(MakeTestFd(), /*offset=*/64);
  };

  // Range straddling two adjacent chunks falls back to uncached export.
  int straddle_fd = -1;
  {
    ASSERT_OK_AND_ASSIGN(GpuDmaBufCache::ExportedDmaBuf straddle,
                         cache->GetOrExport(/*device_ordinal=*/0,
                                            stream_executor::DeviceAddressBase(
                                                FakeAddr(0x41000), 0x2000),
                                            export_fn));
    straddle_fd = straddle.dma_buf->fd();
    EXPECT_EQ(straddle.offset, 64);
    EXPECT_TRUE(IsFdOpen(straddle_fd));
  }
  EXPECT_FALSE(IsFdOpen(straddle_fd));

  // Range starting before a tracked chunk on the same device falls back.
  int before_fd = -1;
  {
    ASSERT_OK_AND_ASSIGN(GpuDmaBufCache::ExportedDmaBuf before,
                         cache->GetOrExport(/*device_ordinal=*/0,
                                            stream_executor::DeviceAddressBase(
                                                FakeAddr(0x3f000), 0x1000),
                                            export_fn));
    before_fd = before.dma_buf->fd();
    EXPECT_TRUE(IsFdOpen(before_fd));
  }
  EXPECT_FALSE(IsFdOpen(before_fd));

  // Completely untracked range also uses uncached fallback.
  int untracked_fd = -1;
  {
    ASSERT_OK_AND_ASSIGN(GpuDmaBufCache::ExportedDmaBuf untracked,
                         cache->GetOrExport(/*device_ordinal=*/1,
                                            stream_executor::DeviceAddressBase(
                                                FakeAddr(0x90000), 0x1000),
                                            export_fn));
    untracked_fd = untracked.dma_buf->fd();
    EXPECT_TRUE(IsFdOpen(untracked_fd));
  }
  EXPECT_FALSE(IsFdOpen(untracked_fd));
  EXPECT_EQ(export_count, 3);
}

TEST(GpuDmaBufCacheTest, EmptyRangeDoesNotMatchCachedChunk) {
  auto cache = std::make_shared<GpuDmaBufCache>();
  cache->alloc_visitor()(FakeAddr(0x48000), /*device_ordinal=*/0, 0x2000);

  stream_executor::DeviceAddressBase last_range;
  auto export_fn = [&](const stream_executor::DeviceAddressBase& range)
      -> absl::StatusOr<stream_executor::gpu::DmaBuf> {
    last_range = range;
    return absl::InvalidArgumentError("empty range");
  };

  EXPECT_THAT(
      cache->GetOrExport(/*device_ordinal=*/0,
                         stream_executor::DeviceAddressBase(FakeAddr(0x48000),
                                                            /*size=*/0),
                         export_fn),
      StatusIs(absl::StatusCode::kInvalidArgument));
  EXPECT_EQ(last_range.size(), 0);
}

TEST(GpuDmaBufCacheTest, ExportErrorsAreNotCached) {
  auto cache = std::make_shared<GpuDmaBufCache>();
  cache->alloc_visitor()(FakeAddr(0x50000), /*device_ordinal=*/0, 0x4000);

  int calls = 0;
  auto export_fn = [&](const stream_executor::DeviceAddressBase&)
      -> absl::StatusOr<stream_executor::gpu::DmaBuf> {
    if (++calls == 1) {
      return absl::UnavailableError("transient failure");
    }
    return stream_executor::gpu::DmaBuf(MakeTestFd(), /*offset=*/0);
  };

  EXPECT_THAT(cache->GetOrExport(
                  /*device_ordinal=*/0,
                  stream_executor::DeviceAddressBase(FakeAddr(0x50000), 0x1000),
                  export_fn),
              StatusIs(absl::StatusCode::kUnavailable));
  ASSERT_OK_AND_ASSIGN(GpuDmaBufCache::ExportedDmaBuf ok,
                       cache->GetOrExport(/*device_ordinal=*/0,
                                          stream_executor::DeviceAddressBase(
                                              FakeAddr(0x50000), 0x1000),
                                          export_fn));
  EXPECT_EQ(calls, 2);
  EXPECT_TRUE(IsFdOpen(ok.dma_buf->fd()));
}

TEST(GpuDmaBufCacheTest, ConcurrentFirstExportDeduplicatesAndClosesExtraFd) {
  auto cache = std::make_shared<GpuDmaBufCache>();
  cache->alloc_visitor()(FakeAddr(0x60000), /*device_ordinal=*/0, 0x4000);

  absl::Notification both_entered;
  absl::Notification release_exports;
  std::atomic<int> entered_count{0};
  std::atomic<int> fds[2] = {-1, -1};

  auto export_fn = [&](const stream_executor::DeviceAddressBase&)
      -> absl::StatusOr<stream_executor::gpu::DmaBuf> {
    const int idx = entered_count.fetch_add(1, std::memory_order_relaxed);
    const int fd = MakeTestFd();
    fds[idx].store(fd, std::memory_order_relaxed);
    if (idx == 1) {
      both_entered.Notify();
    }
    release_exports.WaitForNotification();
    return stream_executor::gpu::DmaBuf(fd, /*offset=*/0);
  };

  tsl::thread::ThreadPool pool(tsl::Env::Default(), "dma_buf_cache_test", 2);
  absl::StatusOr<GpuDmaBufCache::ExportedDmaBuf> res1;
  absl::StatusOr<GpuDmaBufCache::ExportedDmaBuf> res2;
  absl::Notification done1;
  absl::Notification done2;

  pool.Schedule([&] {
    res1 = cache->GetOrExport(
        /*device_ordinal=*/0,
        stream_executor::DeviceAddressBase(FakeAddr(0x60000), 0x1000),
        export_fn);
    done1.Notify();
  });
  pool.Schedule([&] {
    res2 = cache->GetOrExport(
        /*device_ordinal=*/0,
        stream_executor::DeviceAddressBase(FakeAddr(0x60100), 0x1000),
        export_fn);
    done2.Notify();
  });

  both_entered.WaitForNotification();
  release_exports.Notify();
  done1.WaitForNotification();
  done2.WaitForNotification();

  ASSERT_OK(res1);
  ASSERT_OK(res2);
  EXPECT_EQ(res1->dma_buf->fd(), res2->dma_buf->fd());
  const int winning_fd = res1->dma_buf->fd();
  const int losing_fd = (fds[0].load(std::memory_order_relaxed) == winning_fd)
                            ? fds[1].load(std::memory_order_relaxed)
                            : fds[0].load(std::memory_order_relaxed);
  EXPECT_TRUE(IsFdOpen(winning_fd));
  EXPECT_FALSE(IsFdOpen(losing_fd));
}

TEST(GpuDmaBufCacheTest, ConcurrentFreeDuringExportFallsBackToUncached) {
  auto cache = std::make_shared<GpuDmaBufCache>();
  cache->alloc_visitor()(FakeAddr(0x70000), /*device_ordinal=*/0, 0x4000);

  int chunk_fd = -1;
  int call_count = 0;
  auto export_fn = [&](const stream_executor::DeviceAddressBase& range)
      -> absl::StatusOr<stream_executor::gpu::DmaBuf> {
    if (++call_count == 1) {
      // Simulate chunk being freed while the chunk-wide export is in flight.
      chunk_fd = MakeTestFd();
      cache->free_visitor()(FakeAddr(0x70000), /*device_ordinal=*/0, 0x4000);
      return stream_executor::gpu::DmaBuf(chunk_fd, /*offset=*/0);
    }
    // The discarded chunk-wide fd must be closed before the fallback export
    // (checking here avoids POSIX fd-number reuse by `MakeTestFd()`).
    EXPECT_FALSE(IsFdOpen(chunk_fd));
    EXPECT_EQ(range.opaque(), FakeAddr(0x70100));
    EXPECT_EQ(range.size(), 0x1000);
    return stream_executor::gpu::DmaBuf(MakeTestFd(), /*offset=*/32);
  };

  ASSERT_OK_AND_ASSIGN(GpuDmaBufCache::ExportedDmaBuf entry,
                       cache->GetOrExport(/*device_ordinal=*/0,
                                          stream_executor::DeviceAddressBase(
                                              FakeAddr(0x70100), 0x1000),
                                          export_fn));
  EXPECT_EQ(call_count, 2);
  EXPECT_TRUE(IsFdOpen(entry.dma_buf->fd()));
  EXPECT_EQ(entry.offset, 32);
}

}  // namespace
}  // namespace xla
