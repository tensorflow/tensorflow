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

#include "xla/backends/gpu/runtime/collective_memory_cache.h"

#include <cstdint>
#include <memory>
#include <string>
#include <utility>

#include <gtest/gtest.h>
#include "absl/synchronization/notification.h"
#include "xla/backends/gpu/collectives/gpu_clique_key.h"
#include "xla/core/collectives/symmetric_memory.h"
#include "xla/runtime/device_id.h"
#include "xla/stream_executor/device_address.h"
#include "xla/stream_executor/gpu/multicast_memory.h"
#include "xla/tsl/platform/env.h"
#include "xla/tsl/platform/threadpool.h"
#include "xla/tsl/util/tied_ref.h"

namespace xla::gpu {
namespace {

class FakeSymmetricMemory : public SymmetricMemory {
 public:
  explicit FakeSymmetricMemory(stream_executor::DeviceAddressBase addr)
      : addr_(addr) {}

  stream_executor::DeviceAddressBase addr() const override { return addr_; }
  std::string ToString() const override { return "FakeSymmetricMemory"; }
  PackedKernelArg PackKernelArg() const override { return addr_.opaque(); }

 private:
  stream_executor::DeviceAddressBase addr_;
};

class FakeMulticastMemory : public stream_executor::gpu::MulticastMemory {};

GpuCliqueKey MakeCliqueKey() {
  return GpuCliqueKey({GlobalDeviceId(0), GlobalDeviceId(1)},
                      /*num_local_participants=*/2);
}

stream_executor::DeviceAddressBase MakeAddress(uintptr_t ptr, uint64_t size) {
  return stream_executor::DeviceAddressBase(reinterpret_cast<void*>(ptr), size);
}

TEST(CollectiveMemoryCacheTest, FindReturnsNullptrWhenNotCached) {
  CollectiveMemoryCache cache(/*num_devices=*/2);
  GpuCliqueKey clique = MakeCliqueKey();
  stream_executor::DeviceAddressBase addr = MakeAddress(0x1000, 64);

  EXPECT_EQ(cache.FindSymmetricMemory(0, clique, addr), nullptr);
  EXPECT_EQ(cache.FindMulticastMemory(0, clique, addr), nullptr);
  EXPECT_EQ(cache.FindSymmetricMemory(4, clique, addr), nullptr);
  EXPECT_EQ(cache.FindMulticastMemory(4, clique, addr), nullptr);
}

TEST(CollectiveMemoryCacheTest, IsolatesEntriesByDeviceOrdinal) {
  CollectiveMemoryCache cache(/*num_devices=*/2);
  GpuCliqueKey clique = MakeCliqueKey();
  stream_executor::DeviceAddressBase addr = MakeAddress(0x1000, 64);

  tsl::TiedAny owner;
  auto sym0 = std::make_unique<FakeSymmetricMemory>(addr);
  auto* sym0_ptr = sym0.get();
  auto sym1 = std::make_unique<FakeSymmetricMemory>(addr);
  auto* sym1_ptr = sym1.get();
  auto sym5 = std::make_unique<FakeSymmetricMemory>(addr);
  auto* sym5_ptr = sym5.get();

  EXPECT_EQ(cache
                .AddSymmetricMemory(0, clique, addr,
                                    owner.Tie<SymmetricMemory>(std::move(sym0)))
                .get(),
            sym0_ptr);
  EXPECT_EQ(cache
                .AddSymmetricMemory(1, clique, addr,
                                    owner.Tie<SymmetricMemory>(std::move(sym1)))
                .get(),
            sym1_ptr);
  EXPECT_EQ(cache
                .AddSymmetricMemory(5, clique, addr,
                                    owner.Tie<SymmetricMemory>(std::move(sym5)))
                .get(),
            sym5_ptr);

  EXPECT_EQ(cache.FindSymmetricMemory(0, clique, addr).get(), sym0_ptr);
  EXPECT_EQ(cache.FindSymmetricMemory(1, clique, addr).get(), sym1_ptr);
  EXPECT_EQ(cache.FindSymmetricMemory(5, clique, addr).get(), sym5_ptr);
  EXPECT_EQ(cache.FindSymmetricMemory(2, clique, addr), nullptr);
}

TEST(CollectiveMemoryCacheTest, ExpiredTiedRefReturnsNullptr) {
  CollectiveMemoryCache cache(/*num_devices=*/2);
  GpuCliqueKey clique = MakeCliqueKey();
  stream_executor::DeviceAddressBase addr = MakeAddress(0x2000, 64);

  {
    tsl::TiedAny owner;
    cache.AddSymmetricMemory(0, clique, addr,
                             owner.Tie<SymmetricMemory>(
                                 std::make_unique<FakeSymmetricMemory>(addr)));
    cache.AddMulticastMemory(0, clique, addr,
                             owner.Tie<stream_executor::gpu::MulticastMemory>(
                                 std::make_unique<FakeMulticastMemory>()));
    EXPECT_NE(cache.FindSymmetricMemory(0, clique, addr), nullptr);
    EXPECT_NE(cache.FindMulticastMemory(0, clique, addr), nullptr);
  }

  EXPECT_EQ(cache.FindSymmetricMemory(0, clique, addr), nullptr);
  EXPECT_EQ(cache.FindMulticastMemory(0, clique, addr), nullptr);
}

TEST(CollectiveMemoryCacheTest, ConcurrentAddAndFindAcrossDevices) {
  constexpr int kNumDevices = 4;
  constexpr int kNumOrdinals = 8;
  constexpr int kThreadsPerOrdinal = 4;
  constexpr int kNumThreads = kNumOrdinals * kThreadsPerOrdinal;

  CollectiveMemoryCache cache(kNumDevices);
  GpuCliqueKey clique = MakeCliqueKey();
  tsl::TiedAny owner;
  absl::Notification start;

  {
    tsl::thread::ThreadPool pool(tsl::Env::Default(), "cache_test",
                                 kNumThreads);
    for (int t = 0; t < kNumThreads; ++t) {
      pool.Schedule([&, t] {
        int ordinal = t % kNumOrdinals;
        stream_executor::DeviceAddressBase addr =
            MakeAddress(0x1000 * (t + 1), 64);
        auto sym = owner.Tie<SymmetricMemory>(
            std::make_unique<FakeSymmetricMemory>(addr));
        auto mcast = owner.Tie<stream_executor::gpu::MulticastMemory>(
            std::make_unique<FakeMulticastMemory>());

        start.WaitForNotification();

        auto added_sym =
            cache.AddSymmetricMemory(ordinal, clique, addr, std::move(sym));
        auto added_mcast =
            cache.AddMulticastMemory(ordinal, clique, addr, std::move(mcast));
        EXPECT_NE(added_sym, nullptr);
        EXPECT_NE(added_mcast, nullptr);
        EXPECT_EQ(cache.FindSymmetricMemory(ordinal, clique, addr), added_sym);
        EXPECT_EQ(cache.FindMulticastMemory(ordinal, clique, addr),
                  added_mcast);
      });
    }
    start.Notify();
  }
}

}  // namespace
}  // namespace xla::gpu
