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

#include "xla/backends/gpu/runtime/collective_memory.h"

#include <array>
#include <cstddef>
#include <memory>
#include <string>
#include <tuple>

#include <gtest/gtest.h>
#include "xla/backends/gpu/collectives/gpu_clique_key.h"
#include "xla/core/collectives/symmetric_memory.h"
#include "xla/runtime/device_id.h"
#include "xla/service/buffer_assignment.h"
#include "xla/service/gpu/buffer_allocations.h"
#include "xla/stream_executor/device_address.h"

namespace xla::gpu {
namespace {

class FakeSymmetricMemory : public SymmetricMemory {
 public:
  explicit FakeSymmetricMemory(se::DeviceAddressBase addr) : addr_(addr) {}
  se::DeviceAddressBase addr() const override { return addr_; }
  std::string ToString() const override { return "FakeSymmetricMemory"; }
  PackedKernelArg PackKernelArg() const override { return addr_.opaque(); }

 private:
  se::DeviceAddressBase addr_;
};

TEST(CollectiveMemoryTest, FindWeakSymmetricMemory) {
  GpuCliqueKey clique({GlobalDeviceId(0), GlobalDeviceId(1)}, 2);
  std::array<char, 64> data{};
  se::DeviceAddressBase base_addr(data.data(), data.size());
  BufferAllocations buffers({base_addr}, /*device_ordinal=*/0, nullptr);
  BufferAllocation alloc(/*index=*/0, /*size=*/64, /*color=*/0);

  auto sym = std::make_shared<FakeSymmetricMemory>(base_addr);
  std::weak_ptr<SymmetricMemory> weak_by_addr;
  {
    CollectiveMemory memory(buffers, {{{clique, 0}, sym}},
                            /*peer_memories=*/{});

    auto [weak_by_idx, off_idx] =
        memory.FindWeakSymmetricMemory(clique, BufferAllocation::Index(0));
    EXPECT_EQ(weak_by_idx.lock(), sym);
    EXPECT_EQ(off_idx, 0);

    auto [weak_by_slice, off_slice] = memory.FindWeakSymmetricMemory(
        clique, BufferAllocation::Slice(&alloc, /*offset=*/16, /*size=*/32));
    EXPECT_EQ(weak_by_slice.lock(), sym);
    EXPECT_EQ(off_slice, 16);

    size_t off_addr = 0;
    std::tie(weak_by_addr, off_addr) =
        memory.FindWeakSymmetricMemory(clique, base_addr.GetByteSlice(24, 16));
    EXPECT_EQ(weak_by_addr.lock(), sym);
    EXPECT_EQ(off_addr, 24);

    auto [missing, off_missing] =
        memory.FindWeakSymmetricMemory(clique, BufferAllocation::Index(1));
    EXPECT_TRUE(missing.expired());
    EXPECT_EQ(off_missing, 0);
  }

  EXPECT_FALSE(weak_by_addr.expired());
  sym.reset();
  EXPECT_TRUE(weak_by_addr.expired());
}

}  // namespace
}  // namespace xla::gpu
