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

#include "xla/service/gpu/buffer_allocations.h"

#include <optional>
#include <vector>

#include <gtest/gtest.h>
#include "xla/stream_executor/device_address.h"

namespace xla::gpu {
namespace {

TEST(BufferAllocationsTest, FindAllocationIndexReturnsCorrectIndex) {
  // Two 64-byte buffers at distinct addresses.
  char buf0[64];
  char buf1[64];
  std::vector<se::DeviceAddressBase> buffers = {
      se::DeviceAddressBase(buf0, 64),
      se::DeviceAddressBase(buf1, 64),
  };

  BufferAllocations allocs(buffers, /*device_ordinal=*/0,
                           /*memory_allocator=*/nullptr);

  // Address at the start of buf0 → index 0.
  EXPECT_EQ(allocs.FindAllocationIndex(se::DeviceAddressBase(buf0, 1)), 0);

  // Address in the middle of buf1 → index 1.
  EXPECT_EQ(allocs.FindAllocationIndex(se::DeviceAddressBase(buf1 + 32, 1)), 1);
}

TEST(BufferAllocationsTest, FindAllocationIndexReturnsNulloptForUnknown) {
  char buf[64];
  char other[64];
  std::vector<se::DeviceAddressBase> buffers = {
      se::DeviceAddressBase(buf, 64),
  };

  BufferAllocations allocs(buffers, /*device_ordinal=*/0,
                           /*memory_allocator=*/nullptr);

  EXPECT_EQ(allocs.FindAllocationIndex(se::DeviceAddressBase(other, 1)),
            std::nullopt);
}

TEST(BufferAllocationsTest, FindAllocationIndexExcludesEndBoundary) {
  char buf[64];
  std::vector<se::DeviceAddressBase> buffers = {
      se::DeviceAddressBase(buf, 64),
  };

  BufferAllocations allocs(buffers, /*device_ordinal=*/0,
                           /*memory_allocator=*/nullptr);

  // Address at exactly buf + size should NOT match (strict less-than).
  EXPECT_EQ(allocs.FindAllocationIndex(se::DeviceAddressBase(buf + 64, 1)),
            std::nullopt);

  // Address at buf + size - 1 should still match.
  EXPECT_EQ(allocs.FindAllocationIndex(se::DeviceAddressBase(buf + 63, 1)), 0);
}

}  // namespace
}  // namespace xla::gpu
