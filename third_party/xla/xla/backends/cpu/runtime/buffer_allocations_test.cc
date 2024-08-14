/* Copyright 2024 The OpenXLA Authors.

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

#include "xla/backends/cpu/runtime/buffer_allocations.h"

#include <cstddef>
#include <vector>

#include "xla/service/buffer_assignment.h"
#include "xla/service/maybe_owning_device_memory.h"
#include "xla/stream_executor/device_memory.h"
#include "tsl/platform/statusor.h"
#include "tsl/platform/test.h"

namespace xla::cpu {
namespace {

TEST(BufferAllocationsTest, GetDeviceAddress) {
  std::vector<MaybeOwningDeviceMemory> buffers;
  std::vector<float> data = {1.0, 2.0, 3.0, 4.0};

  size_t size_in_bytes = data.size() * sizeof(float);
  buffers.emplace_back(se::DeviceMemoryBase(data.data(), size_in_bytes));

  BufferAllocations allocations(buffers);

  BufferAllocation alloc(0, size_in_bytes, 0);
  BufferAllocation::Slice slice(&alloc, /*offset=*/2 * sizeof(float),
                                /*size=*/sizeof(float));

  TF_ASSERT_OK_AND_ASSIGN(se::DeviceMemoryBase alloc_mem,
                          allocations.GetDeviceAddress(0));
  EXPECT_EQ(alloc_mem.opaque(), &data[0]);

  TF_ASSERT_OK_AND_ASSIGN(se::DeviceMemoryBase slice_mem,
                          allocations.GetDeviceAddress(slice));
  EXPECT_EQ(slice_mem.opaque(), &data[2]);
}

TEST(BufferAllocationsTest, GetDeviceAddressUnchecked) {
  std::vector<MaybeOwningDeviceMemory> buffers;
  std::vector<float> data = {1.0, 2.0, 3.0, 4.0};

  size_t size_in_bytes = data.size() * sizeof(float);
  buffers.emplace_back(se::DeviceMemoryBase(data.data(), size_in_bytes));

  BufferAllocations allocations(buffers);

  BufferAllocation alloc(0, size_in_bytes, 0);
  BufferAllocation::Slice slice(&alloc, /*offset=*/2 * sizeof(float),
                                /*size=*/sizeof(float));

  se::DeviceMemoryBase alloc_mem = allocations.GetDeviceAddressUnchecked(0);
  EXPECT_EQ(alloc_mem.opaque(), &data[0]);

  se::DeviceMemoryBase slice_mem = allocations.GetDeviceAddressUnchecked(slice);
  EXPECT_EQ(slice_mem.opaque(), &data[2]);
}

}  // namespace
}  // namespace xla::cpu
