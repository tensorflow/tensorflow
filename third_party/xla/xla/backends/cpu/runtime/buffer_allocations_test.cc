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

#include "xla/backends/cpu/runtime/thunk_testlib.h"
#include "xla/literal_util.h"
#include "xla/service/buffer_assignment.h"
#include "xla/stream_executor/device_memory.h"
#include "xla/tsl/platform/statusor.h"
#include "xla/tsl/platform/test.h"

namespace xla::cpu {
namespace {

TEST(BufferAllocationsTest, GetDeviceAddress) {
  auto data = LiteralUtil::CreateR1<float>({1.0, 2.0, 3.0, 4.0});

  BufferAllocation alloc = CreateBufferAllocation(0, data);
  BufferAllocation::Slice slice = CreateBufferAllocationSlice(
      alloc, /*offset=*/2 * sizeof(float), /*size=*/sizeof(float));

  BufferAllocations allocations = CreateBufferAllocations(data);

  TF_ASSERT_OK_AND_ASSIGN(se::DeviceMemoryBase alloc_mem,
                          allocations.GetDeviceAddress(0));
  EXPECT_EQ(alloc_mem.opaque(), &data.data<float>()[0]);

  TF_ASSERT_OK_AND_ASSIGN(se::DeviceMemoryBase slice_mem,
                          allocations.GetDeviceAddress(slice));
  EXPECT_EQ(slice_mem.opaque(), &data.data<float>()[2]);
}

TEST(BufferAllocationsTest, GetDeviceAddressUnchecked) {
  auto data = LiteralUtil::CreateR1<float>({1.0, 2.0, 3.0, 4.0});

  BufferAllocation alloc = CreateBufferAllocation(0, data);
  BufferAllocation::Slice slice = CreateBufferAllocationSlice(
      alloc, /*offset=*/2 * sizeof(float), /*size=*/sizeof(float));

  BufferAllocations allocations = CreateBufferAllocations(data);

  se::DeviceMemoryBase alloc_mem = allocations.GetDeviceAddressUnchecked(0);
  EXPECT_EQ(alloc_mem.opaque(), &data.data<float>()[0]);

  se::DeviceMemoryBase slice_mem = allocations.GetDeviceAddressUnchecked(slice);
  EXPECT_EQ(slice_mem.opaque(), &data.data<float>()[2]);
}

}  // namespace
}  // namespace xla::cpu
