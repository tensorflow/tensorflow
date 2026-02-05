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

#include "xla/backends/gpu/runtime/collective_memory_requests.h"

#include <array>
#include <vector>

#include "xla/backends/gpu/collectives/gpu_clique_key.h"
#include "xla/runtime/device_id.h"
#include "xla/service/gpu/buffer_allocations.h"
#include "xla/stream_executor/device_address.h"
#include "xla/tsl/lib/core/status_test_util.h"
#include "xla/tsl/platform/test.h"

namespace xla::gpu {

TEST(CollectiveMemoryRequestsTest, OrderedRequests) {
  GlobalDeviceId d0 = GlobalDeviceId(0);
  GlobalDeviceId d1 = GlobalDeviceId(1);
  GlobalDeviceId d2 = GlobalDeviceId(2);
  GlobalDeviceId d3 = GlobalDeviceId(3);

  GpuCliqueKey k0({d2, d3}, 2);
  GpuCliqueKey k1({d0, d1}, 2);
  GpuCliqueKey k2({d0, d1, d2, d3}, 4);

  std::array<char, 10> data;
  se::DeviceAddressBase buffer(data.data(), 10);
  se::DeviceAddressBase slice = buffer.GetByteSlice(2, 2);

  BufferAllocations buffers({buffer}, /*device_ordinal=*/0, nullptr);
  CollectiveMemoryRequests requests(buffers);

  TF_ASSERT_OK(requests.RequestSymmetricAllocation(k0, 0));
  TF_ASSERT_OK(requests.RequestSymmetricAllocation(k0, 1));
  TF_ASSERT_OK(requests.RequestSymmetricAllocation(k1, 0));
  TF_ASSERT_OK(requests.RequestSymmetricAddress(k2, slice));

  // Check that we create symmetric memories according to the GPU clique key
  // ordering.
  auto ordered_requests = requests.OrderedSymmetricAllocations();
  ASSERT_EQ(ordered_requests.size(), 3);
  EXPECT_EQ(ordered_requests[0].key, k2);
  EXPECT_EQ(ordered_requests[1].key, k0);
  EXPECT_EQ(ordered_requests[2].key, k1);
}

}  // namespace xla::gpu
