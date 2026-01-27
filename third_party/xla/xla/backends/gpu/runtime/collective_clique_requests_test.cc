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

#include "xla/backends/gpu/runtime/collective_clique_requests.h"

#include <vector>

#include "xla/backends/gpu/collectives/gpu_clique_key.h"
#include "xla/backends/gpu/collectives/gpu_communicator.h"
#include "xla/runtime/device_id.h"
#include "xla/tsl/lib/core/status_test_util.h"
#include "xla/tsl/platform/test.h"
#include "xla/xla_data.pb.h"

namespace xla::gpu {

TEST(CollectiveCliqueRequestsTest, OrderedRequests) {
  GlobalDeviceId d0 = GlobalDeviceId(0);
  GlobalDeviceId d1 = GlobalDeviceId(1);
  GlobalDeviceId d2 = GlobalDeviceId(2);
  GlobalDeviceId d3 = GlobalDeviceId(3);

  GpuCliqueKey k0({d2, d3}, 2);
  GpuCliqueKey k1({d0, d1}, 2);
  GpuCliqueKey k2({d0, d1, d2, d3}, 4);

  CollectiveCliqueRequests requests;
  TF_ASSERT_OK(requests.RequestClique(k0));
  TF_ASSERT_OK(requests.RequestClique(k1));
  TF_ASSERT_OK(requests.RequestClique(k2));

  // Check that we acquire larger cliques first, and then cliques with smaller
  // id first, as acquiring cliques according to natural clique key order might
  // lead to deadlocks during communicator splitting.
  auto ordered_requests = requests.OrderedRequestedCliques();
  ASSERT_EQ(ordered_requests.size(), 3);
  EXPECT_EQ(ordered_requests[0].key, k2);
  EXPECT_EQ(ordered_requests[1].key, k0);
  EXPECT_EQ(ordered_requests[2].key, k1);
}

TEST(CollectiveCliqueRequestsTest, RequestDevComms) {
  GlobalDeviceId d0 = GlobalDeviceId(0);
  GlobalDeviceId d1 = GlobalDeviceId(1);

  GpuCliqueKey k0({d0, d1}, 2);

  GpuDeviceCommunicator::Requirements dev_comm0{8};
  GpuDeviceCommunicator::Requirements dev_comm1{16};

  CollectiveCliqueRequests requests;
  TF_ASSERT_OK(requests.RequestClique(k0, {dev_comm0}));
  TF_ASSERT_OK(requests.RequestClique(k0, {dev_comm1}));

  auto ordered_requests = requests.OrderedRequestedCliques();
  ASSERT_EQ(ordered_requests.size(), 1);
  EXPECT_EQ(ordered_requests[0].key, k0);
  ASSERT_EQ(ordered_requests[0].dev_comms.size(), 2);
  EXPECT_TRUE(ordered_requests[0].dev_comms.contains(dev_comm0));
  EXPECT_TRUE(ordered_requests[0].dev_comms.contains(dev_comm1));
}

}  // namespace xla::gpu
