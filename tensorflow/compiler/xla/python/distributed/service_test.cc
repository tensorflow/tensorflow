/* Copyright 2020 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/compiler/xla/python/distributed/service.h"

#include "tensorflow/compiler/xla/python/distributed/protocol.pb.h"
#include "tensorflow/core/lib/core/status_test_util.h"
#include "tensorflow/core/platform/test.h"

namespace xla {
namespace {

TEST(TopologyTest, BuildGlobalTopology) {
  std::vector<LocalTopologyProto> locals(2);
  DeviceProto* d0 = locals[0].add_devices();
  d0->set_local_device_ordinal(0);
  DeviceProto* d1 = locals[0].add_devices();
  d1->set_local_device_ordinal(0);
  DeviceProto* d2 = locals[1].add_devices();
  d2->set_local_device_ordinal(0);
  DeviceProto* d3 = locals[1].add_devices();
  d3->set_local_device_ordinal(1);

  GlobalTopologyProto global;
  BuildGlobalTopology(absl::Span<LocalTopologyProto>(locals), &global);
  EXPECT_EQ(global.nodes_size(), 2);
  EXPECT_EQ(global.nodes()[0].devices_size(), 2);
  EXPECT_EQ(global.nodes()[1].devices_size(), 2);
}

}  // namespace
}  // namespace xla
