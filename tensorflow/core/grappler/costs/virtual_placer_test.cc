/* Copyright 2017 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/core/grappler/costs/virtual_placer.h"
#include "tensorflow/core/framework/node_def.pb.h"
#include "tensorflow/core/grappler/clusters/virtual_cluster.h"
#include "tensorflow/core/platform/test.h"
#include "tensorflow/core/protobuf/device_properties.pb.h"

namespace tensorflow {
namespace grappler {

TEST(VirtualPlacerTest, LocalDevices) {
  // Create a virtual cluster with a local CPU and a local GPU
  std::unordered_map<string, DeviceProperties> devices;
  DeviceProperties cpu_device;
  cpu_device.set_type("CPU");
  devices["/job:localhost/replica:0/task:0/cpu:0"] = cpu_device;
  DeviceProperties gpu_device;
  gpu_device.set_type("GPU");
  devices["/job:localhost/replica:0/task:0/gpu:0"] = gpu_device;
  VirtualCluster cluster(devices);
  VirtualPlacer placer(&cluster);

  NodeDef node;
  node.set_op("Conv2D");
  EXPECT_EQ("GPU", placer.get_device(node).type());
  EXPECT_EQ("/job:localhost/replica:0/task:0/gpu:0",
            placer.get_canonical_device_name(node));

  node.set_device("CPU");
  EXPECT_EQ("CPU", placer.get_device(node).type());
  EXPECT_EQ("/job:localhost/replica:0/task:0/cpu:0",
            placer.get_canonical_device_name(node));

  node.set_device("GPU:0");
  EXPECT_EQ("GPU", placer.get_device(node).type());
  EXPECT_EQ("/job:localhost/replica:0/task:0/gpu:0",
            placer.get_canonical_device_name(node));
}

TEST(VirtualPlacerTest, RemoteDevices) {
  std::unordered_map<string, DeviceProperties> devices;
  DeviceProperties cpu_device;
  cpu_device.set_type("CPU");
  devices["/job:my_job/replica:0/task:0/cpu:0"] = cpu_device;
  DeviceProperties gpu_device;
  gpu_device.set_type("GPU");
  devices["/job:my_job/replica:0/task:0/gpu:0"] = gpu_device;
  VirtualCluster cluster(devices);
  VirtualPlacer placer(&cluster);

  NodeDef node;
  node.set_op("Conv2D");
  // There is no local device available
  EXPECT_EQ("UNKNOWN", placer.get_device(node).type());
  EXPECT_EQ("", placer.get_canonical_device_name(node));

  node.set_device("/job:my_job/replica:0/task:0/cpu:0");
  EXPECT_EQ("CPU", placer.get_device(node).type());
  EXPECT_EQ("/job:my_job/replica:0/task:0/cpu:0",
            placer.get_canonical_device_name(node));

  node.set_device("/job:my_job/replica:0/task:0/gpu:0");
  EXPECT_EQ("GPU", placer.get_device(node).type());
  EXPECT_EQ("/job:my_job/replica:0/task:0/gpu:0",
            placer.get_canonical_device_name(node));

  // There is no local CPU available
  node.set_device("CPU");
  EXPECT_EQ("UNKNOWN", placer.get_device(node).type());
  EXPECT_EQ("", placer.get_canonical_device_name(node));

  node.set_device("GPU:0");
  // There is no local GPU available
  EXPECT_EQ("UNKNOWN", placer.get_device(node).type());
  EXPECT_EQ("", placer.get_canonical_device_name(node));

  // This isn't a valid name
  node.set_device("/job:my_job/replica:0/task:0");
  EXPECT_EQ("UNKNOWN", placer.get_device(node).type());
  EXPECT_EQ("", placer.get_canonical_device_name(node));
}

}  // end namespace grappler
}  // end namespace tensorflow
