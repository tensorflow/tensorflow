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
#include "tensorflow/core/lib/strings/strcat.h"
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
  devices["/job:localhost/replica:0/task:0/device:GPU:0"] = gpu_device;
  VirtualCluster cluster(devices);
  VirtualPlacer placer(devices);

  NodeDef node;
  node.set_op("Conv2D");
  // node.device() is empty, but GPU is default device if there is.
  EXPECT_EQ("GPU", placer.get_device(node).type());
  EXPECT_EQ("/job:localhost/replica:0/task:0/device:GPU:0",
            placer.get_canonical_device_name(node));

  node.set_device("CPU");
  EXPECT_EQ("CPU", placer.get_device(node).type());
  EXPECT_EQ("/job:localhost/replica:0/task:0/cpu:0",
            placer.get_canonical_device_name(node));

  node.set_device("GPU:0");
  EXPECT_EQ("GPU", placer.get_device(node).type());
  EXPECT_EQ("/job:localhost/replica:0/task:0/device:GPU:0",
            placer.get_canonical_device_name(node));
}

TEST(VirtualPlacerTest, ShortNames) {
  // Create a virtual cluster with a local CPU and a local GPU
  std::unordered_map<string, DeviceProperties> devices;
  DeviceProperties cpu_device;
  cpu_device.set_type("CPU");
  devices["/CPU:0"] = cpu_device;
  DeviceProperties gpu_device;
  gpu_device.set_type("GPU");
  devices["/GPU:0"] = gpu_device;
  VirtualCluster cluster(devices);
  VirtualPlacer placer(devices);

  NodeDef node;
  node.set_op("Conv2D");
  // node.device() is empty, but GPU is default device if there is.
  EXPECT_EQ("GPU", placer.get_device(node).type());
  EXPECT_EQ("/GPU:0", placer.get_canonical_device_name(node));

  node.set_device("CPU");
  EXPECT_EQ("CPU", placer.get_device(node).type());
  EXPECT_EQ("/CPU:0", placer.get_canonical_device_name(node));

  node.set_device("GPU:0");
  EXPECT_EQ("GPU", placer.get_device(node).type());
  EXPECT_EQ("/GPU:0", placer.get_canonical_device_name(node));
}

TEST(VirtualPlacerTest, PlacementOnNonDefaultDevice) {
  // Create a virtual cluster with a CPU and a device:TPU
  // Test that placement on TPU works
  // In contrast with GPU, TPU is not selected as default device at the moment.

  std::unordered_map<string, DeviceProperties> devices;
  DeviceProperties cpu_device;
  cpu_device.set_type("CPU");
  devices["/job:localhost/replica:0/task:0/cpu:0"] = cpu_device;
  DeviceProperties tpu_device;
  tpu_device.set_type("TPU");
  devices["/job:localhost/replica:0/task:0/device:TPU:0"] = tpu_device;
  VirtualCluster cluster(devices);
  VirtualPlacer placer(devices);

  NodeDef node;
  node.set_op("Conv2D");
  // node.device() is empty, and CPU is default device.
  EXPECT_EQ("CPU", placer.get_device(node).type());
  EXPECT_EQ("/job:localhost/replica:0/task:0/cpu:0",
            placer.get_canonical_device_name(node));

  node.set_device("/device:TPU:0");
  EXPECT_EQ("TPU", placer.get_device(node).type());
  EXPECT_EQ("/job:localhost/replica:0/task:0/device:TPU:0",
            placer.get_canonical_device_name(node));
}

TEST(VirtualPlacerTest, EmptyJobName) {
  // Virtual placer choose job name from the devices in cluster if a device name
  // of an op is empty. In case there are more than one kind of job name
  // or job names are missing in the devices in cluster, we use local_host.
  for (const string& job_name : {"localhost", "worker", "worker_train"}) {
    std::unordered_map<string, DeviceProperties> devices;
    DeviceProperties cpu_device;
    cpu_device.set_type("CPU");
    devices[strings::StrCat("/job:", job_name, "/replica:0/task:0/cpu:0")] =
        cpu_device;
    DeviceProperties gpu_device;
    gpu_device.set_type("GPU");
    devices[strings::StrCat("/job:", job_name,
                            "/replica:0/task:0/device:GPU:0")] = gpu_device;
    VirtualCluster cluster(devices);
    VirtualPlacer placer(devices);

    NodeDef node;
    node.set_op("Conv2D");
    node.set_device("/device:CPU:0");
    EXPECT_EQ(strings::StrCat("/job:", job_name, "/replica:0/task:0/cpu:0"),
              placer.get_canonical_device_name(node));
    node.set_device("/device:GPU:0");
    EXPECT_EQ(
        strings::StrCat("/job:", job_name, "/replica:0/task:0/device:GPU:0"),
        placer.get_canonical_device_name(node));
  }

  // When more than one job names are used, we use default "localhost"
  // This may be improved later.
  std::unordered_map<string, DeviceProperties> devices;
  DeviceProperties cpu_device;
  cpu_device.set_type("CPU");
  devices["/job:localhost/replica:0/task:0/cpu:0"] = cpu_device;
  devices["/job:ps/replica:0/task:0/cpu:0"] = cpu_device;
  devices["/job:worker/replica:0/task:0/cpu:0"] = cpu_device;
  VirtualCluster cluster(devices);
  VirtualPlacer placer(devices);

  NodeDef node;
  node.set_op("Conv2D");
  node.set_device("/device:CPU:0");
  EXPECT_EQ("/job:localhost/replica:0/task:0/cpu:0",
            placer.get_canonical_device_name(node));
}

string GetDefaultDeviceName(
    const std::unordered_map<string, DeviceProperties>& devices) {
  VirtualCluster cluster(devices);
  VirtualPlacer placer(devices);
  NodeDef node;
  node.set_op("Conv2D");
  // Device is not set to the node, so get_canonical_device_name() will return
  // the default_device_.
  return placer.get_canonical_device_name(node);
}

TEST(VirtualPlacerTest, DefaultDevice) {
  std::unordered_map<string, DeviceProperties> devices;
  DeviceProperties cpu_device;
  cpu_device.set_type("CPU");
  devices["/job:worker/replica:0/task:0/cpu:0"] = cpu_device;

  // CPU is default when there is only CPU.
  EXPECT_EQ("/job:worker/replica:0/task:0/cpu:0",
            GetDefaultDeviceName(devices));

  DeviceProperties gpu_device;
  gpu_device.set_type("GPU");

  // If there is any GPU, then gpu:0 is default device.
  for (int i = 0; i < 8; i++) {
    devices[strings::StrCat("/job:worker/replica:0/task:0/gpu:", i)] =
        gpu_device;
    EXPECT_EQ("/job:worker/replica:0/task:0/gpu:0",
              GetDefaultDeviceName(devices));
  }
}

TEST(VirtualPlacerTest, MultiReplica) {
  // Create a cluster with 8 workers, each with 8 GPUs.
  std::unordered_map<string, DeviceProperties> devices;
  DeviceProperties cpu_device;
  cpu_device.set_type("CPU");
  DeviceProperties gpu_device;
  gpu_device.set_type("GPU");
  for (int i = 0; i < 8; i++) {
    devices[strings::StrCat("/job:worker/replica:", i, "/task:0/cpu:0")] =
        cpu_device;
    for (int j = 0; j < 8; j++) {
      devices[strings::StrCat("/job:worker/replica:", i, "/task:0/gpu:", j)] =
          gpu_device;
    }
  }

  std::unique_ptr<VirtualCluster> cluster(new VirtualCluster(devices));
  std::unique_ptr<VirtualPlacer> placer(new VirtualPlacer(devices));

  auto get_device_name = [&placer](const string& device) -> string {
    NodeDef node;
    node.set_op("Conv2D");
    node.set_device(device);
    return placer->get_canonical_device_name(node);
  };

  // Validate device name is correct when we pass only replica ID and device
  // name.
  EXPECT_EQ("/job:worker/replica:0/task:0/cpu:0",
            get_device_name("/replica:0/cpu:0"));
  EXPECT_EQ("/job:worker/replica:2/task:0/cpu:0",
            get_device_name("/replica:2/cpu:0"));
  EXPECT_EQ("/job:worker/replica:7/task:0/cpu:0",
            get_device_name("/replica:7/cpu:0"));
  EXPECT_EQ("/job:worker/replica:3/task:0/gpu:0",
            get_device_name("/replica:3/gpu:0"));
  EXPECT_EQ("/job:worker/replica:5/task:0/gpu:3",
            get_device_name("/replica:5/gpu:3"));
  EXPECT_EQ("/job:worker/replica:4/task:0/gpu:7",
            get_device_name("/replica:4/gpu:7"));

  // Now add PS replicas; with multiple job names present in the cluster,
  // device names in nodes should specify job names correctly.
  for (int i = 0; i < 4; i++) {
    devices[strings::StrCat("/job:ps/replica:", i, "/task:0/cpu:0")] =
        cpu_device;
  }
  cluster.reset(new VirtualCluster(devices));
  placer.reset(new VirtualPlacer(cluster->GetDevices()));
  EXPECT_EQ("/job:worker/replica:0/task:0/cpu:0",
            get_device_name("/job:worker/replica:0/cpu:0"));
  EXPECT_EQ("/job:worker/replica:7/task:0/gpu:3",
            get_device_name("/job:worker/replica:7/gpu:3"));
  EXPECT_EQ("/job:ps/replica:0/task:0/cpu:0",
            get_device_name("/job:ps/replica:0/cpu:0"));
  EXPECT_EQ("/job:ps/replica:1/task:0/cpu:0",
            get_device_name("/job:ps/replica:1/cpu:0"));
  EXPECT_EQ("/job:ps/replica:2/task:0/cpu:0",
            get_device_name("/job:ps/replica:2/cpu:0"));
  EXPECT_EQ("/job:ps/replica:3/task:0/cpu:0",
            get_device_name("/job:ps/replica:3/cpu:0"));
}

TEST(VirtualPlacerTest, FallBackUnknown) {
  // Virtual placer falls back to "UNKNOWN" only if there are no devices in the
  // cluster.
  std::unordered_map<string, DeviceProperties> devices;
  VirtualCluster cluster(devices);
  VirtualPlacer placer(devices);

  NodeDef node;
  node.set_op("Conv2D");

  // Device falls back to UNKNOWN since the cluster has no devices.
  EXPECT_EQ("UNKNOWN", placer.get_device(node).type());
  EXPECT_EQ("UNKNOWN", placer.get_canonical_device_name(node));
}

TEST(VirtualPlacerTest, FallBackCPU) {
  std::unordered_map<string, DeviceProperties> devices;
  DeviceProperties cpu_device;
  cpu_device.set_type("CPU");
  devices["/job:my_job/replica:0/task:0/cpu:0"] = cpu_device;
  VirtualCluster cluster(devices);
  VirtualPlacer placer(devices);

  NodeDef node;
  node.set_op("Conv2D");

  // Device falls back to CPU since there is no GPU.
  EXPECT_EQ("CPU", placer.get_device(node).type());
  EXPECT_EQ("/job:my_job/replica:0/task:0/cpu:0",
            placer.get_canonical_device_name(node));
}

TEST(VirtualPlacerTest, RemoteDevices) {
  std::unordered_map<string, DeviceProperties> devices;
  DeviceProperties cpu_device;
  cpu_device.set_type("CPU");
  devices["/job:my_job/replica:0/task:0/cpu:0"] = cpu_device;
  DeviceProperties gpu_device;
  gpu_device.set_type("GPU");
  devices["/job:my_job/replica:0/task:0/device:GPU:0"] = gpu_device;
  VirtualCluster cluster(devices);
  VirtualPlacer placer(devices);

  NodeDef node;
  node.set_op("Conv2D");

  // Device falls back to GPU.
  EXPECT_EQ("GPU", placer.get_device(node).type());
  EXPECT_EQ("/job:my_job/replica:0/task:0/device:GPU:0",
            placer.get_canonical_device_name(node));

  node.set_device("/job:my_job/replica:0/task:0/cpu:0");
  EXPECT_EQ("CPU", placer.get_device(node).type());
  EXPECT_EQ("/job:my_job/replica:0/task:0/cpu:0",
            placer.get_canonical_device_name(node));

  node.set_device("/job:my_job/replica:0/task:0/device:GPU:0");
  EXPECT_EQ("GPU", placer.get_device(node).type());
  EXPECT_EQ("/job:my_job/replica:0/task:0/device:GPU:0",
            placer.get_canonical_device_name(node));

  // There is no local cpu available. Device falls back to GPU.
  node.set_device("CPU");
  EXPECT_EQ("GPU", placer.get_device(node).type());
  EXPECT_EQ("/job:my_job/replica:0/task:0/device:GPU:0",
            placer.get_canonical_device_name(node));

  node.set_device("GPU:0");
  // There is no local GPU available. Fall back to default GPU.
  EXPECT_EQ("GPU", placer.get_device(node).type());
  EXPECT_EQ("/job:my_job/replica:0/task:0/device:GPU:0",
            placer.get_canonical_device_name(node));

  // This isn't a valid name. Fall back to GPU.
  node.set_device("/job:my_job/replica:0/task:0");
  EXPECT_EQ("GPU", placer.get_device(node).type());
  EXPECT_EQ("/job:my_job/replica:0/task:0/device:GPU:0",
            placer.get_canonical_device_name(node));
}

}  // end namespace grappler
}  // end namespace tensorflow
