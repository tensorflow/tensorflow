/* Copyright 2019 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/compiler/mlir/tensorflow/utils/tpu_rewrite_device_util.h"

#include <string>
#include <tuple>
#include <utility>

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallVector.h"
#include "tensorflow/core/lib/core/status_test_util.h"
#include "tensorflow/core/platform/test.h"
#include "tensorflow/core/util/device_name_utils.h"

namespace tensorflow {
namespace {

using Device = DeviceNameUtils::ParsedName;

bool DeviceNamesToParsedNames(llvm::ArrayRef<std::string> device_names,
                              llvm::SmallVectorImpl<Device>* parsed_devices) {
  parsed_devices->reserve(device_names.size());
  for (const auto& device_name : device_names) {
    Device parsed_name;
    if (!DeviceNameUtils::ParseFullName(device_name, &parsed_name))
      return false;

    parsed_devices->push_back(parsed_name);
  }
  return true;
}

using DeviceNames = llvm::SmallVector<std::string, 8>;

struct ParameterizedDeviceSetTest
    : ::testing::TestWithParam<std::tuple<DeviceNames, std::string>> {};

TEST_P(ParameterizedDeviceSetTest, BadDeviceSet) {
  llvm::SmallVector<Device, 8> devices;
  ASSERT_TRUE(DeviceNamesToParsedNames(std::get<0>(GetParam()), &devices));
  std::string compilation_device;
  llvm::SmallVector<std::string, 8> execution_devices;

  Status s = GetTPUCompilationAndExecutionDevices(
      devices, /*num_replicas=*/1, /*num_cores_per_replica=*/1,
      &compilation_device, &execution_devices);
  ASSERT_FALSE(s.ok());
  EXPECT_EQ(s.error_message(), std::get<1>(GetParam()));
}

INSTANTIATE_TEST_SUITE_P(
    BadDeviceSet, ParameterizedDeviceSetTest,
    ::testing::Values(
        std::make_tuple<DeviceNames, std::string>(
            {"/job:localhost/replica:0/task:0/device:CPU:0"},
            "no TPU_SYSTEM devices found"),
        std::make_tuple<DeviceNames, std::string>(
            {"/job:localhost/replica:0/task:0/device:TPU_SYSTEM:0",
             "/job:worker/replica:0/task:0/device:TPU_SYSTEM:0"},
            "found TPU_SYSTEM devices with conflicting jobs 'localhost' and "
            "'worker'"),
        std::make_tuple<DeviceNames, std::string>(
            {"/job:localhost/replica:0/task:0/device:TPU_SYSTEM:0",
             "/job:localhost/replica:1/task:0/device:TPU_SYSTEM:0"},
            "found TPU_SYSTEM devices with conflicting replicas '0' and '1'"),
        std::make_tuple<DeviceNames, std::string>(
            {"/job:localhost/replica:0/task:0/device:TPU_SYSTEM:0",
             "/job:localhost/replica:0/task:0/device:TPU:0",
             "/job:localhost/replica:0/task:0/device:TPU:1",
             "/job:localhost/replica:0/task:1/device:TPU_SYSTEM:0",
             "/job:localhost/replica:0/task:1/device:TPU:0"},
            "expected the number of TPU devices per host to be 2, got 1")));

struct ParameterizedMetadataTest
    : ::testing::TestWithParam<std::tuple<int, int, std::string>> {};

TEST_P(ParameterizedMetadataTest, BadMetadata) {
  llvm::SmallVector<Device, 8> devices;
  ASSERT_TRUE(DeviceNamesToParsedNames(
      {"/job:worker/replica:0/task:0/device:TPU_SYSTEM:0",
       "/job:worker/replica:0/task:0/device:TPU:0",
       "/job:worker/replica:0/task:0/device:TPU:1",
       "/job:worker/replica:0/task:1/device:TPU_SYSTEM:0",
       "/job:worker/replica:0/task:1/device:TPU:0",
       "/job:worker/replica:0/task:1/device:TPU:1"},
      &devices));
  std::string compilation_device;
  llvm::SmallVector<std::string, 8> execution_devices;

  Status s = GetTPUCompilationAndExecutionDevices(
      devices, std::get<0>(GetParam()), std::get<1>(GetParam()),
      &compilation_device, &execution_devices);
  ASSERT_FALSE(s.ok());
  EXPECT_EQ(s.error_message(), std::get<2>(GetParam()));
}

INSTANTIATE_TEST_SUITE_P(
    BadMetadata, ParameterizedMetadataTest,
    ::testing::Values(
        std::make_tuple(1, 2,
                        "num_cores_per_replica must be equal to 1, got 2"),
        std::make_tuple(8, 1, "num_replicas must be equal to 1 or 4, got 8")));

TEST(TPURewriteDeviceUtilTest, NumReplicasNumTPUs) {
  llvm::SmallVector<Device, 8> devices;
  ASSERT_TRUE(DeviceNamesToParsedNames(
      {"/job:localhost/replica:0/task:0/device:CPU:0",
       "/job:worker/replica:0/task:1/device:TPU_SYSTEM:0",
       "/job:worker/replica:0/task:1/device:TPU:1",
       "/job:worker/replica:0/task:0/device:TPU:0",
       "/job:worker/replica:0/task:1/device:CPU:0",
       "/job:worker/replica:0/task:0/device:TPU:1",
       "/job:worker/replica:0/task:0/device:TPU_SYSTEM:0",
       "/job:worker/replica:0/task:1/device:TPU:0",
       "/job:worker/replica:0/task:0/device:CPU:0"},
      &devices));
  std::string compilation_device;
  llvm::SmallVector<std::string, 8> execution_devices;

  TF_EXPECT_OK(GetTPUCompilationAndExecutionDevices(
      devices, /*num_replicas=*/4, /*num_cores_per_replica=*/1,
      &compilation_device, &execution_devices));

  EXPECT_EQ(compilation_device, "/job:worker/replica:0/task:0/device:CPU:0");
  ASSERT_EQ(execution_devices.size(), 4);
  EXPECT_EQ(execution_devices[0], "/job:worker/replica:0/task:0/device:TPU:0");
  EXPECT_EQ(execution_devices[1], "/job:worker/replica:0/task:0/device:TPU:1");
  EXPECT_EQ(execution_devices[2], "/job:worker/replica:0/task:1/device:TPU:0");
  EXPECT_EQ(execution_devices[3], "/job:worker/replica:0/task:1/device:TPU:1");
}

}  // anonymous namespace
}  // namespace tensorflow
