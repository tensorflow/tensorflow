/* Copyright 2026 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/core/common_runtime/eager/validate_function_devices.h"

#include <string>
#include <vector>

#include "absl/status/status.h"
#include "tensorflow/core/framework/device_attributes.pb.h"
#include "tensorflow/core/framework/function.pb.h"
#include "tensorflow/core/framework/node_def.pb.h"
#include "tensorflow/core/platform/status_matchers.h"
#include "tensorflow/core/platform/test.h"

namespace tensorflow {

namespace {

// A FunctionDef with a single node carrying the given device constraint.
FunctionDef MakeFunctionWithDevice(const std::string& node_name,
                                   const std::string& op,
                                   const std::string& device) {
  FunctionDef fdef;
  NodeDef* node = fdef.add_node_def();
  node->set_name(node_name);
  node->set_op(op);
  node->set_device(device);
  return fdef;
}

std::vector<DeviceAttributes> LocalCpuDevices() {
  DeviceAttributes attrs;
  attrs.set_name("/job:localhost/replica:0/task:0/device:CPU:0");
  attrs.set_device_type("CPU");
  return {attrs};
}

bool IsInvalidArgument(const absl::Status& status) {
  return status.code() == absl::StatusCode::kInvalidArgument;
}

bool Contains(const absl::Status& status, const std::string& substring) {
  return std::string(status.message()).find(substring) != std::string::npos;
}

}  // namespace

TEST(ValidateFunctionDevicesTest, EmptyFunctionIsOk) {
  FunctionDef fdef;
  EXPECT_THAT(ValidateFunctionDeviceConstraints(fdef, LocalCpuDevices()),
              ::tensorflow::testing::IsOk());
}

TEST(ValidateFunctionDevicesTest, NoDeviceConstraintIsOk) {
  FunctionDef fdef = MakeFunctionWithDevice("add", "AddV2", "");
  EXPECT_THAT(ValidateFunctionDeviceConstraints(fdef, LocalCpuDevices()),
              ::tensorflow::testing::IsOk());
}

TEST(ValidateFunctionDevicesTest, FullySpecifiedDeviceIsOk) {
  FunctionDef fdef = MakeFunctionWithDevice(
      "add", "AddV2", "/job:localhost/replica:0/task:0/device:CPU:0");
  EXPECT_THAT(ValidateFunctionDeviceConstraints(fdef, LocalCpuDevices()),
              ::tensorflow::testing::IsOk());
}

TEST(ValidateFunctionDevicesTest, PartialJobConstraintIsOk) {
  // '/job:localhost' is a partial constraint without a device type; it must
  // be satisfied by the fully-specified available CPU device.
  FunctionDef fdef = MakeFunctionWithDevice("add", "AddV2", "/job:localhost");
  EXPECT_THAT(ValidateFunctionDeviceConstraints(fdef, LocalCpuDevices()),
              ::tensorflow::testing::IsOk());
}

TEST(ValidateFunctionDevicesTest, PartialTaskConstraintIsOk) {
  FunctionDef fdef = MakeFunctionWithDevice("add", "AddV2", "/task:0");
  EXPECT_THAT(ValidateFunctionDeviceConstraints(fdef, LocalCpuDevices()),
              ::tensorflow::testing::IsOk());
}

TEST(ValidateFunctionDevicesTest, LocalDeviceNameIsOk) {
  FunctionDef fdef = MakeFunctionWithDevice("add", "AddV2", "CPU:0");
  EXPECT_THAT(ValidateFunctionDeviceConstraints(fdef, LocalCpuDevices()),
              ::tensorflow::testing::IsOk());
}

TEST(ValidateFunctionDevicesTest, UnsatisfiedJobFails) {
  FunctionDef fdef = MakeFunctionWithDevice("add", "AddV2", "/job:worker");
  absl::Status status =
      ValidateFunctionDeviceConstraints(fdef, LocalCpuDevices());
  EXPECT_TRUE(IsInvalidArgument(status));
  EXPECT_TRUE(Contains(status, "Could not satisfy device specification"));
  EXPECT_TRUE(Contains(status, "/job:worker"));
  EXPECT_TRUE(Contains(status, "/job:localhost/replica:0/task:0/device:CPU:0"));
}

TEST(ValidateFunctionDevicesTest, InvalidDeviceIdFails) {
  // The device type exists, but the id does not: the partial constraint must
  // still be rejected.
  FunctionDef fdef = MakeFunctionWithDevice("add", "AddV2", "/device:CPU:99");
  absl::Status status =
      ValidateFunctionDeviceConstraints(fdef, LocalCpuDevices());
  EXPECT_TRUE(IsInvalidArgument(status));
  EXPECT_TRUE(Contains(status, "Could not satisfy device specification"));
}

TEST(ValidateFunctionDevicesTest, InvalidDeviceTypeFails) {
  FunctionDef fdef =
      MakeFunctionWithDevice("add", "AddV2", "/device:NONEXISTENT:0");
  absl::Status status =
      ValidateFunctionDeviceConstraints(fdef, LocalCpuDevices());
  EXPECT_TRUE(IsInvalidArgument(status));
  EXPECT_TRUE(Contains(status, "Could not satisfy device specification"));
}

TEST(ValidateFunctionDevicesTest, LocalDeviceNameInvalidIdFails) {
  FunctionDef fdef = MakeFunctionWithDevice("add", "AddV2", "CPU:99");
  absl::Status status =
      ValidateFunctionDeviceConstraints(fdef, LocalCpuDevices());
  EXPECT_TRUE(IsInvalidArgument(status));
  EXPECT_TRUE(Contains(status, "Could not satisfy device specification"));
}

TEST(ValidateFunctionDevicesTest, MalformedDeviceSpecFails) {
  FunctionDef fdef = MakeFunctionWithDevice("add", "AddV2", "not_a_device");
  absl::Status status =
      ValidateFunctionDeviceConstraints(fdef, LocalCpuDevices());
  EXPECT_TRUE(IsInvalidArgument(status));
  EXPECT_TRUE(Contains(status, "Malformed device specification"));
  EXPECT_TRUE(Contains(status, "not_a_device"));
}

TEST(ValidateFunctionDevicesTest, ErrorMentionsOperation) {
  FunctionDef fdef = MakeFunctionWithDevice("my_add", "AddV2", "/job:worker");
  absl::Status status =
      ValidateFunctionDeviceConstraints(fdef, LocalCpuDevices());
  EXPECT_TRUE(Contains(status, "my_add"));
  EXPECT_TRUE(Contains(status, "AddV2"));
}

TEST(ValidateFunctionDevicesTest, EmptyAvailableDevicesRejectsConstraint) {
  FunctionDef fdef = MakeFunctionWithDevice("add", "AddV2", "/device:CPU:0");
  absl::Status status = ValidateFunctionDeviceConstraints(fdef, {});
  EXPECT_TRUE(IsInvalidArgument(status));
  EXPECT_TRUE(Contains(status, "Could not satisfy device specification"));
}

}  // namespace tensorflow
