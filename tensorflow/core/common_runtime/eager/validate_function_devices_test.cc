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

#include <gmock/gmock.h>
#include "absl/status/status.h"
#include "tensorflow/core/framework/attr_value.pb.h"
#include "tensorflow/core/framework/device_attributes.pb.h"
#include "tensorflow/core/framework/function.h"
#include "tensorflow/core/framework/function.pb.h"
#include "tensorflow/core/framework/node_def.pb.h"
#include "tensorflow/core/framework/op.h"
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

// A FunctionDef containing a single PartitionedCall node that invokes the
// given nested function name through its 'f' attribute.
FunctionDef MakeFunctionWithFuncNode(const std::string& node_name,
                                     const std::string& func_name) {
  FunctionDef fdef;
  NodeDef* node = fdef.add_node_def();
  node->set_name(node_name);
  node->set_op("PartitionedCall");
  AttrValue func_attr;
  func_attr.mutable_func()->set_name(func_name);
  (*node->mutable_attr())[FunctionLibraryDefinition::kFuncAttr] = func_attr;
  return fdef;
}

// A FunctionLibraryDefinition holding one inner function whose single node
// carries the given device constraint.
FunctionLibraryDefinition MakeFlibDefWithInnerFunction(
    const OpRegistryInterface* registry, const std::string& inner_name,
    const std::string& device) {
  FunctionDefLibrary lib_def;
  FunctionDef* inner = lib_def.add_function();
  inner->mutable_signature()->set_name(inner_name);
  NodeDef* node = inner->add_node_def();
  node->set_name("add");
  node->set_op("AddV2");
  node->set_device(device);
  FunctionLibraryDefinition flib_def(registry, lib_def);
  return flib_def;
}

}  // namespace

TEST(ValidateFunctionDevicesTest, EmptyFunctionIsOk) {
  FunctionDef fdef;
  EXPECT_THAT(ValidateFunctionDeviceConstraints(fdef, nullptr, LocalCpuDevices()),
              ::tensorflow::testing::IsOk());
}

TEST(ValidateFunctionDevicesTest, NoDeviceConstraintIsOk) {
  FunctionDef fdef = MakeFunctionWithDevice("add", "AddV2", "");
  EXPECT_THAT(ValidateFunctionDeviceConstraints(fdef, nullptr, LocalCpuDevices()),
              ::tensorflow::testing::IsOk());
}

TEST(ValidateFunctionDevicesTest, FullySpecifiedDeviceIsOk) {
  FunctionDef fdef = MakeFunctionWithDevice(
      "add", "AddV2", "/job:localhost/replica:0/task:0/device:CPU:0");
  EXPECT_THAT(ValidateFunctionDeviceConstraints(fdef, nullptr, LocalCpuDevices()),
              ::tensorflow::testing::IsOk());
}

TEST(ValidateFunctionDevicesTest, PartialJobConstraintIsOk) {
  // '/job:localhost' is a partial constraint without a device type; it must
  // be satisfied by the fully-specified available CPU device.
  FunctionDef fdef = MakeFunctionWithDevice("add", "AddV2", "/job:localhost");
  EXPECT_THAT(ValidateFunctionDeviceConstraints(fdef, nullptr, LocalCpuDevices()),
              ::tensorflow::testing::IsOk());
}

TEST(ValidateFunctionDevicesTest, PartialTaskConstraintIsOk) {
  FunctionDef fdef = MakeFunctionWithDevice("add", "AddV2", "/task:0");
  EXPECT_THAT(ValidateFunctionDeviceConstraints(fdef, nullptr, LocalCpuDevices()),
              ::tensorflow::testing::IsOk());
}

TEST(ValidateFunctionDevicesTest, LocalDeviceNameIsOk) {
  FunctionDef fdef = MakeFunctionWithDevice("add", "AddV2", "CPU:0");
  EXPECT_THAT(ValidateFunctionDeviceConstraints(fdef, nullptr, LocalCpuDevices()),
              ::tensorflow::testing::IsOk());
}

TEST(ValidateFunctionDevicesTest, UnsatisfiedJobFails) {
  FunctionDef fdef = MakeFunctionWithDevice("add", "AddV2", "/job:worker");
  absl::Status status =
      ValidateFunctionDeviceConstraints(fdef, nullptr, LocalCpuDevices());
  EXPECT_THAT(status, ::tensorflow::testing::StatusIs(
                          absl::StatusCode::kInvalidArgument,
                          ::testing::AllOf(
                              ::testing::HasSubstr(
                                  "Could not satisfy device specification"),
                              ::testing::HasSubstr("/job:worker"),
                              ::testing::HasSubstr("/job:localhost/replica:0/"
                                                   "task:0/device:CPU:0"))));
}

TEST(ValidateFunctionDevicesTest, InvalidDeviceIdFails) {
  // The device type exists, but the id does not: the partial constraint must
  // still be rejected.
  FunctionDef fdef = MakeFunctionWithDevice("add", "AddV2", "/device:CPU:99");
  absl::Status status =
      ValidateFunctionDeviceConstraints(fdef, nullptr, LocalCpuDevices());
  EXPECT_THAT(status, ::tensorflow::testing::StatusIs(
                          absl::StatusCode::kInvalidArgument,
                          ::testing::HasSubstr(
                              "Could not satisfy device specification")));
}

TEST(ValidateFunctionDevicesTest, InvalidDeviceTypeFails) {
  FunctionDef fdef =
      MakeFunctionWithDevice("add", "AddV2", "/device:NONEXISTENT:0");
  absl::Status status =
      ValidateFunctionDeviceConstraints(fdef, nullptr, LocalCpuDevices());
  EXPECT_THAT(status, ::tensorflow::testing::StatusIs(
                          absl::StatusCode::kInvalidArgument,
                          ::testing::HasSubstr(
                              "Could not satisfy device specification")));
}

TEST(ValidateFunctionDevicesTest, LocalDeviceNameInvalidIdFails) {
  FunctionDef fdef = MakeFunctionWithDevice("add", "AddV2", "CPU:99");
  absl::Status status =
      ValidateFunctionDeviceConstraints(fdef, nullptr, LocalCpuDevices());
  EXPECT_THAT(status, ::tensorflow::testing::StatusIs(
                          absl::StatusCode::kInvalidArgument,
                          ::testing::HasSubstr(
                              "Could not satisfy device specification")));
}

TEST(ValidateFunctionDevicesTest, MalformedDeviceSpecFails) {
  FunctionDef fdef = MakeFunctionWithDevice("add", "AddV2", "not_a_device");
  absl::Status status =
      ValidateFunctionDeviceConstraints(fdef, nullptr, LocalCpuDevices());
  EXPECT_THAT(
      status, ::tensorflow::testing::StatusIs(
                  absl::StatusCode::kInvalidArgument,
                  ::testing::AllOf(
                      ::testing::HasSubstr("Malformed device specification"),
                      ::testing::HasSubstr("not_a_device"))));
}

TEST(ValidateFunctionDevicesTest, ErrorMentionsOperation) {
  FunctionDef fdef = MakeFunctionWithDevice("my_add", "AddV2", "/job:worker");
  absl::Status status =
      ValidateFunctionDeviceConstraints(fdef, nullptr, LocalCpuDevices());
  EXPECT_THAT(status, ::tensorflow::testing::StatusIs(
                          absl::StatusCode::kInvalidArgument,
                          ::testing::AllOf(::testing::HasSubstr("my_add"),
                                           ::testing::HasSubstr("AddV2"))));
}

TEST(ValidateFunctionDevicesTest, EmptyAvailableDevicesRejectsConstraint) {
  FunctionDef fdef = MakeFunctionWithDevice("add", "AddV2", "/device:CPU:0");
  absl::Status status = ValidateFunctionDeviceConstraints(
      fdef, /*flib_def=*/nullptr, {});
  EXPECT_THAT(status, ::tensorflow::testing::StatusIs(
                          absl::StatusCode::kInvalidArgument,
                          ::testing::HasSubstr(
                              "Could not satisfy device specification")));
}

TEST(ValidateFunctionDevicesTest, NullFunctionLibraryOnlyChecksTopLevel) {
  // With a nullptr library, nested functions cannot be inspected: only the
  // top-level nodes are validated, and the validation passes.
  FunctionDef fdef = MakeFunctionWithFuncNode("call", "inner_func");
  EXPECT_THAT(
      ValidateFunctionDeviceConstraints(
          fdef, /*flib_def=*/nullptr, LocalCpuDevices()),
      ::tensorflow::testing::IsOk());
}

TEST(ValidateFunctionDevicesTest, NestedFunctionInvalidDeviceFails) {
  // The top-level function has no constraints, but the nested function called
  // through PartitionedCall places its node on a nonexistent device.
  FunctionDef fdef = MakeFunctionWithFuncNode("call", "inner_func");
  FunctionLibraryDefinition flib_def = MakeFlibDefWithInnerFunction(
      OpRegistry::Global(), "inner_func", "/device:NONEXISTENT:0");
  absl::Status status = ValidateFunctionDeviceConstraints(
      fdef, &flib_def, LocalCpuDevices());
  EXPECT_THAT(status, ::tensorflow::testing::StatusIs(
                          absl::StatusCode::kInvalidArgument,
                          ::testing::HasSubstr(
                              "Could not satisfy device specification")));
}

TEST(ValidateFunctionDevicesTest, NestedFunctionValidDeviceIsOk) {
  FunctionDef fdef = MakeFunctionWithFuncNode("call", "inner_func");
  FunctionLibraryDefinition flib_def = MakeFlibDefWithInnerFunction(
      OpRegistry::Global(), "inner_func", "/job:localhost");
  EXPECT_THAT(
      ValidateFunctionDeviceConstraints(fdef, &flib_def, LocalCpuDevices()),
      ::tensorflow::testing::IsOk());
}

TEST(ValidateFunctionDevicesTest, NestedFunctionCycleTerminates) {
  // Two functions calling each other in a cycle must not hang or crash; the
  // visited-functions guard ends the traversal.
  FunctionLibraryDefinition flib_def = MakeFlibDefWithInnerFunction(
      OpRegistry::Global(), "inner_func", "/job:localhost");
  FunctionDef fdef = MakeFunctionWithFuncNode("call", "inner_func");
  // Make the inner function call itself, creating a cycle.
  const FunctionDef* inner = flib_def.Find("inner_func");
  ASSERT_NE(inner, nullptr);
  FunctionDef cyclic = *inner;
  NodeDef* call_node = cyclic.add_node_def();
  call_node->set_name("self_call");
  call_node->set_op("StatefulPartitionedCall");
  AttrValue func_attr;
  func_attr.mutable_func()->set_name("inner_func");
  (*call_node->mutable_attr())[FunctionLibraryDefinition::kFuncAttr] =
      func_attr;
  // Validate the cyclic function directly (the library copy stays valid so
  // Find() keeps returning the original definition).
  EXPECT_THAT(
      ValidateFunctionDeviceConstraints(cyclic, &flib_def, LocalCpuDevices()),
      ::tensorflow::testing::IsOk());
}

}  // namespace tensorflow
