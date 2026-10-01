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

#include <memory>
#include <string>
#include <vector>

#include <gmock/gmock.h>
#include "absl/status/status.h"
#include "tensorflow/core/framework/attr_value.pb.h"
#include "tensorflow/core/framework/device.h"
#include "tensorflow/core/framework/device_attributes.pb.h"
#include "tensorflow/core/framework/function.h"
#include "tensorflow/core/framework/function.pb.h"
#include "tensorflow/core/framework/node_def.pb.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/platform/status_matchers.h"
#include "tensorflow/core/platform/test.h"

namespace tensorflow {

namespace {

// A fake device that has specific device attributes, used to simulate the
// presence of a device without depending on the real runtime.
class FakeDevice : public Device {
 private:
  explicit FakeDevice(const DeviceAttributes& device_attributes)
      : Device(nullptr, device_attributes) {}

 public:
  absl::Status Sync() override {
    return absl::UnimplementedError("FakeDevice::Sync()");
  }

  Allocator* GetAllocator(AllocatorAttributes attr) override { return nullptr; }

  static std::unique_ptr<Device> Make(const std::string& name) {
    DeviceAttributes device_attributes;
    device_attributes.set_name(name);
    device_attributes.set_device_type("FakeCPU");
    return std::unique_ptr<Device>(new FakeDevice(device_attributes));
  }
};

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

// A FunctionDef with a single node of type `op` carrying one function-bearing
// attribute named `attr_name` pointing at `func_name`.
FunctionDef MakeFunctionWithFuncAttr(const std::string& node_name,
                                     const std::string& op,
                                     const std::string& attr_name,
                                     const std::string& func_name) {
  FunctionDef fdef;
  NodeDef* node = fdef.add_node_def();
  node->set_name(node_name);
  node->set_op(op);
  AttrValue func_attr;
  func_attr.mutable_func()->set_name(func_name);
  (*node->mutable_attr())[attr_name] = func_attr;
  return fdef;
}

// A FunctionDef with a single node carrying a list-of-functions attribute.
FunctionDef MakeFunctionWithFuncListAttr(
    const std::string& node_name, const std::string& op,
    const std::string& attr_name, const std::vector<std::string>& names) {
  FunctionDef fdef;
  NodeDef* node = fdef.add_node_def();
  node->set_name(node_name);
  node->set_op(op);
  AttrValue list_attr;
  for (const std::string& name : names) {
    NameAttrList* func = list_attr.mutable_list()->add_func();
    func->set_name(name);
  }
  (*node->mutable_attr())[attr_name] = list_attr;
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

class ValidateFunctionDevicesTest : public ::testing::Test {
 protected:
  void SetUp() override {
    devices_.push_back(
        FakeDevice::Make("/job:localhost/replica:0/task:0/device:CPU:0"));
    available_devices_.push_back(devices_.back().get());
  }

  // The unique_ptr owns the devices; available_devices_ holds raw pointers
  // for the duration of the test.
  std::vector<std::unique_ptr<Device>> devices_;
  std::vector<Device*> available_devices_;
};

}  // namespace

TEST_F(ValidateFunctionDevicesTest, EmptyFunctionIsOk) {
  FunctionDef fdef;
  EXPECT_THAT(
      ValidateFunctionDeviceConstraints(fdef, nullptr, available_devices_),
      ::tensorflow::testing::IsOk());
}

TEST_F(ValidateFunctionDevicesTest, NoDeviceConstraintIsOk) {
  FunctionDef fdef = MakeFunctionWithDevice("add", "AddV2", "");
  EXPECT_THAT(
      ValidateFunctionDeviceConstraints(fdef, nullptr, available_devices_),
      ::tensorflow::testing::IsOk());
}

TEST_F(ValidateFunctionDevicesTest, FullySpecifiedDeviceIsOk) {
  FunctionDef fdef = MakeFunctionWithDevice(
      "add", "AddV2", "/job:localhost/replica:0/task:0/device:CPU:0");
  EXPECT_THAT(
      ValidateFunctionDeviceConstraints(fdef, nullptr, available_devices_),
      ::tensorflow::testing::IsOk());
}

TEST_F(ValidateFunctionDevicesTest, PartialJobConstraintIsOk) {
  // '/job:localhost' is a partial constraint without a device type; it must
  // be satisfied by the fully-specified available CPU device.
  FunctionDef fdef = MakeFunctionWithDevice("add", "AddV2", "/job:localhost");
  EXPECT_THAT(
      ValidateFunctionDeviceConstraints(fdef, nullptr, available_devices_),
      ::tensorflow::testing::IsOk());
}

TEST_F(ValidateFunctionDevicesTest, PartialTaskConstraintIsOk) {
  FunctionDef fdef = MakeFunctionWithDevice("add", "AddV2", "/task:0");
  EXPECT_THAT(
      ValidateFunctionDeviceConstraints(fdef, nullptr, available_devices_),
      ::tensorflow::testing::IsOk());
}

TEST_F(ValidateFunctionDevicesTest, LocalDeviceNameIsOk) {
  FunctionDef fdef = MakeFunctionWithDevice("add", "AddV2", "CPU:0");
  EXPECT_THAT(
      ValidateFunctionDeviceConstraints(fdef, nullptr, available_devices_),
      ::tensorflow::testing::IsOk());
}

TEST_F(ValidateFunctionDevicesTest, UnsatisfiedJobFails) {
  FunctionDef fdef = MakeFunctionWithDevice("add", "AddV2", "/job:worker");
  absl::Status status =
      ValidateFunctionDeviceConstraints(fdef, nullptr, available_devices_);
  EXPECT_THAT(status, ::tensorflow::testing::StatusIs(
                          absl::StatusCode::kInvalidArgument,
                          ::testing::AllOf(
                              ::testing::HasSubstr(
                                  "Could not satisfy device specification"),
                              ::testing::HasSubstr("/job:worker"),
                              ::testing::HasSubstr("/job:localhost/replica:0/"
                                                   "task:0/device:CPU:0"))));
}

TEST_F(ValidateFunctionDevicesTest, InvalidDeviceIdFails) {
  // The device type exists, but the id does not: the partial constraint must
  // still be rejected.
  FunctionDef fdef = MakeFunctionWithDevice("add", "AddV2", "/device:CPU:99");
  absl::Status status =
      ValidateFunctionDeviceConstraints(fdef, nullptr, available_devices_);
  EXPECT_THAT(status, ::tensorflow::testing::StatusIs(
                          absl::StatusCode::kInvalidArgument,
                          ::testing::HasSubstr(
                              "Could not satisfy device specification")));
}

TEST_F(ValidateFunctionDevicesTest, InvalidDeviceTypeFails) {
  FunctionDef fdef =
      MakeFunctionWithDevice("add", "AddV2", "/device:NONEXISTENT:0");
  absl::Status status =
      ValidateFunctionDeviceConstraints(fdef, nullptr, available_devices_);
  EXPECT_THAT(status, ::tensorflow::testing::StatusIs(
                          absl::StatusCode::kInvalidArgument,
                          ::testing::HasSubstr(
                              "Could not satisfy device specification")));
}

TEST_F(ValidateFunctionDevicesTest, LocalDeviceNameInvalidIdFails) {
  FunctionDef fdef = MakeFunctionWithDevice("add", "AddV2", "CPU:99");
  absl::Status status =
      ValidateFunctionDeviceConstraints(fdef, nullptr, available_devices_);
  EXPECT_THAT(status, ::tensorflow::testing::StatusIs(
                          absl::StatusCode::kInvalidArgument,
                          ::testing::HasSubstr(
                              "Could not satisfy device specification")));
}

TEST_F(ValidateFunctionDevicesTest, MalformedDeviceSpecFails) {
  FunctionDef fdef = MakeFunctionWithDevice("add", "AddV2", "not_a_device");
  absl::Status status =
      ValidateFunctionDeviceConstraints(fdef, nullptr, available_devices_);
  EXPECT_THAT(
      status, ::tensorflow::testing::StatusIs(
                  absl::StatusCode::kInvalidArgument,
                  ::testing::AllOf(
                      ::testing::HasSubstr("Malformed device specification"),
                      ::testing::HasSubstr("not_a_device"))));
}

TEST_F(ValidateFunctionDevicesTest, ErrorMentionsOperation) {
  FunctionDef fdef = MakeFunctionWithDevice("my_add", "AddV2", "/job:worker");
  absl::Status status =
      ValidateFunctionDeviceConstraints(fdef, nullptr, available_devices_);
  EXPECT_THAT(status, ::tensorflow::testing::StatusIs(
                          absl::StatusCode::kInvalidArgument,
                          ::testing::AllOf(::testing::HasSubstr("my_add"),
                                           ::testing::HasSubstr("AddV2"))));
}

TEST_F(ValidateFunctionDevicesTest, EmptyAvailableDevicesRejectsConstraint) {
  FunctionDef fdef = MakeFunctionWithDevice("add", "AddV2", "/device:CPU:0");
  std::vector<Device*> no_devices;
  absl::Status status =
      ValidateFunctionDeviceConstraints(fdef, /*flib_def=*/nullptr, no_devices);
  EXPECT_THAT(status, ::tensorflow::testing::StatusIs(
                          absl::StatusCode::kInvalidArgument,
                          ::testing::HasSubstr(
                              "Could not satisfy device specification")));
}

TEST_F(ValidateFunctionDevicesTest, NullFunctionLibraryOnlyChecksTopLevel) {
  // With a nullptr library, nested functions cannot be inspected: only the
  // top-level nodes are validated, and the validation passes.
  FunctionDef fdef =
      MakeFunctionWithFuncAttr("call", "PartitionedCall", "f", "inner_func");
  EXPECT_THAT(
      ValidateFunctionDeviceConstraints(fdef, /*flib_def=*/nullptr,
                                        available_devices_),
      ::tensorflow::testing::IsOk());
}

TEST_F(ValidateFunctionDevicesTest, NestedFunctionInvalidDeviceFails) {
  // The top-level function has no constraints, but the nested function called
  // through PartitionedCall places its node on a nonexistent device.
  FunctionDef fdef =
      MakeFunctionWithFuncAttr("call", "PartitionedCall", "f", "inner_func");
  FunctionLibraryDefinition flib_def = MakeFlibDefWithInnerFunction(
      OpRegistry::Global(), "inner_func", "/device:NONEXISTENT:0");
  absl::Status status =
      ValidateFunctionDeviceConstraints(fdef, &flib_def, available_devices_);
  EXPECT_THAT(status, ::tensorflow::testing::StatusIs(
                          absl::StatusCode::kInvalidArgument,
                          ::testing::HasSubstr(
                              "Could not satisfy device specification")));
}

TEST_F(ValidateFunctionDevicesTest, NestedFunctionValidDeviceIsOk) {
  FunctionDef fdef =
      MakeFunctionWithFuncAttr("call", "PartitionedCall", "f", "inner_func");
  FunctionLibraryDefinition flib_def = MakeFlibDefWithInnerFunction(
      OpRegistry::Global(), "inner_func", "/job:localhost");
  EXPECT_THAT(
      ValidateFunctionDeviceConstraints(fdef, &flib_def, available_devices_),
      ::tensorflow::testing::IsOk());
}

TEST_F(ValidateFunctionDevicesTest, NestedFunctionCycleTerminates) {
  // Two functions calling each other in a cycle must not hang or crash; the
  // visited-functions guard ends the traversal.
  FunctionLibraryDefinition flib_def = MakeFlibDefWithInnerFunction(
      OpRegistry::Global(), "inner_func", "/job:localhost");
  FunctionDef fdef = MakeFunctionWithFuncAttr("call", "StatefulPartitionedCall",
                                              "f", "inner_func");
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
      ValidateFunctionDeviceConstraints(cyclic, &flib_def, available_devices_),
      ::tensorflow::testing::IsOk());
}

TEST_F(ValidateFunctionDevicesTest, NestedIfBranchInvalidDeviceFails) {
  // A tf.cond lowers to an If op whose branches live in the 'then_branch'
  // and 'else_branch' attributes; functions invoked there must be validated.
  FunctionDef fdef = MakeFunctionWithFuncAttr("cond", "If", "then_branch",
                                              "then_func");
  FunctionLibraryDefinition flib_def = MakeFlibDefWithInnerFunction(
      OpRegistry::Global(), "then_func", "/device:NONEXISTENT:0");
  absl::Status status =
      ValidateFunctionDeviceConstraints(fdef, &flib_def, available_devices_);
  EXPECT_THAT(status, ::tensorflow::testing::StatusIs(
                          absl::StatusCode::kInvalidArgument,
                          ::testing::HasSubstr(
                              "Could not satisfy device specification")));
}

TEST_F(ValidateFunctionDevicesTest, NestedWhileBodyInvalidDeviceFails) {
  // A tf.while_loop lowers to a While op with 'cond' and 'body' attributes;
  // the body function's constraints must be validated too.
  FunctionDef fdef =
      MakeFunctionWithFuncAttr("loop", "While", "body", "body_func");
  FunctionLibraryDefinition flib_def = MakeFlibDefWithInnerFunction(
      OpRegistry::Global(), "body_func", "/device:NONEXISTENT:0");
  absl::Status status =
      ValidateFunctionDeviceConstraints(fdef, &flib_def, available_devices_);
  EXPECT_THAT(status, ::tensorflow::testing::StatusIs(
                          absl::StatusCode::kInvalidArgument,
                          ::testing::HasSubstr(
                              "Could not satisfy device specification")));
}

TEST_F(ValidateFunctionDevicesTest, NestedIfBranchValidDeviceIsOk) {
  FunctionDef fdef = MakeFunctionWithFuncAttr("cond", "If", "else_branch",
                                              "else_func");
  FunctionLibraryDefinition flib_def = MakeFlibDefWithInnerFunction(
      OpRegistry::Global(), "else_func", "/job:localhost");
  EXPECT_THAT(
      ValidateFunctionDeviceConstraints(fdef, &flib_def, available_devices_),
      ::tensorflow::testing::IsOk());
}

TEST_F(ValidateFunctionDevicesTest, NestedFuncListInvalidDeviceFails) {
  // Case-style ops carry their branches in a list(attr) attribute; every
  // function in the list must be validated. Names that match no function in
  // the library are ignored.
  FunctionDef fdef = MakeFunctionWithFuncListAttr(
      "switch_case", "StatelessCase", "branches",
      {"first_func", "second_func"});
  FunctionDefLibrary lib_def;
  FunctionDef* second = lib_def.add_function();
  second->mutable_signature()->set_name("second_func");
  NodeDef* node = second->add_node_def();
  node->set_name("add");
  node->set_op("AddV2");
  node->set_device("/device:NONEXISTENT:0");
  FunctionLibraryDefinition flib_def(OpRegistry::Global(), lib_def);
  absl::Status status =
      ValidateFunctionDeviceConstraints(fdef, &flib_def, available_devices_);
  EXPECT_THAT(status, ::tensorflow::testing::StatusIs(
                          absl::StatusCode::kInvalidArgument,
                          ::testing::HasSubstr(
                              "Could not satisfy device specification")));
}

TEST_F(ValidateFunctionDevicesTest, DeviceAttributesOverloadMatchesDevice) {
  // The DeviceAttributes overload exists for callers holding protos; it must
  // behave the same as the Device-based one.
  FunctionDef fdef =
      MakeFunctionWithDevice("add", "AddV2", "/device:NONEXISTENT:0");
  DeviceAttributes attrs;
  attrs.set_name("/job:localhost/replica:0/task:0/device:CPU:0");
  attrs.set_device_type("CPU");
  std::vector<DeviceAttributes> available_devices = {attrs};
  absl::Status status =
      ValidateFunctionDeviceConstraints(fdef, nullptr, available_devices);
  EXPECT_THAT(status, ::tensorflow::testing::StatusIs(
                          absl::StatusCode::kInvalidArgument,
                          ::testing::HasSubstr(
                              "Could not satisfy device specification")));
}

}  // namespace tensorflow
