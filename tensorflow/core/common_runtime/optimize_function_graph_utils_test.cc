/* Copyright 2022 The TensorFlow Authors. All Rights Reserved.

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
#include "tensorflow/core/common_runtime/optimize_function_graph_utils.h"

#include <memory>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/strings/string_view.h"
#include "tensorflow/core/common_runtime/device.h"
#include "tensorflow/core/common_runtime/device_factory.h"
#include "tensorflow/core/common_runtime/device_set.h"
#include "tensorflow/core/common_runtime/function_testlib.h"
#include "tensorflow/core/platform/test.h"
#include "tensorflow/core/public/session_options.h"
#include "tensorflow/tsl/lib/core/status_test_util.h"
#include "tensorflow/tsl/platform/status.h"

namespace tensorflow {
namespace {
using ::testing::ElementsAre;

constexpr absl::string_view kDevicePrefix = "/job:a/replica:0/task:0/device:";

// Creates a vector of `num_devices` CPU deivces with prefix as `name_prefix` in
// output `devices`.
void CreateCpuDeviceList(absl::string_view name_prefix, int num_devices,
                         std::vector<std::unique_ptr<Device>>& devices) {
  SessionOptions options;
  auto* device_count = options.config.mutable_device_count();
  device_count->insert({"CPU", num_devices});
  TF_ASSERT_OK(
      DeviceFactory::AddDevices(options, "/job:a/replica:0/task:0", &devices));
}

TEST(OptimizeFunctionGraphTest,
     OptimizeFunctionGraphReturnsErrorIfNoFunctionFound) {
  FunctionLibraryRuntime::InstantiateOptions opts;
  opts.is_multi_device_function = true;
  auto lib_def =
      std::make_unique<FunctionLibraryDefinition>(OpRegistry::Global());

  std::vector<std::unique_ptr<Device>> devices;
  CreateCpuDeviceList(kDevicePrefix, 1, devices);
  DeviceSet device_set;
  for (const auto& device : devices) {
    device_set.AddDevice(device.get());
  }

  // Try to optimize a function called "FindDevice" which does not exist in
  // library.
  const StatusOr<OptimizedFunctionGraphInfo> aot_result =
      OptimizeFunctionGraph("FindDevice", {}, opts, device_set, lib_def.get(),
                            /*composite_devices=*/{}, devices[0].get(),
                            devices[0].get(), Env::Default());
  EXPECT_TRUE(errors::IsInvalidArgument(aot_result.status()))
      << "Actual status: " << aot_result.status();
  EXPECT_TRUE(absl::StrContains(aot_result.status().error_message(),
                                "Failed to find function"))
      << "Actual error message: " << aot_result.status().error_message();
}

TEST(OptimizeFunctionGraphTest, OptimizeFunctionGraphReturnsCorrectResult) {
  FunctionLibraryRuntime::InstantiateOptions opts;
  opts.is_multi_device_function = true;

  // Create a function library with a trivial function `FindDevice` which has
  // one string output.
  FunctionDefLibrary proto;
  *(proto.add_function()) = test::function::FindDevice();
  auto lib_def =
      std::make_unique<FunctionLibraryDefinition>(OpRegistry::Global(), proto);

  std::vector<std::unique_ptr<Device>> devices;
  CreateCpuDeviceList(kDevicePrefix, 3, devices);
  DeviceSet device_set;
  for (const auto& device : devices) {
    device_set.AddDevice(device.get());
  }

  const StatusOr<OptimizedFunctionGraphInfo> aot_result =
      OptimizeFunctionGraph("FindDevice", {}, opts, device_set, lib_def.get(),
                            /*composite_devices=*/{}, devices[0].get(),
                            devices[1].get(), Env::Default());
  TF_EXPECT_OK(aot_result.status());
  EXPECT_EQ(aot_result->name, "FindDevice");
  // FindDevice function has one return node.
  EXPECT_EQ(aot_result->num_return_nodes, 1);
  // Return node type is string.
  EXPECT_THAT(aot_result->ret_types, ElementsAre(DT_STRING));
}

}  // namespace
}  // namespace tensorflow
