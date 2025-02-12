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
#include "absl/time/time.h"
#include "xla/tsl/lib/core/status_test_util.h"
#include "xla/tsl/platform/env.h"
#include "xla/tsl/platform/status.h"
#include "tensorflow/core/common_runtime/device.h"
#include "tensorflow/core/common_runtime/device_factory.h"
#include "tensorflow/core/common_runtime/device_set.h"
#include "tensorflow/core/common_runtime/function_testlib.h"
#include "tensorflow/core/common_runtime/optimized_function_graph_info.h"
#include "tensorflow/core/framework/metrics.h"
#include "tensorflow/core/platform/test.h"
#include "tensorflow/core/public/session_options.h"

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

void TestOptimizeFunctionGraphWithFunctionNotFound(bool load_from_cache) {
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
  absl::StatusOr<OptimizedFunctionGraphInfo> optimized_function_graph_info;
  if (load_from_cache) {
    optimized_function_graph_info = OptimizeFunctionGraphOrReadFromFileCache(
        "FindDevice", {}, opts, device_set, lib_def.get(),
        /*composite_devices=*/{}, devices[0].get(), devices[0].get(),
        Env::Default(), absl::ZeroDuration());
  } else {
    optimized_function_graph_info = OptimizeFunctionGraph(
        "FindDevice", {}, opts, device_set, lib_def.get(),
        /*composite_devices=*/{}, devices[0].get(), devices[0].get(),
        Env::Default(), OptimizedFunctionGraph::AOT);
  }
  EXPECT_TRUE(absl::IsInvalidArgument(optimized_function_graph_info.status()))
      << "Actual status: " << optimized_function_graph_info.status();
  EXPECT_TRUE(
      absl::StrContains(optimized_function_graph_info.status().message(),
                        "Failed to find function"))
      << "Actual error message: "
      << optimized_function_graph_info.status().message();
}

TEST(OptimizeFunctionGraphTest,
     OptimizeFunctionGraphReturnsErrorIfNoFunctionFound) {
  TestOptimizeFunctionGraphWithFunctionNotFound(/*load_from_cache=*/false);
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

  const absl::StatusOr<OptimizedFunctionGraphInfo> aot_result =
      OptimizeFunctionGraph("FindDevice", {}, opts, device_set, lib_def.get(),
                            /*composite_devices=*/{}, devices[0].get(),
                            devices[1].get(), Env::Default(),
                            OptimizedFunctionGraph::AOT);
  TF_EXPECT_OK(aot_result.status());
  EXPECT_EQ(aot_result->name, "FindDevice");
  // FindDevice function has one return node.
  EXPECT_EQ(aot_result->num_return_nodes, 1);
  // Return node type is string.
  EXPECT_THAT(aot_result->ret_types, ElementsAre(DT_STRING));
  EXPECT_GT(aot_result->optimization_duration_usecs, 0);
  EXPECT_EQ(aot_result->optimization_source, OptimizedFunctionGraph::AOT);
}

TEST(OptimizeFunctionGraphTest, ReloadFromCacheReturnsErrorIfNoFunctionFound) {
  TestOptimizeFunctionGraphWithFunctionNotFound(/*load_from_cache=*/true);
}

TEST(OptimizeFunctionGraphTest, OptimizeFunctionGraphAndWriteToCache) {
  Env* env = Env::Default();

  // Create a temp directory and set to env variable for the purpose of testing.
  const string temp_dir = "/tmp/testing_cache_direcroty";
  EXPECT_TRUE(env->RecursivelyCreateDir(temp_dir).ok());
  setenv(kGraphCachingEnvVariableName, temp_dir.c_str(), 1);

  // Check that no file exists before caching.
  std::vector<string> empty_file_list;
  TF_ASSERT_OK(
      env->GetMatchingPaths(absl::StrCat(temp_dir, "/*"), &empty_file_list));
  ASSERT_TRUE(empty_file_list.empty());

  // Setup InstantiateOptions, FunctionLibraryDefinition, and devices.
  FunctionLibraryRuntime::InstantiateOptions opts;
  opts.is_multi_device_function = true;
  FunctionDefLibrary proto;
  *(proto.add_function()) = test::function::FindDeviceWithUuid();
  auto lib_def =
      std::make_unique<FunctionLibraryDefinition>(OpRegistry::Global(), proto);
  std::vector<std::unique_ptr<Device>> devices;
  CreateCpuDeviceList(kDevicePrefix, 3, devices);
  DeviceSet device_set;
  for (const auto& device : devices) {
    device_set.AddDevice(device.get());
  }

  EXPECT_EQ(metrics::GetFunctionGraphOptimizationSavingTimeUsecs(
                metrics::GraphOptimizationSource::kJit),
            0);
  EXPECT_EQ(metrics::GetFunctionGraphOptimizationCacheHitCount(
                metrics::GraphOptimizationSource::kJit),
            0);
  EXPECT_EQ(metrics::GetFunctionGraphOptimizationCacheMissCount(
                metrics::GraphOptimizationSource::kJit),
            0);

  // Expect no caching with an extremely high caching threshold.
  absl::StatusOr<OptimizedFunctionGraphInfo> optimized_info =
      OptimizeFunctionGraphOrReadFromFileCache(
          "FindDevice_1234", {}, opts, device_set, lib_def.get(),
          /*composite_devices=*/{}, devices[0].get(), devices[1].get(),
          Env::Default(), /*caching_threshold_duration=*/absl::Hours(48));
  TF_ASSERT_OK(optimized_info.status());
  std::vector<string> file_list;
  TF_ASSERT_OK(env->GetMatchingPaths(absl::StrCat(temp_dir, "/*"), &file_list));
  EXPECT_EQ(file_list.size(), 0);
  EXPECT_EQ(metrics::GetFunctionGraphOptimizationSavingTimeUsecs(
                metrics::GraphOptimizationSource::kJit),
            0);
  EXPECT_EQ(metrics::GetFunctionGraphOptimizationCacheHitCount(
                metrics::GraphOptimizationSource::kJit),
            0);
  EXPECT_EQ(metrics::GetFunctionGraphOptimizationCacheMissCount(
                metrics::GraphOptimizationSource::kJit),
            1);

  // Expect one file cache with zero caching threshold duration.
  optimized_info = OptimizeFunctionGraphOrReadFromFileCache(
      "FindDevice_1234", {}, opts, device_set, lib_def.get(),
      /*composite_devices=*/{}, devices[0].get(), devices[1].get(),
      Env::Default(), /*caching_threshold_duration=*/absl::ZeroDuration());
  TF_ASSERT_OK(optimized_info.status());
  // Check that only one cache file exists.
  file_list.clear();
  TF_ASSERT_OK(env->GetMatchingPaths(
      absl::StrCat(temp_dir, "/_-1_FindDevice_1"), &file_list));
  EXPECT_EQ(file_list.size(), 1);
  EXPECT_EQ(metrics::GetFunctionGraphOptimizationSavingTimeUsecs(
                metrics::GraphOptimizationSource::kJit),
            0);
  EXPECT_EQ(metrics::GetFunctionGraphOptimizationCacheHitCount(
                metrics::GraphOptimizationSource::kJit),
            0);
  EXPECT_EQ(metrics::GetFunctionGraphOptimizationCacheMissCount(
                metrics::GraphOptimizationSource::kJit),
            2);

  // Expect one file cache after running for the same function again.
  optimized_info = OptimizeFunctionGraphOrReadFromFileCache(
      "FindDevice_1234", {}, opts, device_set, lib_def.get(),
      /*composite_devices=*/{}, devices[0].get(), devices[1].get(),
      Env::Default(), /*caching_threshold_duration=*/absl::ZeroDuration());
  TF_ASSERT_OK(optimized_info.status());
  file_list.clear();
  TF_ASSERT_OK(env->GetMatchingPaths(
      absl::StrCat(temp_dir, "/_-1_FindDevice_1"), &file_list));
  EXPECT_EQ(file_list.size(), 1);
  EXPECT_GT(metrics::GetFunctionGraphOptimizationSavingTimeUsecs(
                metrics::GraphOptimizationSource::kJit),
            0);
  EXPECT_EQ(metrics::GetFunctionGraphOptimizationCacheHitCount(
                metrics::GraphOptimizationSource::kJit),
            1);
  EXPECT_EQ(metrics::GetFunctionGraphOptimizationCacheMissCount(
                metrics::GraphOptimizationSource::kJit),
            2);
  EXPECT_EQ(optimized_info->name, "FindDevice_1234");
  EXPECT_EQ(optimized_info->num_return_nodes, 1);
  EXPECT_THAT(optimized_info->ret_types, ElementsAre(DT_STRING));

  // Clean up the cache directory for cases when the test is run multiple times
  // in a row without clearing the filesystem where the test is running.
  int64_t undeleted_files;
  int64_t undeleted_dirs;

  TF_EXPECT_OK(
      env->DeleteRecursively(temp_dir, &undeleted_files, &undeleted_dirs));

  // Check that the caching files have been cleaned up.
  EXPECT_EQ(undeleted_files, 0);
  EXPECT_EQ(undeleted_dirs, 0);
  TF_ASSERT_OK(
      env->GetMatchingPaths(absl::StrCat(temp_dir, "/*"), &empty_file_list));
  ASSERT_TRUE(empty_file_list.empty());
}

}  // namespace
}  // namespace tensorflow
