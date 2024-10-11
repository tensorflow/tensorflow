// Copyright 2024 Google LLC.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "tensorflow/lite/experimental/lrt/core/compiler_plugin_api.h"

#include <gtest/gtest.h>
#include "absl/strings/string_view.h"
#include "tensorflow/lite/experimental/lrt/core/dynamic_loading.h"
#include "tensorflow/lite/experimental/lrt/test/common.h"
#include "tensorflow/lite/experimental/lrt/vendors/c/lite_rt_compiler_plugin.h"

namespace {

using ::lrt::internal::LrtPluginApi;
using ::lrt::internal::ResolvePluginApi;

constexpr absl::string_view kTestPluginPath =
    "third_party/tensorflow/lite/experimental/lrt/vendors/examples/"
    "libLrtPlugin_ExampleSocManufacturer_ExampleSocModel.so";

TEST(TestCompilerPluginApi, TestResolveApi) {
  void* lib_handle = nullptr;
  ASSERT_STATUS_OK(lrt::OpenLib(kTestPluginPath, &lib_handle));

  LrtPluginApi api;
  ASSERT_STATUS_OK(ResolvePluginApi(lib_handle, api));

  EXPECT_NE(api.init, nullptr);
  EXPECT_NE(api.destroy, nullptr);

  EXPECT_NE(api.soc_manufacturer, nullptr);
  EXPECT_NE(api.num_supported_models, nullptr);
  EXPECT_NE(api.get_supported_soc_model, nullptr);

  EXPECT_NE(api.partition_model, nullptr);
  EXPECT_NE(api.compile, nullptr);

  EXPECT_NE(api.compiled_result_destroy, nullptr);
  EXPECT_NE(api.compiled_result_get_byte_code, nullptr);
  EXPECT_NE(api.compiled_result_get_num_calls, nullptr);
  EXPECT_NE(api.compiled_result_get_call_info, nullptr);

  ASSERT_STREQ(api.soc_manufacturer(), "ExampleSocManufacturer");

  LrtCompilerPlugin plugin;
  ASSERT_STATUS_OK(api.init(&plugin));
  ASSERT_EQ(api.num_supported_models(plugin), 1);

  const char* soc_model;
  ASSERT_STATUS_OK(api.get_supported_soc_model(plugin, 0, &soc_model));
  EXPECT_STREQ(soc_model, "ExampleSocModel");

  api.destroy(plugin);
}

}  // namespace
