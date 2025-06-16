/* Copyright 2024 The OpenXLA Authors.

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

#include "xla/pjrt/plugin/example_plugin/myplugin_c_pjrt.h"

#include <dirent.h>
#include <dlfcn.h>
#include <unistd.h>

#include <memory>
#include <string>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/log/log.h"
#include "absl/status/statusor.h"
#include "xla/pjrt/c/pjrt_c_api.h"
#include "xla/pjrt/pjrt_c_api_client.h"
#include "xla/pjrt/pjrt_client.h"
#include "xla/pjrt/plugin/dynamic_registration.h"
#include "tsl/platform/platform.h"

namespace {

TEST(MypluginCPjRtTest, CreatesPjRtAPI) {
  const PJRT_Api* myplugin = GetPjrtApi();
  EXPECT_THAT(myplugin, ::testing::NotNull());
}

static constexpr char kMyPluginName[] = "myplugin";

// This test builds the dynamic library and registers it as a PJRT plugin. This
// exists to test the dynamic registration path.
TEST(MypluginCPjRtTest, FindSharedLibrary) {
  if (tsl::kIsOpenSource) {
    GTEST_SKIP() << "Skipping test in open source mode.";
  }
  char build_dir[PATH_MAX];
  getcwd(build_dir, sizeof(build_dir));

  std::string build_dir_str(build_dir);

  std::string library_path = std::string(build_dir_str) +
                             "/third_party/tensorflow/compiler/xla/pjrt/plugin/"
                             "example_plugin/pjrt_c_api_myplugin_plugin.so";

  setenv("MYPLUGIN_DYNAMIC_PATH", library_path.c_str(), 1);
  REGISTER_DYNAMIC_PJRT_PLUGIN(kMyPluginName, "MYPLUGIN_DYNAMIC_PATH");

  absl::StatusOr<std::unique_ptr<xla::PjRtClient>> c_api_client =
      xla::GetCApiClient(kMyPluginName);
  EXPECT_OK(c_api_client);
  EXPECT_THAT(c_api_client.value(), ::testing::NotNull());
}
}  // namespace
