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

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/status/statusor.h"
#include "xla/pjrt/c/pjrt_c_api.h"
#include "xla/pjrt/pjrt_c_api_client.h"
#include "xla/pjrt/pjrt_client.h"
#include "xla/tsl/lib/core/status_test_util.h"

namespace {

TEST(MypluginCPjRtTest, CreatesPjRtAPI) {
  const PJRT_Api* myplugin = GetPjrtApi();
  EXPECT_THAT(myplugin, ::testing::NotNull());
}

// This test builds the dynamic library and registers it as a PJRT plugin. This
// exists to test the dynamic registration path.
TEST(MypluginCPjRtTest, FindSharedLibrary) {
  absl::StatusOr<std::unique_ptr<xla::PjRtClient>> c_api_client =
      xla::GetCApiClient("myplugin");
  TF_EXPECT_OK(c_api_client);
}
}  // namespace
