/* Copyright 2022 The OpenXLA Authors.

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

#include <string>

#include <gtest/gtest.h>
#include "absl/types/span.h"
#include "xla/client/client_library.h"
#include "xla/client/executable_build_options.h"
#include "xla/client/local_client.h"
#include "xla/executable_run_options.h"
#include "xla/literal.h"
#include "xla/literal_util.h"
#include "xla/service/platform_util.h"
#include "xla/service/shaped_buffer.h"
#include "xla/tsl/lib/core/status_test_util.h"
#include "tsl/platform/env.h"
#include "tsl/platform/path.h"
#include "tsl/platform/statusor.h"
#include "tsl/platform/test.h"

namespace xla {
namespace xla_compile {
namespace {

TEST(XlaCompileTest, LoadCpuExecutable) {
  std::string path = tsl::io::JoinPath(tsl::testing::XlaSrcRoot(), "service",
                                       "xla_aot_compile_test_cpu_executable");
  std::string serialized_aot_result;
  TF_ASSERT_OK(
      tsl::ReadFileToString(tsl::Env::Default(), path, &serialized_aot_result));

  // Get a LocalClient
  TF_ASSERT_OK_AND_ASSIGN(se::Platform * platform,
                          PlatformUtil::GetPlatform("Host"));
  if (platform->VisibleDeviceCount() <= 0) {
    EXPECT_TRUE(false) << "CPU platform has no visible devices.";
  }
  LocalClientOptions local_client_options;
  local_client_options.set_platform(platform);
  TF_ASSERT_OK_AND_ASSIGN(
      LocalClient * client,
      ClientLibrary::GetOrCreateLocalClient(local_client_options));

  // Load from AOT result.
  ExecutableBuildOptions executable_build_options;
  TF_ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<LocalExecutable> local_executable,
      client->Load(serialized_aot_result, executable_build_options));

  // Run loaded excutable.
  Literal input1 = LiteralUtil::CreateR1<double>({0.0f, 1.0f, 2.0f});
  Literal input2 = LiteralUtil::CreateR1<double>({1.0f, 2.0f, 4.0f});
  TF_ASSERT_OK_AND_ASSIGN(
      ScopedShapedBuffer array1,
      client->LiteralToShapedBuffer(input1, client->default_device_ordinal()));
  TF_ASSERT_OK_AND_ASSIGN(
      ScopedShapedBuffer array2,
      client->LiteralToShapedBuffer(input2, client->default_device_ordinal()));
  ExecutableRunOptions executable_run_options;
  executable_run_options.set_allocator(client->backend().memory_allocator());
  TF_ASSERT_OK_AND_ASSIGN(
      ScopedShapedBuffer result,
      local_executable->Run(
          absl::Span<const ShapedBuffer* const>{&array1, &array2},
          executable_run_options));

  TF_ASSERT_OK_AND_ASSIGN(Literal output,
                          client->ShapedBufferToLiteral(result));
  Literal expected = LiteralUtil::CreateR1<double>({1.0f, 3.0f, 6.0f});
  EXPECT_EQ(expected, output);
}

}  // namespace
}  // namespace xla_compile
}  // namespace xla
