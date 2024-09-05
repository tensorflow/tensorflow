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

#include <memory>
#include <string>

#include <gtest/gtest.h>
#include "absl/strings/match.h"
#include "absl/strings/string_view.h"
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

class XlaAotCompileTest : public ::testing::TestWithParam<absl::string_view> {};

TEST_P(XlaAotCompileTest, LoadGpuExecutable) {
  std::string path =
      tsl::io::JoinPath(tsl::testing::XlaSrcRoot(), "service", GetParam()
                        /*"xla_aot_compile_test_gpu_executable"*/);
  std::string serialized_aot_result;
  TF_ASSERT_OK(
      tsl::ReadFileToString(tsl::Env::Default(), path, &serialized_aot_result));

  // Get a LocalClient
  TF_ASSERT_OK_AND_ASSIGN(se::Platform * platform,
                          PlatformUtil::GetPlatform("CUDA"));
  ASSERT_GT(platform->VisibleDeviceCount(), 0);

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

  // Run loaded executable.
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
      local_executable->Run({&array1, &array2}, executable_run_options));

  TF_ASSERT_OK_AND_ASSIGN(Literal output,
                          client->ShapedBufferToLiteral(result));
  Literal expected = LiteralUtil::CreateR1<double>({1.0f, 3.0f, 6.0f});
  EXPECT_EQ(expected, output);
}

INSTANTIATE_TEST_SUITE_P(
    TestingAotFormats, XlaAotCompileTest,
    testing::Values("xla_aot_compile_test_gpu_executable",
                    "xla_aot_compile_test_gpu_executable_hlo"));

TEST(XlaCompileTest, LoadGpuExecutableWithConstant) {
  std::string path =
      tsl::io::JoinPath(tsl::testing::XlaSrcRoot(), "service",
                        "xla_aot_compile_test_gpu_executable_constant");
  std::string serialized_aot_result;
  TF_ASSERT_OK(
      tsl::ReadFileToString(tsl::Env::Default(), path, &serialized_aot_result));

  // Get a LocalClient
  TF_ASSERT_OK_AND_ASSIGN(se::Platform * platform,
                          PlatformUtil::GetPlatform("CUDA"));
  ASSERT_GT(platform->VisibleDeviceCount(), 0);

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

  // Run loaded executable.
  Literal input = LiteralUtil::CreateR1<double>({3.0f, 3.0f, 3.0f});
  TF_ASSERT_OK_AND_ASSIGN(
      ScopedShapedBuffer array,
      client->LiteralToShapedBuffer(input, client->default_device_ordinal()));
  ExecutableRunOptions executable_run_options;
  executable_run_options.set_allocator(client->backend().memory_allocator());
  TF_ASSERT_OK_AND_ASSIGN(
      ScopedShapedBuffer result,
      local_executable->Run({&array}, executable_run_options));

  TF_ASSERT_OK_AND_ASSIGN(Literal output,
                          client->ShapedBufferToLiteral(result));
  Literal expected = LiteralUtil::CreateR1<double>({4.0f, 5.0f, 6.0f});
  EXPECT_EQ(expected, output);
}

// Should also cover the case of loading a GPU executable with a GEMM.
TEST(XlaCompileTest, LoadGpuExecutableWithConvolution) {
  std::string path =
      tsl::io::JoinPath(tsl::testing::XlaSrcRoot(), "service",
                        "xla_aot_compile_test_gpu_executable_convolution");
  std::string serialized_aot_result;
  TF_ASSERT_OK(
      tsl::ReadFileToString(tsl::Env::Default(), path, &serialized_aot_result));

  // Check that GpuConvAlgorithmPicker successfully loaded autotune results.
  EXPECT_TRUE(absl::StrContains(serialized_aot_result, "\"algo_id\":\"28\""))
      << serialized_aot_result;

  // Get a LocalClient
  TF_ASSERT_OK_AND_ASSIGN(se::Platform * platform,
                          PlatformUtil::GetPlatform("CUDA"));
  ASSERT_GT(platform->VisibleDeviceCount(), 0);

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

  // Run loaded executable.
  Literal input1 = LiteralUtil::CreateR4<float>(
      {{{{1.0, 2.0}, {3.0, 4.0}, {5.0, 6.0}, {7.0, 8.0}},
        {{11.0, 12.0}, {13.0, 14.0}, {15.0, 16.0}, {17.0, 18.0}},
        {{21.0, 22.0}, {23.0, 24.0}, {25.0, 26.0}, {27.0, 28.0}},
        {{31.0, 32.0}, {33.0, 34.0}, {35.0, 36.0}, {37.0, 38.0}}}});
  Literal input2 =
      LiteralUtil::CreateR4<float>({{{{1.0}, {2.0}}, {{3.0}, {4.0}}},
                                    {{{5.0}, {6.0}}, {{7.0}, {8.0}}},
                                    {{{9.0}, {10.0}}, {{11.0}, {12.0}}}});

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
      local_executable->Run({&array1, &array2}, executable_run_options));

  TF_ASSERT_OK_AND_ASSIGN(Literal output,
                          client->ShapedBufferToLiteral(result));
  Literal expected = LiteralUtil::CreateR4<float>({{
      {{1310.0}, {1466.0}, {1622.0}},
      {{2090.0}, {2246.0}, {2402.0}},
  }});
  EXPECT_EQ(expected, output);
}

}  // namespace
}  // namespace xla_compile
}  // namespace xla
