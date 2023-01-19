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

#include <string>

#include <gtest/gtest.h>
#include "tensorflow/compiler/xla/client/client_library.h"
#include "tensorflow/compiler/xla/client/local_client.h"
#include "tensorflow/compiler/xla/executable_run_options.h"
#include "tensorflow/compiler/xla/literal_util.h"
#include "tensorflow/compiler/xla/service/platform_util.h"
#include "tensorflow/compiler/xla/tests/literal_test_util.h"
#include "tensorflow/tsl/lib/core/status_test_util.h"
#include "tensorflow/tsl/platform/env.h"
#include "tensorflow/tsl/platform/path.h"
#include "tensorflow/tsl/platform/statusor.h"
#include "tensorflow/tsl/platform/test.h"

namespace xla {
namespace xla_compile {
namespace {

TEST(XlaCompileTest, LoadCpuExecutable) {
  std::string path =
      tsl::io::JoinPath(tsl::testing::XlaSrcRoot(), "service",
                        "xla_aot_compile_stablehlo_test_cpu_executable");
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
  auto alpha_literal = xla::LiteralUtil::CreateR0<float>(3.14f);
  auto x_literal = xla::LiteralUtil::CreateR1<float>({1.0f, 2.0f, 3.0f, 4.0f});
  auto y_literal =
      xla::LiteralUtil::CreateR1<float>({10.5f, 20.5f, 30.5f, 40.5f});
  TF_ASSERT_OK_AND_ASSIGN(ScopedShapedBuffer alpha,
                          client->LiteralToShapedBuffer(
                              alpha_literal, client->default_device_ordinal()));
  TF_ASSERT_OK_AND_ASSIGN(ScopedShapedBuffer x,
                          client->LiteralToShapedBuffer(
                              x_literal, client->default_device_ordinal()));
  TF_ASSERT_OK_AND_ASSIGN(ScopedShapedBuffer y,
                          client->LiteralToShapedBuffer(
                              y_literal, client->default_device_ordinal()));
  ExecutableRunOptions executable_run_options;
  executable_run_options.set_allocator(client->backend().memory_allocator());
  TF_ASSERT_OK_AND_ASSIGN(
      ScopedShapedBuffer axpy_result,
      local_executable->Run({&alpha, &x, &y}, executable_run_options));

  TF_ASSERT_OK_AND_ASSIGN(Literal axpy_result_literal,
                          client->ShapedBufferToLiteral(axpy_result));
  xla::LiteralTestUtil::ExpectR1Near<float>({13.64f, 26.78f, 39.92f, 53.06f},
                                            axpy_result_literal,
                                            xla::ErrorSpec(0.01f));
}

}  // namespace
}  // namespace xla_compile
}  // namespace xla
