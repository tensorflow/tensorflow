/* Copyright 2023 The OpenXLA Authors.

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

#include "xla/pjrt/tf_pjrt_client.h"

#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/base/casts.h"
#include "xla/hlo/builder/xla_computation.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/parser/hlo_parser.h"
#include "xla/literal.h"
#include "xla/literal_util.h"
#include "xla/pjrt/compiled_memory_stats.h"
#include "xla/pjrt/pjrt_client.h"
#include "xla/pjrt/pjrt_executable.h"
#include "xla/pjrt/plugin/xla_cpu/cpu_client_options.h"
#include "xla/pjrt/plugin/xla_cpu/xla_cpu_pjrt_client.h"
#include "xla/service/hlo.pb.h"
#include "xla/shape.h"
#include "xla/shape_util.h"
#include "xla/tsl/platform/env.h"
#include "xla/tsl/platform/file_system.h"
#include "xla/tsl/platform/statusor.h"
#include "xla/tsl/platform/test.h"
#include "xla/xla_data.pb.h"

namespace xla {
namespace {

TEST(TfClientTest, ExecuteAndHloSnapshot) {
  constexpr char kProgram[] = R"(
    HloModule add
    ENTRY add {
      x = f32[3,2] parameter(0)
      y = f32[3,2] parameter(1)
      ROOT add = f32[3,2] add(x, y)
    })";

  xla::CpuClientOptions cpu_options{.asynchronous = true};
  TF_ASSERT_OK_AND_ASSIGN(auto client, xla::GetXlaPjrtCpuClient(cpu_options));
  client = TfPjRtClient::CreateTfPjRtClient(std::move(client));
  TF_ASSERT_OK_AND_ASSIGN(auto hlo_module,
                          ParseAndReturnUnverifiedModule(kProgram, {}));

  std::string dir = tsl::testing::TmpDir();
  xla::CompileOptions options;
  auto* debug_opts = options.executable_build_options.mutable_debug_options();
  debug_opts->set_xla_dump_to(dir);
  debug_opts->set_xla_dump_hlo_snapshots(true);
  XlaComputation xla_computation(hlo_module->ToProto());
  TF_ASSERT_OK_AND_ASSIGN(auto pjrt_executable,
                          client->CompileAndLoad(xla_computation, options));

  auto* device = client->addressable_devices()[0];

  std::vector<float> data1{1.0, 2.0, 3.0, 4.0, 5.0, 6.0};
  std::vector<float> data2{10.0, 20.0, 30.0, 40.0, 50.0, 60.0};
  Shape shape = ShapeUtil::MakeShape(F32, {3, 2});
  TF_ASSERT_OK_AND_ASSIGN(
      auto buffer1,
      client->BufferFromHostBuffer(
          data1.data(), shape.element_type(), shape.dimensions(),
          /*byte_strides=*/std::nullopt,
          PjRtClient::HostBufferSemantics::kImmutableOnlyDuringCall, nullptr,
          *device->default_memory_space(), /*device_layout=*/nullptr));
  TF_ASSERT_OK_AND_ASSIGN(
      auto buffer2,
      client->BufferFromHostBuffer(
          data2.data(), shape.element_type(), shape.dimensions(),
          /*byte_strides=*/std::nullopt,
          PjRtClient::HostBufferSemantics::kImmutableOnlyDuringCall, nullptr,
          *device->default_memory_space(), /*device_layout=*/nullptr));

  auto result = pjrt_executable->Execute(
      /*argument_handles=*/{{buffer1.get(), buffer2.get()}},
      /*options=*/{});
  ASSERT_TRUE(result.ok());

  tsl::FileSystem* fs;
  ASSERT_TRUE(tsl::Env::Default()->GetFileSystemForFile(dir, &fs).ok());

  std::vector<std::string> paths;
  ASSERT_TRUE(fs->GetMatchingPaths(dir + "/*.snapshot.*.pb", &paths).ok());
  ASSERT_EQ(paths.size(), 1);

  HloSnapshot snapshot;
  ASSERT_TRUE(
      tsl::ReadBinaryProto(tsl::Env::Default(), paths[0], &snapshot).ok());

  ASSERT_EQ(*Literal::CreateFromProto(snapshot.arguments(0)),
            LiteralUtil::CreateR2<float>({{1.0, 2.0}, {3.0, 4.0}, {5.0, 6.0}}));
  ASSERT_EQ(
      *Literal::CreateFromProto(snapshot.arguments(1)),
      LiteralUtil::CreateR2<float>({{10.0, 20.0}, {30.0, 40.0}, {50.0, 60.0}}));
  ASSERT_EQ(
      *Literal::CreateFromProto(snapshot.result()),
      LiteralUtil::CreateR2<float>({{11.0, 22.0}, {33.0, 44.0}, {55.0, 66.0}}));

  auto* tf_pjrt_client = absl::down_cast<TfPjRtClient*>(client.get());
  tf_pjrt_client->DestroyWrappedBuffersAndClient();
}

TEST(TfClientTest, ForwardsGetExecutableAndGetCompiledMemoryStats) {
  constexpr char kProgram[] = R"hlo(
    HloModule add
    ENTRY add {
      x = f32[3,2] parameter(0)
      y = f32[3,2] parameter(1)
      ROOT add = f32[3,2] add(x, y)
    }
  )hlo";

  ASSERT_OK_AND_ASSIGN(std::unique_ptr<PjRtClient> cpu_client,
                       xla::GetXlaPjrtCpuClient(xla::CpuClientOptions{}));
  std::unique_ptr<TfPjRtClient> client =
      TfPjRtClient::CreateTfPjRtClient(std::move(cpu_client));
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> hlo_module,
                       ParseAndReturnUnverifiedModule(kProgram, {}));
  XlaComputation xla_computation(hlo_module->ToProto());
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<PjRtLoadedExecutable> pjrt_executable,
                       client->CompileAndLoad(xla_computation, {}));

  auto* tf_executable =
      absl::down_cast<TfPjRtExecutable*>(pjrt_executable.get());
  ASSERT_NE(tf_executable, nullptr);
  EXPECT_NE(pjrt_executable->GetExecutable(), nullptr);
  EXPECT_EQ(pjrt_executable->GetExecutable(),
            tf_executable->wrapped()->GetExecutable());

  ASSERT_OK_AND_ASSIGN(CompiledMemoryStats stats,
                       pjrt_executable->GetCompiledMemoryStats());
  ASSERT_OK_AND_ASSIGN(CompiledMemoryStats wrapped_stats,
                       tf_executable->wrapped()->GetCompiledMemoryStats());
  EXPECT_EQ(stats.generated_code_size_in_bytes,
            wrapped_stats.generated_code_size_in_bytes);
  EXPECT_EQ(stats.argument_size_in_bytes, wrapped_stats.argument_size_in_bytes);
  EXPECT_EQ(stats.output_size_in_bytes, wrapped_stats.output_size_in_bytes);

  client->DestroyWrappedBuffersAndClient();
}

}  // namespace
}  // namespace xla
