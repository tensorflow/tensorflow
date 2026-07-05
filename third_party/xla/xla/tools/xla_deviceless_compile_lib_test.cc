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

#include <memory>
#include <optional>
#include <string>
#include <utility>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/status/status_matchers.h"
#include "absl/status/statusor.h"
#include "xla/tsl/platform/status_macros.h"
#include "xla/backends/gpu/target_config/target_config.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/hlo/parser/hlo_parser.h"
#include "xla/service/compiler.h"
#include "xla/service/hlo_module_config.h"
#include "xla/service/platform_util.h"
#include "xla/service/symbol_repository.h"
#include "xla/service/xla_compile_result.pb.h"
#include "xla/shape.h"
#include "xla/stream_executor/device_description.pb.h"
#include "xla/tools/xla_compile_lib.h"
#include "xla/tsl/platform/env.h"
#include "xla/tsl/platform/errors.h"
#include "xla/tsl/platform/statusor.h"
#include "xla/tsl/platform/test.h"
#include "xla/xla.pb.h"
#include "tsl/platform/path.h"

namespace xla {
namespace {

using ::testing::IsEmpty;
using ::testing::Not;

absl::StatusOr<Compiler::GpuTargetConfig> GetGpuTargetConfig() {
  const std::string spec_file = [&] {
    const std::string platform_name =
        PlatformUtil::CanonicalPlatformName("gpu").value_or("");
    if (platform_name == "rocm") {
      return "mi200.txtpb";
    }
    if (platform_name == "sycl") {
      return "bmg_g21.txtpb";
    }
    return "h100_sxm.txtpb";
  }();
  const std::string target_config_path =
      tsl::io::JoinPath(tsl::testing::XlaSrcRoot(),
                        "backends/gpu/target_config/specs", spec_file);
  stream_executor::GpuTargetConfigProto target_config_proto;
  RETURN_IF_ERROR(tsl::ReadTextProto(tsl::Env::Default(), target_config_path,
                                     &target_config_proto));
  return Compiler::GpuTargetConfig::FromProto(target_config_proto);
}

class XlaDevicelessCompileLibTest : public testing::TestWithParam<bool> {};

TEST_F(XlaDevicelessCompileLibTest, CompilesForGpuWithoutDevice) {
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          ParseAndReturnUnverifiedModule(R"hlo(
    HloModule module
    ENTRY entry_computation {
      a = f32[2,10] parameter(0)
      b = bf16[10,2] parameter(1)
      b_f32 = f32[10,2] convert(b)
      ROOT dot = f32[2,2] dot(a, b_f32), lhs_contracting_dims={1}, rhs_contracting_dims={0}
    })hlo"));

  TF_ASSERT_OK_AND_ASSIGN(gpu::GpuTargetConfig target_config,
                          GetGpuTargetConfig());

  CompilationResult result;
  ASSERT_THAT(
      CompileExecutable(std::move(module), BackendType::kGpu, target_config,
                        /*cpu_target_config=*/std::nullopt,
                        /*num_partitions=*/1, /*num_replicas=*/1, result),
      absl_testing::IsOkAndHolds(Not(IsEmpty())));
  EXPECT_TRUE(result.has_hlo_module()) << result.DebugString();
}

TEST_F(XlaDevicelessCompileLibTest, CompilesWithModuleConfig) {
  std::string hlo_string = R"hlo(
    HloModule module
    ENTRY entry_computation {
      p = f32[2,2] parameter(0)
      zero = f32[2,2] constant(0)
      ROOT add = f32[2,2] add(p, zero)
    })hlo";

  std::string tmp_dir = tsl::testing::TmpDir();
  std::string hlo_path = tsl::io::JoinPath(tmp_dir, "module.hlo");
  ASSERT_THAT(tsl::WriteStringToFile(tsl::Env::Default(), hlo_path, hlo_string),
              absl_testing::IsOk());

  HloModuleConfigProto config_proto;
  config_proto.mutable_debug_options()->add_xla_disable_hlo_passes(
      "priority-fusion");
  config_proto.mutable_debug_options()->add_xla_disable_hlo_passes(
      "multi_output_fusion");
  std::string config_path = tsl::io::JoinPath(tmp_dir, "config.txtpb");
  ASSERT_THAT(
      tsl::WriteTextProto(tsl::Env::Default(), config_path, config_proto),
      absl_testing::IsOk());

  ASSERT_OK_AND_ASSIGN(gpu::GpuTargetConfig target_config,
                       GetGpuTargetConfig());
  std::string target_config_path =
      tsl::io::JoinPath(tmp_dir, "gpu_target_config_2.txtpb");
  ASSERT_THAT(tsl::WriteTextProto(tsl::Env::Default(), target_config_path,
                                  target_config.ToProto()),
              absl_testing::IsOk());

  XlaCompileOptions options;
  options.module_path = hlo_path;
  options.module_config_path = config_path;
  options.platform = "gpu";
  options.gpu_options.gpu_target_config_path = target_config_path;
  options.result_output_file = tsl::io::JoinPath(tmp_dir, "result_disabled.pb");

  ASSERT_THAT(XlaCompileMain(options), absl_testing::IsOk());

  std::string result_str;
  ASSERT_THAT(tsl::ReadFileToString(tsl::Env::Default(),
                                    options.result_output_file, &result_str),
              absl_testing::IsOk());
  CompilationResult result;
  ASSERT_TRUE(result.ParseFromString(result_str));
  ASSERT_TRUE(result.has_hlo_module());

  ASSERT_OK_AND_ASSIGN(
      ProgramShape shape,
      ProgramShape::FromProto(result.hlo_module().host_program_shape()));
  HloModuleConfig config(shape);
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> out_module,
                       HloModule::CreateFromProto(result.hlo_module(), config));
  const HloInstruction* root =
      out_module->entry_computation()->root_instruction();
  EXPECT_EQ(root->opcode(), HloOpcode::kCopy);
}

}  // namespace
}  // namespace xla
