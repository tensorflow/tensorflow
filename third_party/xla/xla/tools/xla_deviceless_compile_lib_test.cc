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
#include "xla/backends/gpu/target_config/target_config.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/parser/hlo_parser.h"
#include "xla/service/compiler.h"
#include "xla/service/symbol_repository.h"
#include "xla/service/xla_compile_result.pb.h"
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
  const std::string target_config_path =
      tsl::io::JoinPath(tsl::testing::XlaSrcRoot(),
                        "backends/gpu/target_config/specs", "h100_sxm.txtpb");
  stream_executor::GpuTargetConfigProto target_config_proto;
  TF_RETURN_IF_ERROR(tsl::ReadTextProto(tsl::Env::Default(), target_config_path,
                                        &target_config_proto));
  return Compiler::GpuTargetConfig::FromProto(target_config_proto);
}

class XlaDevicelessCompileLibTest : public testing::TestWithParam<bool> {};

TEST_P(XlaDevicelessCompileLibTest, CompilesForGpuWithoutDevice) {
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          ParseAndReturnUnverifiedModule(R"hlo(
    HloModule module
    ENTRY entry_computation {
      a = f32[2,10] parameter(0)
      b = bf16[10,2] parameter(1)
      b_f32 = f32[10,2] convert(b)
      ROOT dot = f32[2,2] dot(a, b_f32), lhs_contracting_dims={1}, rhs_contracting_dims={0}
    })hlo"));
  module->mutable_config()
      .mutable_debug_options()
      .set_xla_gpu_experimental_aot_compiled_thunks(GetParam());

  TF_ASSERT_OK_AND_ASSIGN(gpu::GpuTargetConfig target_config,
                          GetGpuTargetConfig());

  CompilationResult result;
  ASSERT_THAT(CompileExecutable(std::move(module), BackendType::kGpu,
                                target_config, std::nullopt, result),
              absl_testing::IsOkAndHolds(Not(IsEmpty())));
  EXPECT_TRUE(result.has_hlo_module()) << result.DebugString();
}

INSTANTIATE_TEST_SUITE_P(XlaDevicelessCompileLibTest,
                         XlaDevicelessCompileLibTest, testing::Bool(),
                         [](const testing::TestParamInfo<bool>& info) {
                           return info.param ? "NewAotFlow" : "LegacyAotFlow";
                         });

}  // namespace
}  // namespace xla
