/* Copyright 2025 The OpenXLA Authors.

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

#include <cstdint>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "absl/strings/string_view.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/ir/hlo_module_group.h"
#include "xla/literal.h"
#include "xla/literal_util.h"
#include "xla/service/compiler.h"
#include "xla/service/cpu/cpu_aot_compilation_result.h"
#include "xla/service/cpu/test_target_triple_helper.h"
#include "xla/service/executable.h"
#include "xla/service/hlo_runner.h"
#include "xla/service/hlo_runner_interface.h"
#include "xla/stream_executor/platform.h"
#include "xla/stream_executor/platform_manager.h"
#include "xla/tests/hlo_test_base.h"
#include "xla/tests/literal_test_util.h"
#include "xla/tsl/platform/statusor.h"
#include "xla/tsl/platform/test.h"

namespace xla {
namespace cpu {
namespace {

using CpuAotCompilerTest = HloTestBase;

// Separate from cpu_compiler_test.cc because we have to use HloTestBase to
// get the HloRunner since HloRunnerAgnosticTestBase doesn't support AOT
// compilation.
// TODO(basioli): Unify this test with the gpu_compiler one.
TEST_F(CpuAotCompilerTest, AheadOfTimeCompilation) {
  constexpr absl::string_view kHloAdd1 = R"(
add1 {
  p = s32[] parameter(0)
  c = s32[] constant(1)
  ROOT a = s32[] add(p, c)
}

ENTRY e {
  p = s32[] parameter(0)
  ROOT r = s32[] fusion(p), kind=kLoop, calls=add1
})";

  constexpr absl::string_view kHloAdd2 = R"(
add2 {
  p = s32[] parameter(0)
  c = s32[] constant(2)
  ROOT a = s32[] add(p, c)
}

ENTRY e {
  p = s32[] parameter(0)
  ROOT r = s32[] fusion(p), kind=kLoop, calls=add2
})";

  TF_ASSERT_OK_AND_ASSIGN(se::Platform * platform,
                          se::PlatformManager::PlatformWithName("host"));
  TF_ASSERT_OK_AND_ASSIGN(se::StreamExecutor * stream_exec,
                          platform->ExecutorForDevice(0));

  Compiler* compiler = backend().compiler();
  ASSERT_NE(compiler, nullptr);

  std::unique_ptr<AotCompilationOptions> aot_options = std::make_unique<
      CpuAotCompilationOptions>(
      /*triple=*/kTargetTripleForHost, /*cpu_name=*/kTargetCpuForHost,
      /*features=*/"",
      /*entry_point_name=*/"entry",
      /*relocation_model=*/CpuAotCompilationOptions::RelocationModel::BigPic);
  aot_options->set_executor(stream_exec);

  auto test = [this, &compiler, aot_options = std::move(aot_options)](
                  absl::string_view test_name, absl::string_view hlo, int input,
                  int expected_result) {
    SCOPED_TRACE(test_name);
    TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                            ParseAndReturnVerifiedModule(hlo));
    auto module_group = std::make_unique<HloModuleGroup>(std::move(module));
    TF_ASSERT_OK_AND_ASSIGN(
        std::vector<std::unique_ptr<AotCompilationResult>> aot_results,
        compiler->CompileAheadOfTime(std::move(module_group), *aot_options));

    TF_ASSERT_OK_AND_ASSIGN(std::string serialized_aot_result,
                            aot_results[0]->SerializeAsString());
    TF_ASSERT_OK_AND_ASSIGN(
        std::unique_ptr<AotCompilationResult> aot_result,
        compiler->LoadAotCompilationResult(serialized_aot_result));

    TF_ASSERT_OK_AND_ASSIGN(
        std::unique_ptr<Executable> executable,
        std::move(*aot_result)
            .LoadExecutable(compiler, aot_options->executor()));
    std::unique_ptr<OpaqueExecutable> wrapped_executable =
        test_runner_as_hlo_runner().WrapExecutable(std::move(executable));

    const xla::Literal literal_input =
        xla::LiteralUtil::CreateR0<int32_t>(input);
    const xla::Literal literal_expected_result =
        xla::LiteralUtil::CreateR0<int32_t>(expected_result);

    TF_ASSERT_OK_AND_ASSIGN(Literal result,
                            test_runner_as_hlo_runner().ExecuteWithExecutable(
                                wrapped_executable.get(), {&literal_input}));

    EXPECT_TRUE(LiteralTestUtil::Equal(result, literal_expected_result));
  };

  test("Test kHloAdd1", kHloAdd1, 1, 2);
  test("Test kHloAdd2", kHloAdd2, 1, 3);
}

}  // namespace
}  // namespace cpu
}  // namespace xla
