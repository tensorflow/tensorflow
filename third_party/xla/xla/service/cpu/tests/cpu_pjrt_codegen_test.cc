/* Copyright 2026 The OpenXLA Authors.

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

#include "xla/service/cpu/tests/cpu_pjrt_codegen_test.h"

#include <memory>
#include <string>
#include <utility>

#include "absl/log/check.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/service/compiler.h"
#include "xla/service/executable.h"
#include "xla/service/llvm_compiler.h"
#include "xla/service/platform_util.h"
#include "xla/stream_executor/platform.h"
#include "xla/tests/codegen_utils.h"
#include "xla/tsl/lib/core/status_test_util.h"
#include "xla/tsl/platform/statusor.h"
#include "tsl/platform/casts.h"

namespace xla::cpu {

namespace {
std::unique_ptr<Compiler> GetCpuCompiler() {
  absl::StatusOr<std::string> name = PlatformUtil::CanonicalPlatformName("cpu");
  CHECK_OK(name.status());
  absl::StatusOr<stream_executor::Platform::Id> platform_id =
      PlatformUtil::GetPlatformIdFromCanonicalName(*name);
  CHECK_OK(platform_id.status());
  absl::StatusOr<std::unique_ptr<Compiler>> compiler =
      Compiler::GetForPlatform(*platform_id);
  CHECK_OK(compiler.status());
  return std::move(*compiler);
}
}  // namespace

CpuPjRtCodegenTest::CpuPjRtCodegenTest() : compiler_(GetCpuCompiler()) {}

absl::StatusOr<std::unique_ptr<Executable>>
CpuPjRtCodegenTest::CompileToExecutable(std::unique_ptr<HloModule> hlo_module,
                                        bool run_optimization_passes) {
  return xla::CompileToExecutable(compiler(), compile_options_,
                                  std::move(hlo_module),
                                  run_optimization_passes);
}

void CpuPjRtCodegenTest::CompileAndVerifyIr(
    std::unique_ptr<HloModule> hlo_module, absl::string_view expected_llvm_ir,
    bool match_optimized_ir, bool run_optimization_passes) {
  auto llvm_compiler = absl::down_cast<LLVMCompiler*>(compiler());
  TF_ASSERT_OK(xla::CompileAndVerifyIr(
      llvm_compiler, compile_options_, std::move(hlo_module), expected_llvm_ir,
      match_optimized_ir, run_optimization_passes));
}

void CpuPjRtCodegenTest::CompileAndVerifyIr(absl::string_view hlo_text,
                                            absl::string_view expected_llvm_ir,
                                            bool match_optimized_ir,
                                            bool run_optimization_passes) {
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> hlo_module,
                          ParseAndReturnVerifiedModule(hlo_text));
  CompileAndVerifyIr(std::move(hlo_module), expected_llvm_ir,
                     match_optimized_ir, run_optimization_passes);
}

void CpuPjRtCodegenTest::CompileAheadOfTimeAndVerifyIr(
    std::unique_ptr<HloModule> hlo_module,
    const AotCompilationOptions& aot_options,
    absl::string_view expected_llvm_ir, bool match_optimized_ir) {
  auto llvm_compiler = absl::down_cast<LLVMCompiler*>(compiler());
  TF_ASSERT_OK(xla::CompileAheadOfTimeAndVerifyIr(
      llvm_compiler, aot_options, std::move(hlo_module), expected_llvm_ir,
      match_optimized_ir));
}

}  // namespace xla::cpu
