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

#ifndef XLA_SERVICE_CPU_TESTS_CPU_PJRT_CODEGEN_TEST_H_
#define XLA_SERVICE_CPU_TESTS_CPU_PJRT_CODEGEN_TEST_H_

#include <memory>
#include <string>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/service/compiler.h"
#include "xla/service/executable.h"
#include "xla/tests/hlo_pjrt_test_base.h"

namespace xla::cpu {

// Tests that verify IR emitted by the CPU backend is as expected.
class CpuPjRtCodegenTest : public HloPjRtTestBase {
 public:
  CpuPjRtCodegenTest();

 protected:
  // Compiles `hlo_module` with the CPU compiler. If `run_optimization_passes`
  // is true, also the HLO optimization pass pipeline is run.
  absl::StatusOr<std::unique_ptr<Executable>> CompileToExecutable(
      std::unique_ptr<HloModule> hlo_module, bool run_optimization_passes);

  // A thin wrapper around CompileAndVerifyIr.
  void CompileAndVerifyIr(std::unique_ptr<HloModule> module,
                          absl::string_view expected_llvm_ir,
                          bool match_optimized_ir = false,
                          bool run_optimization_passes = true);

  // A thin wrapper around CompileAndVerifyIr that takes a string instead of a
  // HloModule.
  void CompileAndVerifyIr(absl::string_view hlo_text,
                          absl::string_view expected_llvm_ir,
                          bool match_optimized_ir = false,
                          bool run_optimization_passes = true);

  // A thin wrapper around CompileAheadOfTimeAndVerifyIr.
  void CompileAheadOfTimeAndVerifyIr(std::unique_ptr<HloModule> hlo_module,
                                     const AotCompilationOptions& aot_options,
                                     absl::string_view expected_llvm_ir,
                                     bool match_optimized_ir = false);

  Compiler* compiler() const { return compiler_.get(); }

 private:
  Compiler::CompileOptions compile_options_;
  std::unique_ptr<Compiler> compiler_;
};

}  // namespace xla::cpu

#endif  // XLA_SERVICE_CPU_TESTS_CPU_PJRT_CODEGEN_TEST_H_
