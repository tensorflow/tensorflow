/* Copyright 2019 The OpenXLA Authors.

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
#include <utility>

#include <gtest/gtest.h>
#include "absl/strings/string_view.h"
#include "llvm-c/Target.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/service/cpu/cpu_compiler.h"
#include "xla/service/cpu/tests/cpu_codegen_test.h"
#include "xla/tsl/platform/test.h"

namespace xla {
namespace cpu {
namespace {

const char* const kTriple_x86_64 = "x86_64-pc-linux";
const char* const kTriple_android_arm = "armv7-none-android";

struct ProfilingTestSpec {
  absl::string_view triple;
  absl::string_view check_lines;
};

// Tests that profiling intrinsics get inserted.
class CpuProfilingTest
    : public CpuCodegenTest,
      public ::testing::WithParamInterface<ProfilingTestSpec> {
 public:
  static std::string Name(
      const ::testing::TestParamInfo<ProfilingTestSpec>& info) {
    auto spec = info.param;

    std::string triple{spec.triple.data(), spec.triple.size()};
    if (triple == kTriple_x86_64) {
      triple = "x86_64";
    } else if (triple == kTriple_android_arm) {
      triple = "android_arm";
    } else {
      triple = "Unknown";
    }

    return triple;
  }
};

TEST_P(CpuProfilingTest, DoIt) {
  HloComputation::Builder builder(TestName());
  ProfilingTestSpec spec = GetParam();

  LLVMInitializeX86Target();
  LLVMInitializeX86TargetInfo();
  LLVMInitializeX86TargetMC();
  LLVMInitializeARMTarget();
  LLVMInitializeARMTargetInfo();
  LLVMInitializeARMTargetMC();

  std::string triple{spec.triple.data(), spec.triple.size()};

  CpuAotCompilationOptions options{
      /*triple=*/triple, /*cpu_name=*/"", /*features=*/"",
      /*entry_point_name=*/"entry",
      /*relocation_model=*/CpuAotCompilationOptions::RelocationModel::Static};

  constexpr char hlo_text[] = R"(HloModule test
    ENTRY test {
      p = f32[1024] parameter(0)
      ROOT sin = f32[1024] sine(p)
    })";

  auto config = GetModuleConfigForTest();
  auto debug_options = config.debug_options();
  debug_options.set_xla_hlo_profile(true);
  config.set_debug_options(debug_options);
  auto hlo_module = ParseAndReturnVerifiedModule(hlo_text, config);

  std::string check_lines{spec.check_lines.data(), spec.check_lines.size()};

  hlo_module.value()
      ->mutable_config()
      .mutable_debug_options()
      .set_xla_cpu_use_thunk_runtime(false);

  CompileAheadOfTimeAndVerifyIr(std::move(hlo_module).value(), options,
                                check_lines,
                                /*match_optimized_ir=*/true);
}

ProfilingTestSpec CpuProfilingTestCases[] = {
    ProfilingTestSpec{
        kTriple_x86_64,
        R"(CHECK: [[START:%[^ ]*]] = tail call { i64, i32 } @llvm.x86.rdtscp
           CHECK: extractvalue { i64, i32 } [[START]], 0
           CHECK: [[STOP:%[^ ]*]] = tail call { i64, i32 } @llvm.x86.rdtscp
           CHECK: extractvalue { i64, i32 } [[STOP]], 0)"},
    ProfilingTestSpec{kTriple_android_arm,
                      R"(CHECK: call i64 @llvm.readcyclecounter
                         CHECK: call i64 @llvm.readcyclecounter)"},
};

INSTANTIATE_TEST_SUITE_P(CpuProfilingTestInstantiation, CpuProfilingTest,
                         ::testing::ValuesIn(CpuProfilingTestCases),
                         CpuProfilingTest::Name);

}  // namespace
}  // namespace cpu
}  // namespace xla
