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

#include "xla/service/multi_module_driver.h"

#include <memory>
#include <utility>

#include "absl/status/statusor.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/testlib/filecheck.h"
#include "xla/hlo/testlib/verified_hlo_module.h"
#include "xla/service/compiler.h"
#include "xla/service/cpu/cpu_compiler.h"
#include "xla/tests/hlo_pjrt_test_base.h"
#include "xla/tsl/platform/test.h"

namespace xla {
namespace {

class MultiModuleDriverTest : public HloTestBase {
 protected:
  absl::StatusOr<std::unique_ptr<VerifiedHloModule>>
  GetModuleWithCompilationUnit() {
    const char* hlo_string = R"(
HloModule module
callee {
  p0 = f32[] parameter(0)
  p1 = f32[] parameter(1)
  ROOT add = f32[] add(p0, p1)
}
ENTRY entry {
  p0 = f32[] parameter(0)
  p1 = f32[] parameter(1)
  ROOT call = f32[] call(p0, p1), to_apply=callee,
    frontend_attributes={compilation_unit="callee"}
}
)";
    return ParseAndReturnVerifiedModule(hlo_string);
  }

  absl::StatusOr<std::unique_ptr<VerifiedHloModule>> GetNestedModule() {
    const char* hlo_string = R"(
HloModule module
inner_callee {
  p0 = f32[] parameter(0)
  p1 = f32[] parameter(1)
  ROOT add = f32[] add(p0, p1)
}
outer_callee {
  p0 = f32[] parameter(0)
  p1 = f32[] parameter(1)
  ROOT call = f32[] call(p0, p1), to_apply=inner_callee,
    frontend_attributes={compilation_unit="inner"}
}
ENTRY entry {
  p0 = f32[] parameter(0)
  p1 = f32[] parameter(1)
  ROOT call = f32[] call(p0, p1), to_apply=outer_callee,
    frontend_attributes={compilation_unit="outer"}
}
)";
    return ParseAndReturnVerifiedModule(hlo_string);
  }
};

TEST_F(MultiModuleDriverTest, CompileAndRunCompilationUnit) {
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<VerifiedHloModule> module,
                       GetModuleWithCompilationUnit());

  EXPECT_TRUE(Run(std::move(module), /*run_hlo_passes=*/true));
}

TEST_F(MultiModuleDriverTest, VerifySplittingHappens) {
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<VerifiedHloModule> module,
                       GetModuleWithCompilationUnit());

  cpu::CpuCompiler compiler;
  auto options = Compiler::CompileOptions();
  ASSERT_OK_AND_ASSIGN(auto executables,
                       compiler.Compile(std::move(module), {nullptr}, options));

  EXPECT_EQ(executables.size(), 1);
  const HloModule& optimized_module = executables[0]->module();

  absl::StatusOr<bool> filecheck_result =
      RunFileCheck(optimized_module.ToString(), R"(
// CHECK-NOT: custom-call
// CHECK-NOT: _xla_multi_module_call
// CHECK: call(
)");
  ASSERT_OK(filecheck_result.status());
  EXPECT_TRUE(filecheck_result.value());
}

TEST_F(MultiModuleDriverTest, VerifySplittingInRunHloPasses) {
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<VerifiedHloModule> module,
                       GetModuleWithCompilationUnit());

  cpu::CpuCompiler compiler;
  auto options = Compiler::CompileOptions();

  MultiModuleDriver::ResetCompileCount();

  ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<HloModule> optimized_module,
      compiler.RunHloPasses(std::move(module), nullptr, options));

  EXPECT_GT(MultiModuleDriver::GetCompileCount(), 0);
}

TEST_F(MultiModuleDriverTest, CompileAndRunNested) {
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<VerifiedHloModule> module,
                       GetNestedModule());

  EXPECT_TRUE(Run(std::move(module), /*run_hlo_passes=*/true));
}

TEST_F(MultiModuleDriverTest, VerifyNestedSplittingHappens) {
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<VerifiedHloModule> module,
                       GetNestedModule());

  cpu::CpuCompiler compiler;
  auto options = Compiler::CompileOptions();
  ASSERT_OK_AND_ASSIGN(auto executables,
                       compiler.Compile(std::move(module), {nullptr}, options));

  EXPECT_EQ(executables.size(), 1);
  const HloModule& optimized_module = executables[0]->module();

  absl::StatusOr<bool> filecheck_result =
      RunFileCheck(optimized_module.ToString(), R"(
// CHECK-NOT: custom-call
// CHECK-NOT: _xla_multi_module_call
// CHECK: call(
// CHECK: call(
)");
  ASSERT_OK(filecheck_result.status());
  EXPECT_TRUE(filecheck_result.value());
}

TEST_F(MultiModuleDriverTest, CompileAndRunSharedCompilationUnit) {
  const char* hlo_string = R"(
HloModule module
callee {
  p0 = f32[] parameter(0)
  ROOT neg = f32[] negate(p0)
}
ENTRY entry {
  p0 = f32[] parameter(0)
  p1 = f32[] parameter(1)
  call1 = f32[] call(p0), to_apply=callee,
    frontend_attributes={compilation_unit="callee"}
  call2 = f32[] call(p1), to_apply=callee,
    frontend_attributes={compilation_unit="callee"}
  ROOT add = f32[] add(call1, call2)
}
)";
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<VerifiedHloModule> module,
                       ParseAndReturnVerifiedModule(hlo_string));

  EXPECT_TRUE(Run(std::move(module), /*run_hlo_passes=*/true));
}

}  // namespace
}  // namespace xla
