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
#include <vector>

#include "absl/status/statusor.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/ir/hlo_module_metadata.h"
#include "xla/hlo/testlib/filecheck.h"
#include "xla/hlo/testlib/verified_hlo_module.h"
#include "xla/service/compiler.h"
#include "xla/service/cpu/cpu_compiler.h"
#include "xla/tests/hlo_test_base.h"
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

TEST_F(MultiModuleDriverTest, ShouldProcessInlineableXlaLate) {
  const char* hlo_string = R"(
HloModule module
callee {
  p0 = f32[] parameter(0)
  ROOT neg = f32[] negate(p0)
}
ENTRY entry {
  p0 = f32[] parameter(0)
  ROOT call = f32[] call(p0), to_apply=callee, frontend_attributes={inlineable="xla_late"}
}
)";

  ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnVerifiedModule(hlo_string));

  EXPECT_TRUE(MultiModuleDriver::ShouldProcess(*module));
}

TEST_F(MultiModuleDriverTest, DeduplicatesLargeXlaLateComputations) {
  const char* hlo_string = R"(
HloModule module
callee1 {
  p0 = f32[300]{0} parameter(0)
  c = f32[] constant(1.0)
  b = f32[300]{0} broadcast(c), dimensions={}
  c_large = f32[10,30]{1,0} constant({
    {1,2,3,4,5,6,7,8,9,10,1,2,3,4,5,6,7,8,9,10,1,2,3,4,5,6,7,8,9,10},
    {1,2,3,4,5,6,7,8,9,10,1,2,3,4,5,6,7,8,9,10,1,2,3,4,5,6,7,8,9,10},
    {1,2,3,4,5,6,7,8,9,10,1,2,3,4,5,6,7,8,9,10,1,2,3,4,5,6,7,8,9,10},
    {1,2,3,4,5,6,7,8,9,10,1,2,3,4,5,6,7,8,9,10,1,2,3,4,5,6,7,8,9,10},
    {1,2,3,4,5,6,7,8,9,10,1,2,3,4,5,6,7,8,9,10,1,2,3,4,5,6,7,8,9,10},
    {1,2,3,4,5,6,7,8,9,10,1,2,3,4,5,6,7,8,9,10,1,2,3,4,5,6,7,8,9,10},
    {1,2,3,4,5,6,7,8,9,10,1,2,3,4,5,6,7,8,9,10,1,2,3,4,5,6,7,8,9,10},
    {1,2,3,4,5,6,7,8,9,10,1,2,3,4,5,6,7,8,9,10,1,2,3,4,5,6,7,8,9,10},
    {1,2,3,4,5,6,7,8,9,10,1,2,3,4,5,6,7,8,9,10,1,2,3,4,5,6,7,8,9,10},
    {1,2,3,4,5,6,7,8,9,10,1,2,3,4,5,6,7,8,9,10,1,2,3,4,5,6,7,8,9,10}
  })
  r = f32[300]{0} reshape(c_large)
  a = f32[300]{0} add(p0, r)
  ROOT out = f32[300]{0} add(a, b)
}
callee2 {
  p0 = f32[300]{0} parameter(0)
  c = f32[] constant(1.0)
  b = f32[300]{0} broadcast(c), dimensions={}
  c_large = f32[10,30]{1,0} constant({
    {1,2,3,4,5,6,7,8,9,10,1,2,3,4,5,6,7,8,9,10,1,2,3,4,5,6,7,8,9,10},
    {1,2,3,4,5,6,7,8,9,10,1,2,3,4,5,6,7,8,9,10,1,2,3,4,5,6,7,8,9,10},
    {1,2,3,4,5,6,7,8,9,10,1,2,3,4,5,6,7,8,9,10,1,2,3,4,5,6,7,8,9,10},
    {1,2,3,4,5,6,7,8,9,10,1,2,3,4,5,6,7,8,9,10,1,2,3,4,5,6,7,8,9,10},
    {1,2,3,4,5,6,7,8,9,10,1,2,3,4,5,6,7,8,9,10,1,2,3,4,5,6,7,8,9,10},
    {1,2,3,4,5,6,7,8,9,10,1,2,3,4,5,6,7,8,9,10,1,2,3,4,5,6,7,8,9,10},
    {1,2,3,4,5,6,7,8,9,10,1,2,3,4,5,6,7,8,9,10,1,2,3,4,5,6,7,8,9,10},
    {1,2,3,4,5,6,7,8,9,10,1,2,3,4,5,6,7,8,9,10,1,2,3,4,5,6,7,8,9,10},
    {1,2,3,4,5,6,7,8,9,10,1,2,3,4,5,6,7,8,9,10,1,2,3,4,5,6,7,8,9,10},
    {1,2,3,4,5,6,7,8,9,10,1,2,3,4,5,6,7,8,9,10,1,2,3,4,5,6,7,8,9,10}
  })
  r = f32[300]{0} reshape(c_large)
  a = f32[300]{0} add(p0, r)
  ROOT out = f32[300]{0} add(a, b)
}
ENTRY entry {
  p0 = f32[300]{0} parameter(0)
  call1 = f32[300]{0} call(p0), to_apply=callee1,
    frontend_attributes={inlineable="xla_late"}
  ROOT call2 = f32[300]{0} call(call1), to_apply=callee2,
    frontend_attributes={inlineable="xla_late"}
}
)";
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<VerifiedHloModule> module,
                       ParseAndReturnVerifiedModule(hlo_string));

  cpu::CpuCompiler compiler;
  auto options = Compiler::CompileOptions();

  ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<HloModule> optimized_module,
      compiler.RunHloPasses(std::move(module), nullptr, options));

  ASSERT_OK_AND_ASSIGN(bool filecheck_result,
                       RunFileCheck(optimized_module->ToString(), R"(
// CHECK-NOT: callee2
// CHECK: call(
// CHECK: call(
)"));
  EXPECT_TRUE(filecheck_result);
}

// Regression test for b/576620699: when HloModuleSplitter extracts an
// inlineable="xla_late" callee containing nested calls with stack-frame
// metadata, CallInliner must be able to concatenate stack frames inside the
// extracted submodule without creating a self-loop in StackFrames, and
// HloModuleStitcher must intern the resulting frames back into the main module.
TEST_F(MultiModuleDriverTest, XlaLateWithNestedCallsAndStackFramesDoesNotHang) {
  const char* hlo_string = R"(
HloModule module

FileNames
1 "file.py"

FunctionNames
1 "f1"
2 "f2"
3 "f3"

FileLocations
1 {file_name_id=1 function_name_id=1 line=10 end_line=10 column=1 end_column=5}
2 {file_name_id=1 function_name_id=2 line=20 end_line=20 column=1 end_column=5}
3 {file_name_id=1 function_name_id=3 line=30 end_line=30 column=1 end_column=5}

StackFrames
1 {file_location_id=1 parent_frame_id=1}
2 {file_location_id=2 parent_frame_id=1}
3 {file_location_id=3 parent_frame_id=1}

leaf2 {
  p0 = f32[4]{0} parameter(0)
  ROOT exp = f32[4]{0} exponential(p0), metadata={op_name="exp" stack_frame_id=1}
}

leaf1 {
  p0 = f32[4]{0} parameter(0)
  ROOT neg = f32[4]{0} negate(p0), metadata={op_name="neg" stack_frame_id=2}
}

late_callee {
  p0 = f32[4]{0} parameter(0)
  c1 = f32[4]{0} call(p0), to_apply=leaf1, metadata={op_name="c1" stack_frame_id=1}
  ROOT c2 = f32[4]{0} call(c1), to_apply=leaf2, metadata={op_name="c2" stack_frame_id=3}
}

ENTRY entry {
  p0 = f32[4]{0} parameter(0)
  ROOT late_call = f32[4]{0} call(p0), to_apply=late_callee,
    frontend_attributes={inlineable="xla_late"},
    metadata={op_name="late_call" stack_frame_id=1}
}
)";
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<VerifiedHloModule> module,
                       ParseAndReturnVerifiedModule(hlo_string));

  cpu::CpuCompiler compiler;
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> optimized_module,
                       compiler.RunHloPasses(std::move(module), nullptr,
                                             Compiler::CompileOptions()));

  // After splitting, inlining leaf1/leaf2 inside late_callee, stitching back,
  // and late-inlining late_call into entry, the instructions should carry
  // valid concatenated stack traces.
  bool found_concatenated_chain = false;
  for (const HloComputation* comp : optimized_module->computations()) {
    for (const HloInstruction* inst : comp->instructions()) {
      if (inst->metadata().stack_frame_id() > 0) {
        std::vector<HloStackFrame> trace = inst->GetStackTraceFromMetadata();
        EXPECT_FALSE(trace.empty());
        if (trace.size() >= 2) {
          found_concatenated_chain = true;
        }
      }
    }
  }
  EXPECT_TRUE(found_concatenated_chain);
}

}  // namespace
}  // namespace xla
