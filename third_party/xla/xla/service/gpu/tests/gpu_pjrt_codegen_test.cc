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

#include "xla/service/gpu/tests/gpu_pjrt_codegen_test.h"

#include <memory>
#include <string>
#include <utility>

#include <gtest/gtest.h>
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_replace.h"
#include "absl/strings/string_view.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/testlib/filecheck.h"
#include "xla/hlo/testlib/hlo_hardware_independent_test_base.h"
#include "xla/hlo/testlib/verified_hlo_module.h"
#include "xla/service/executable.h"
#include "xla/service/gpu/gpu_executable.h"
#include "xla/service/hlo_module_config.h"
#include "xla/service/llvm_compiler.h"
#include "xla/shape_util.h"
#include "xla/tests/codegen_utils.h"
#include "xla/tsl/platform/statusor.h"
#include "xla/xla.pb.h"
#include "tsl/platform/casts.h"

namespace xla {
namespace gpu {

std::unique_ptr<VerifiedHloModule>
GpuPjRtCodegenTest::CreateNewVerifiedModuleWithFTZ(bool ftz) {
  HloModuleConfig config;
  DebugOptions debug_options =
      HloHardwareIndependentTestBase::GetDebugOptionsForTest();
  debug_options.set_xla_gpu_ftz(ftz);
  config.set_debug_options(debug_options);

  return std::make_unique<VerifiedHloModule>(
      TestName(), config, /*verifier_layout_sensitive=*/true,
      /*allow_mixed_precision_in_hlo_verifier=*/false,
      ShapeUtil::ByteSizeOfElements);
}

void GpuPjRtCodegenTest::CompileAndOptionallyVerifyPtx(
    std::unique_ptr<VerifiedHloModule> hlo_module, absl::string_view pattern,
    bool run_optimization_passes) {
  TF_ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<Executable> executable,
      CompileToExecutable(std::move(hlo_module), run_optimization_passes));
  std::string ptx_str(
      absl::down_cast<GpuExecutable*>(executable.get())->text());

  // On the ROCM platform the "ptx" string is not populated for the compiled
  // executable, and hence the "ptx_str" will be empty. So disabling the
  // pattern check on the ROCm platform
  if (!is_built_with_rocm_) {
    absl::StatusOr<bool> filecheck_result = RunFileCheck(ptx_str, pattern);
    ASSERT_TRUE(filecheck_result.ok());
    EXPECT_TRUE(filecheck_result.value());
  }
}

std::string GpuPjRtCodegenTest::MakePlatformSpecificLlvm(
    absl::string_view input) {
  return absl::StrReplaceAll(
      input,
      {{"KERNEL_ANNOTATION",
        is_built_with_rocm_ ? "amdgpu_kernel void" : "ptx_kernel void"},
       {"BARRIER()", is_built_with_rocm_
                         ? "@llvm.amdgcn.s.barrier()"
                         : "@llvm.nvvm.barrier.cta.sync.aligned.all(i32 0)"},
       {"SHUFFLE", is_built_with_rocm_ ? "i32 @llvm.amdgcn.ds.swizzle"
                                       : "float @llvm.nvvm.shfl.sync.down.f32"},
       {"TIDX", is_built_with_rocm_ ? "@llvm.amdgcn.workitem.id.x"
                                    : "@llvm.nvvm.read.ptx.sreg.tid.x"},
       {"LCAL", is_built_with_rocm_ ? "%[[LOGICAL_T1:.*]] = call { i1, i64 } "
                                      "@llvm.amdgcn.if.i64(i1 %[[LOGICAL_T0]])"
                                    : "0"},
       {"EXTV",
        is_built_with_rocm_
            ? "%[[LOGICAL_T2:.*]] = extractvalue { i1, i64 } %[[LOGICAL_T1]], 0"
            : "0"},
       {"BR_CAL", is_built_with_rocm_ ? "br i1 %[[LOGICAL_T2]],"
                                      : "br i1 %[[LOGICAL_T0]]"}});
}

absl::StatusOr<std::unique_ptr<Executable>>
GpuPjRtCodegenTest::CompileToExecutable(std::unique_ptr<HloModule> hlo_module,
                                        bool run_optimization_passes) {
  return xla::CompileToExecutable(compiler(), compile_options_,
                                  std::move(hlo_module),
                                  run_optimization_passes);
}

absl::Status GpuPjRtCodegenTest::CompileAndVerifyIr(
    absl::string_view hlo_text, absl::string_view expected_llvm_ir,
    bool match_optimized_ir, bool run_optimization_passes) {
  TF_ASSIGN_OR_RETURN(std::unique_ptr<HloModule> hlo_module,
                      ParseAndReturnVerifiedModule(hlo_text));
  auto llvm_compiler = absl::down_cast<LLVMCompiler*>(compiler());
  return xla::CompileAndVerifyIr(llvm_compiler, compile_options_,
                                 std::move(hlo_module), expected_llvm_ir,
                                 match_optimized_ir, run_optimization_passes);
}

}  // namespace gpu
}  // namespace xla
