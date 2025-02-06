/* Copyright 2018 The OpenXLA Authors.

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

#include "xla/service/gpu/tests/gpu_codegen_test.h"

#include <memory>
#include <string>
#include <utility>

#include "absl/status/statusor.h"
#include "absl/strings/str_replace.h"
#include "absl/strings/string_view.h"
#include "xla/debug_options_flags.h"
#include "xla/hlo/testlib/filecheck.h"
#include "xla/hlo/testlib/verified_hlo_module.h"
#include "xla/service/executable.h"
#include "xla/service/gpu/gpu_executable.h"
#include "xla/service/hlo_module_config.h"
#include "xla/shape_util.h"

namespace xla {
namespace gpu {

std::unique_ptr<VerifiedHloModule>
GpuCodegenTest::CreateNewVerifiedModuleWithFTZ(bool ftz) {
  HloModuleConfig config;
  auto debug_options = GetDebugOptionsFromFlags();
  debug_options.set_xla_gpu_ftz(ftz);
  // TODO(b/38354253): Change tests to use Parameters instead of Constants.
  debug_options.add_xla_disable_hlo_passes("constant_folding");
  config.set_debug_options(debug_options);

  return std::make_unique<VerifiedHloModule>(
      TestName(), config, /*verifier_layout_sensitive=*/true,
      /*allow_mixed_precision_in_hlo_verifier=*/false,
      ShapeUtil::ByteSizeOfElements);
}

void GpuCodegenTest::CompileAndOptionallyVerifyPtx(
    std::unique_ptr<VerifiedHloModule> hlo_module, absl::string_view pattern) {
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<Executable> executable,
                          CompileToExecutable(std::move(hlo_module)));
  std::string ptx_str(static_cast<GpuExecutable*>(executable.get())->text());

  // On the ROCM platform the "ptx" string is not populated for the compiled
  // executable, and hence the "ptx_str" will be empty. So disabling the
  // pattern check on the ROCm platform
  if (!is_built_with_rocm_) {
    absl::StatusOr<bool> filecheck_result = RunFileCheck(ptx_str, pattern);
    ASSERT_TRUE(filecheck_result.ok());
    EXPECT_TRUE(filecheck_result.value());
  }
}

std::string GpuCodegenTest::MakePlatformSpecificLlvm(absl::string_view input) {
  return absl::StrReplaceAll(
      input,
      {{"KERNEL_ANNOTATION",
        is_built_with_rocm_ ? "amdgpu_kernel void" : "ptx_kernel void"},
       {"BARRIER",
        is_built_with_rocm_ ? "@llvm.amdgcn.s.barrier" : "@llvm.nvvm.barrier0"},
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

}  // namespace gpu
}  // namespace xla
