/* Copyright 2018 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/compiler/xla/service/gpu/tests/gpu_codegen_test.h"

#include <memory>

#include "absl/strings/str_replace.h"
#include "tensorflow/compiler/xla/debug_options_flags.h"
#include "tensorflow/compiler/xla/service/gpu/gpu_executable.h"
#include "tensorflow/compiler/xla/shape_util.h"
#include "tensorflow/compiler/xla/tests/filecheck.h"
#include "tensorflow/compiler/xla/tests/verified_hlo_module.h"

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
  std::unique_ptr<Executable> executable =
      std::move(CompileToExecutable(std::move(hlo_module)).value());
  std::string ptx_str(static_cast<GpuExecutable*>(executable.get())->text());

  // On the ROCM platform the "ptx" string is not populated for the compiled
  // executable, and hence the "ptx_str" will be empty. So disabling the
  // pattern check on the ROCm platform
  if (!is_built_with_rocm_) {
    StatusOr<bool> filecheck_result = RunFileCheck(ptx_str, pattern);
    ASSERT_TRUE(filecheck_result.ok());
    EXPECT_TRUE(filecheck_result.value());
  }
}

std::string GpuCodegenTest::MakePlatformSpecificLlvm(absl::string_view input) {
  return absl::StrReplaceAll(
      input,
      {{"KERNEL_ANNOTATION",
        is_built_with_rocm_ ? "amdgpu_kernel void" : "void"},
       {"BARRIER",
        is_built_with_rocm_ ? "@llvm.amdgcn.s.barrier" : "@llvm.nvvm.barrier0"},
       {"SHUFFLE", is_built_with_rocm_ ? "i32 @llvm.amdgcn.ds.bpermute"
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
                                      : "br i1 %[[LOGICAL_T0]]"},
       {
        "FUSION_LDS" , is_built_with_rocm_ ? "llvm.amdgcn.kernel.fusion.lds"
                                               : "0"}});
}

}  // namespace gpu
}  // namespace xla
