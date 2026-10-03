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

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "llvm/AsmParser/Parser.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Module.h"
#include "llvm/Support/SourceMgr.h"
#include "xla/debug_options_flags.h"
#include "xla/service/gpu/llvm_gpu_backend/amdgpu_backend.h"
#include "xla/stream_executor/gpu/asm_compiler.h"
#include "xla/stream_executor/platform.h"
#include "xla/stream_executor/platform_manager.h"
#include "xla/stream_executor/stream_executor.h"
#include "xla/tsl/platform/env.h"
#include "xla/tsl/platform/rocm_rocdl_path.h"
#include "xla/tsl/platform/statusor.h"
#include "tsl/platform/path.h"

namespace stream_executor {
namespace {

TEST(GpuHsacoBundleTest, CompileToHsacoThenBundleGpuAsmWorks) {
  const std::string rocm_root = tsl::RocmRoot();
  if (rocm_root.empty()) {
    GTEST_SKIP() << "ROCm root is empty; clang-offload-bundler is unavailable";
  }
  const std::string bundler =
      tsl::io::JoinPath(rocm_root, "llvm/bin/clang-offload-bundler");
  if (!tsl::Env::Default()->FileExists(bundler).ok()) {
    GTEST_SKIP() << "clang-offload-bundler not found at " << bundler;
  }

  ASSERT_OK_AND_ASSIGN(Platform * platform,
                       PlatformManager::PlatformWithName("ROCM"));
  ASSERT_GT(platform->VisibleDeviceCount(), 0);
  ASSERT_OK_AND_ASSIGN(StreamExecutor * executor,
                       platform->ExecutorForDevice(0));
  const std::string gfx_arch =
      executor->GetDeviceDescription().rocm_compute_capability().gfx_version();
  ASSERT_FALSE(gfx_arch.empty());

  // CompileToHsaco takes an LLVM module; this kernel has no host or ROCDL deps.
  constexpr char kIr[] = R"(
target triple = "amdgcn-amd-amdhsa"
define amdgpu_kernel void @simple_add(ptr addrspace(1) %a, ptr addrspace(1) %b,
                                      ptr addrspace(1) %c) {
entry:
  %tid = call i32 @llvm.amdgcn.workitem.id.x()
  %tidx = zext i32 %tid to i64
  %a_ptr = getelementptr float, ptr addrspace(1) %a, i64 %tidx
  %b_ptr = getelementptr float, ptr addrspace(1) %b, i64 %tidx
  %c_ptr = getelementptr float, ptr addrspace(1) %c, i64 %tidx
  %a_val = load float, ptr addrspace(1) %a_ptr, align 4
  %b_val = load float, ptr addrspace(1) %b_ptr, align 4
  %sum = fadd float %a_val, %b_val
  store float %sum, ptr addrspace(1) %c_ptr, align 4
  ret void
}
declare i32 @llvm.amdgcn.workitem.id.x()
)";

  llvm::LLVMContext llvm_context;
  llvm::SMDiagnostic parse_err;
  std::unique_ptr<llvm::Module> module =
      llvm::parseAssemblyString(kIr, parse_err, llvm_context);
  ASSERT_NE(module, nullptr) << parse_err.getMessage().str();
  // EmitModuleToHsaco joins the module id into a temp path.
  module->setModuleIdentifier("gpu_hsaco_bundle_test");

  ASSERT_OK_AND_ASSIGN(
      xla::gpu::amdgpu::HsacoResult hsaco_result,
      xla::gpu::amdgpu::CompileToHsaco(
          module.get(),
          executor->GetDeviceDescription().gpu_compute_capability(),
          xla::GetDebugOptionsFromFlags(), "gpu_hsaco_bundle_test"));
  ASSERT_FALSE(hsaco_result.hsaco.empty());

  HsacoImage image;
  image.gfx_arch = gfx_arch;
  image.bytes = std::move(hsaco_result.hsaco);
  ASSERT_OK_AND_ASSIGN(std::vector<uint8_t> bundle,
                       BundleGpuAsm({image}, rocm_root));
  ASSERT_FALSE(bundle.empty());
}

}  // namespace
}  // namespace stream_executor
