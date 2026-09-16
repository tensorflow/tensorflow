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

#include "xla/service/gpu/llvm_gpu_backend/spirv_backend.h"

#include <memory>
#include <set>
#include <string>

#include <gtest/gtest.h>
#include "absl/status/status.h"
#include "absl/status/status_matchers.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "llvm/AsmParser/Parser.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Module.h"
#include "llvm/Support/SourceMgr.h"
#include "xla/stream_executor/device_description.h"
#include "xla/stream_executor/sycl/oneapi_compute_capability.h"
#include "xla/xla.pb.h"

namespace xla::gpu::spirv {
namespace {

stream_executor::GpuComputeCapability TestComputeCapability() {
  return stream_executor::GpuComputeCapability(
      stream_executor::OneAPIComputeCapability::BMG());
}

absl::StatusOr<std::unique_ptr<llvm::Module>> ParseLlvmIr(
    absl::string_view ir, llvm::LLVMContext& context) {
  llvm::SMDiagnostic diagnostic;
  std::unique_ptr<llvm::Module> module =
      llvm::parseAssemblyString(ir, diagnostic, context);
  if (module == nullptr) {
    return absl::InvalidArgumentError(absl::StrCat(
        "Failed to parse LLVM IR: ", diagnostic.getMessage().str()));
  }
  return module;
}

TEST(SpirvBackendTest, TestSPIRVExtensions) {
  auto extensions = SPIRVExtensionsEnumToString(common_spirv_extensions);
  auto extensions_set =
      std::set<std::string>(extensions.begin(), extensions.end());

  EXPECT_NE(extensions_set.find("SPV_EXT_optnone"), extensions_set.end());
  EXPECT_NE(extensions_set.find("SPV_KHR_uniform_group_instructions"),
            extensions_set.end());
  EXPECT_NE(extensions_set.find("SPV_KHR_linkonce_odr"), extensions_set.end());
  EXPECT_NE(extensions_set.find("SPV_KHR_cooperative_matrix"),
            extensions_set.end());
  EXPECT_NE(extensions_set.find("SPV_EXT_shader_atomic_float_add"),
            extensions_set.end());
  EXPECT_EQ(extensions_set.find("SPV_NV_cooperative_matrix"),
            extensions_set.end());
}

TEST(SpirvBackendTest, CompilesKernelWithScalarArguments) {
  llvm::LLVMContext context;
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<llvm::Module> module,
                       ParseLlvmIr(R"(
define spir_kernel void @kernel_argument_rewrite(i32 %value,
                                                 ptr %in,
                                                 ptr addrspace(1) %out) {
entry:
  %in_value = load i32, ptr %in, align 4
  %sum = add i32 %in_value, %value
  store i32 %sum, ptr addrspace(1) %out, align 4
  ret void
}
)",
                                   context));

  EXPECT_OK(
      CompileToSPIRV(module.get(), TestComputeCapability(), DebugOptions()));
}

}  // namespace
}  // namespace xla::gpu::spirv
