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
#include "xla/backends/gpu/codegen/cubin_custom_kernel_compiler.h"

#include <cstdint>
#include <memory>
#include <utility>
#include <vector>

#include <gtest/gtest.h>
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "llvm/IR/Module.h"
#include "xla/backends/gpu/runtime/thunk.h"
#include "xla/codegen/emitters/kernel_arguments.h"
#include "xla/codegen/llvm_kernel_source.h"
#include "xla/service/gpu/gpu_device_info_for_tests.h"
#include "xla/service/gpu/launch_dimensions.h"
#include "xla/stream_executor/device_description.h"
#include "xla/tsl/platform/test.h"

namespace xla::gpu {
namespace {

TEST(CubinCustomKernelCompilerTest, CallbackInvoked) {
  int compiler_invoked = 0;
  auto llvm_compiler =
      [&](llvm::Module& llvm_module, const se::DeviceDescription& descr,
          const DebugOptions& opts) -> absl::StatusOr<std::vector<uint8_t>> {
    compiler_invoked++;
    return std::vector<uint8_t>{1};
  };

  DebugOptions debug_options;
  CubinCustomKernelCompiler kernel_compiler(
      llvm_compiler, TestGpuDeviceInfo::H100SXMDeviceInfo(), debug_options);

  int hook_invoked = 0;
  kernel_compiler.SetPreOptimizationHook(
      [&](const llvm::Module&) { hook_invoked++; });

  constexpr int kIterations = 3;
  for (int i = 0; i < kIterations; i++) {
    auto llvm_context = std::make_unique<llvm::LLVMContext>();
    auto llvm_module = std::make_unique<llvm::Module>("Test", *llvm_context);
    LlvmKernelSource kernel_source(std::move(llvm_context),
                                   std::move(llvm_module));
    emitters::KernelArguments kernel_arguments({});

    Thunk::ThunkInfo thunk_info;
    LaunchDimensions dimensions;
    ASSERT_OK_AND_ASSIGN(
        std::unique_ptr<Thunk> thunk,
        kernel_compiler
            .Compile(thunk_info, std::move(kernel_source),
                     absl::StrCat("kernel", i), kernel_arguments, dimensions)
            .Await());
  }

  EXPECT_EQ(kIterations, compiler_invoked);
  EXPECT_EQ(kIterations, hook_invoked);
}

}  // namespace
}  // namespace xla::gpu
