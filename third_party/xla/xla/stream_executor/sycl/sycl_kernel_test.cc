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

#include <cstddef>
#include <cstdint>
#include <memory>

#include <gtest/gtest.h>
#include "absl/types/span.h"
#include "xla/backends/gpu/runtime/kernel_thunk.h"
#include "xla/backends/gpu/runtime/thunk_executor.h"
#include "xla/service/gpu/gpu_executable.h"
#include "xla/stream_executor/device_address.h"
#include "xla/stream_executor/platform_manager.h"
#include "xla/stream_executor/stream_executor.h"
#include "xla/stream_executor/sycl/sycl_platform_id.h"
#include "xla/tests/llvm_irgen_test_base.h"

namespace stream_executor::sycl {
namespace {

class SyclKernelTest : public xla::LlvmIrGenTestBase {};

TEST_F(SyclKernelTest, CheckKernelLoading) {
  TF_ASSERT_OK_AND_ASSIGN(
      Platform * platform,
      stream_executor::PlatformManager::PlatformWithId(kSyclPlatformId));
  TF_ASSERT_OK_AND_ASSIGN(StreamExecutor * executor,
                          platform->ExecutorForDevice(0));

  absl::string_view hlo_ir = R"(
    ENTRY e {
      p0 = u32[4] parameter(0)
      p1 = u32[4] parameter(1)
      ROOT res = u32[4] add(p0, p1)
    })";

  xla::HloModuleConfig config;
  config.set_debug_options(GetDebugOptionsForTest());
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<xla::HloModule> hlo_module,
                          xla::ParseAndReturnUnverifiedModule(hlo_ir, config));

  TF_ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<xla::Executable> exec,
      CompileToExecutable(std::move(hlo_module),
                          /*run_optimization_passes=*/true));

  auto* gpu_exec = static_cast<xla::gpu::GpuExecutable*>(exec.get());
  ASSERT_NE(gpu_exec, nullptr);

  const xla::gpu::ThunkExecutor& thunk_exec = gpu_exec->thunk_executor();
  EXPECT_EQ(thunk_exec.thunks().size(), 1);

  const xla::gpu::Thunk* thunk = thunk_exec.thunks().at(0).get();
  ASSERT_NE(thunk, nullptr);
  EXPECT_EQ(thunk->kind(), xla::gpu::Thunk::Kind::kKernel);

  const auto* kernel_thunk = dynamic_cast<const xla::gpu::KernelThunk*>(thunk);
  ASSERT_NE(kernel_thunk, nullptr);

  std::string kernel_name = kernel_thunk->kernel_name();

  std::vector<uint8_t> spirv_binary(gpu_exec->binary());

  KernelLoaderSpec spec = KernelLoaderSpec::CreateCudaCubinInMemorySpec(
      spirv_binary, kernel_name, 3);

  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<Kernel> sycl_kernel,
                          executor->LoadKernel(spec));

  EXPECT_EQ(sycl_kernel->Arity(), 3);
  // TODO(intel-tf): Add check for GetMaxOccupiedBlocksPerCore
}

}  // namespace
}  // namespace stream_executor::sycl
