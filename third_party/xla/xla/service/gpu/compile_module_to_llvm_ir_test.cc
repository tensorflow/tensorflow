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

#include "xla/service/gpu/compile_module_to_llvm_ir.h"

#include <atomic>
#include <cstdint>
#include <memory>
#include <utility>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/status/status.h"
#include "absl/status/status_matchers.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "absl/synchronization/notification.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Module.h"
#include "llvm/TargetParser/Triple.h"
#include "xla/backends/cpu/target_machine_options.h"
#include "xla/backends/gpu/codegen/cubin_custom_kernel_compiler.h"
#include "xla/backends/gpu/codegen/emitters/mlir_kernel_emitter.h"
#include "xla/hlo/testlib/hlo_hardware_independent_test_base.h"
#include "xla/hlo/testlib/verified_hlo_module.h"
#include "xla/service/buffer_value.h"
#include "xla/service/gpu/alias_info.h"
#include "xla/service/gpu/mlir_context_pool.h"
#include "xla/service/gpu/target_constants.h"
#include "xla/service/gpu_topology.h"
#include "xla/service/llvm_ir/llvm_command_line_options.h"
#include "xla/shape_util.h"
#include "xla/stream_executor/cuda/cuda_platform_id.h"
#include "xla/stream_executor/device_description.h"
#include "xla/tsl/platform/env.h"
#include "xla/tsl/platform/threadpool.h"
#include "xla/xla.pb.h"

namespace xla::gpu {
namespace {

using ::absl_testing::StatusIs;
using ::testing::HasSubstr;

constexpr absl::string_view kScheduledHloWithConstantAndFusion = R"hlo(
HloModule test, is_scheduled=true

fused_computation {
  p0 = f32[4] parameter(0)
  p1 = f32[4] parameter(1)
  ROOT add = f32[4] add(p0, p1)
}

ENTRY main {
  p0 = f32[4] parameter(0)
  c = f32[4] constant({1, 2, 3, 4})
  ROOT fusion = f32[4] fusion(p0, c), kind=kLoop, calls=fused_computation
}
)hlo";

using CompileModuleToLlvmIrTest = HloHardwareIndependentTestBase;

TEST_F(CompileModuleToLlvmIrTest,
       AwaitsInFlightThunkCompilationWhenConstantsModuleCompilationFails) {
  ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<VerifiedHloModule> module,
      ParseAndReturnVerifiedModule(kScheduledHloWithConstantAndFusion));
  ASSERT_OK_AND_ASSIGN(
      GpuTopology gpu_topology,
      GetGpuTopologyForPlatform("nvidia_h100", /*num_partitions=*/1,
                                /*num_hosts_per_partition=*/1,
                                /*num_devices_per_host=*/1));

  absl::Notification constants_compile_started;
  std::atomic<bool> fusion_compile_finished = false;
  auto llvm_compiler =
      [&](llvm::Module& llvm_module, const se::DeviceDescription& descr,
          const DebugOptions& opts) -> absl::StatusOr<std::vector<uint8_t>> {
    if (llvm_module.empty()) {
      constants_compile_started.Notify();
      return absl::InternalError("Failed to compile constants module");
    }
    constants_compile_started.WaitForNotification();
    fusion_compile_finished.store(true);
    return std::vector<uint8_t>{1};
  };

  tsl::thread::ThreadPool thread_pool(tsl::Env::Default(), "compile", 2);
  CubinCustomKernelCompiler kernel_compiler(
      std::move(llvm_compiler),
      gpu_topology.gpu_target_config().device_description,
      module->config().debug_options(), gpu_topology,
      llvm::Triple(nvptx::TargetTriple()), nvptx::DataLayout(), &thread_pool);

  llvm::LLVMContext llvm_context;
  GpuAliasInfo alias_info(gpu_topology.gpu_target_config().device_description);
  BufferValue::SizeFunction buffer_size_bytes_function =
      [](const BufferValue& buffer_value) {
        return ShapeUtil::ByteSizeOf(buffer_value.shape(), /*pointer_size=*/8);
      };
  llvm_ir::LLVMCommandLineOptionsReleasableLock llvm_options_lock({});
  MlirContextPool mlir_context_pool([]() { return CreateMlirContext(); });

  EXPECT_THAT(CompileModuleToLlvmIr(
                  module.get(), &llvm_context, nvptx::TargetTriple(),
                  nvptx::DataLayout(), se::cuda::kCudaPlatformId, gpu_topology,
                  &alias_info, std::move(buffer_size_bytes_function),
                  llvm_options_lock, &kernel_compiler,
                  xla::cpu::TargetMachineOptions{}, &mlir_context_pool),
              StatusIs(absl::StatusCode::kInternal,
                       HasSubstr("Failed to compile constants module")));
  EXPECT_TRUE(fusion_compile_finished.load());
}

TEST_F(CompileModuleToLlvmIrTest,
       AwaitsInFlightConstantsCompilationWhenThunkCompilationFails) {
  ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<VerifiedHloModule> module,
      ParseAndReturnVerifiedModule(kScheduledHloWithConstantAndFusion));
  ASSERT_OK_AND_ASSIGN(
      GpuTopology gpu_topology,
      GetGpuTopologyForPlatform("nvidia_h100", /*num_partitions=*/1,
                                /*num_hosts_per_partition=*/1,
                                /*num_devices_per_host=*/1));

  absl::Notification fusion_compile_started;
  std::atomic<bool> constants_compile_finished = false;
  auto llvm_compiler =
      [&](llvm::Module& llvm_module, const se::DeviceDescription& descr,
          const DebugOptions& opts) -> absl::StatusOr<std::vector<uint8_t>> {
    if (llvm_module.empty()) {
      fusion_compile_started.WaitForNotification();
      constants_compile_finished.store(true);
      return std::vector<uint8_t>{1};
    }
    fusion_compile_started.Notify();
    return absl::InternalError("Failed to compile fusion module");
  };

  tsl::thread::ThreadPool thread_pool(tsl::Env::Default(), "compile", 2);
  CubinCustomKernelCompiler kernel_compiler(
      std::move(llvm_compiler),
      gpu_topology.gpu_target_config().device_description,
      module->config().debug_options(), gpu_topology,
      llvm::Triple(nvptx::TargetTriple()), nvptx::DataLayout(), &thread_pool);

  llvm::LLVMContext llvm_context;
  GpuAliasInfo alias_info(gpu_topology.gpu_target_config().device_description);
  BufferValue::SizeFunction buffer_size_bytes_function =
      [](const BufferValue& buffer_value) {
        return ShapeUtil::ByteSizeOf(buffer_value.shape(), /*pointer_size=*/8);
      };
  llvm_ir::LLVMCommandLineOptionsReleasableLock llvm_options_lock({});
  MlirContextPool mlir_context_pool([]() { return CreateMlirContext(); });

  EXPECT_THAT(CompileModuleToLlvmIr(
                  module.get(), &llvm_context, nvptx::TargetTriple(),
                  nvptx::DataLayout(), se::cuda::kCudaPlatformId, gpu_topology,
                  &alias_info, std::move(buffer_size_bytes_function),
                  llvm_options_lock, &kernel_compiler,
                  xla::cpu::TargetMachineOptions{}, &mlir_context_pool),
              StatusIs(absl::StatusCode::kInternal,
                       HasSubstr("Failed to compile fusion module")));
  EXPECT_TRUE(constants_compile_finished.load());
}

}  // namespace
}  // namespace xla::gpu
