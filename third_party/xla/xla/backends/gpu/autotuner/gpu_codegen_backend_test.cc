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

#include "xla/backends/gpu/autotuner/gpu_codegen_backend.h"

#include <gtest/gtest.h>
#include "xla/xla.pb.h"

namespace xla {
namespace gpu {
namespace {

class GpuCodegenBackendTest : public ::testing::Test {};

TEST_F(GpuCodegenBackendTest, AdjustDebugOptionsForAutotuning) {
  DebugOptions debug_options;
  debug_options.set_xla_enable_dumping(true);
  debug_options.set_xla_gpu_force_compilation_parallelism(4);
  debug_options.set_xla_gpu_enable_llvm_module_compilation_parallelism(true);
  debug_options.add_xla_gpu_enable_command_buffer(DebugOptions::FUSION);
  debug_options.set_xla_gpu_async_dot(true);
  debug_options.set_xla_embed_ir_in_executable(true);
  debug_options.set_xla_gpu_kernel_cache_file("foo.txt");
  debug_options.set_xla_gpu_filter_kernels_spilling_registers_on_autotuning(
      true);
  debug_options.set_xla_gpu_fail_ptx_compilation_on_register_spilling(false);

  GpuCodegenBackend::AdjustDebugOptionsForAutotuning(
      debug_options, /*force_allow_register_spills=*/false);

  EXPECT_FALSE(debug_options.xla_enable_dumping());
  EXPECT_EQ(debug_options.xla_gpu_force_compilation_parallelism(), 1);
  EXPECT_FALSE(
      debug_options.xla_gpu_enable_llvm_module_compilation_parallelism());
  EXPECT_TRUE(debug_options.xla_gpu_enable_command_buffer().empty());
  EXPECT_FALSE(debug_options.xla_gpu_async_dot());
  EXPECT_FALSE(debug_options.xla_embed_ir_in_executable());
  EXPECT_EQ(debug_options.xla_gpu_kernel_cache_file(), "");
  EXPECT_TRUE(
      debug_options.xla_gpu_fail_ptx_compilation_on_register_spilling());
}

TEST_F(GpuCodegenBackendTest, AdjustDebugOptionsForAutotuningAllowSpilling) {
  DebugOptions debug_options;
  debug_options.set_xla_gpu_filter_kernels_spilling_registers_on_autotuning(
      true);
  debug_options.set_xla_gpu_fail_ptx_compilation_on_register_spilling(false);
  GpuCodegenBackend::AdjustDebugOptionsForAutotuning(
      debug_options, /*force_allow_register_spills=*/true);
  EXPECT_FALSE(
      debug_options.xla_gpu_fail_ptx_compilation_on_register_spilling());
}

TEST_F(GpuCodegenBackendTest,
       AdjustDebugOptionsForAutotuningKeepsAlreadySetFailOnSpilling) {
  DebugOptions debug_options;
  debug_options.set_xla_gpu_filter_kernels_spilling_registers_on_autotuning(
      false);
  debug_options.set_xla_gpu_fail_ptx_compilation_on_register_spilling(true);
  GpuCodegenBackend::AdjustDebugOptionsForAutotuning(
      debug_options, /*force_allow_register_spills=*/false);
  EXPECT_TRUE(
      debug_options.xla_gpu_fail_ptx_compilation_on_register_spilling());
  GpuCodegenBackend::AdjustDebugOptionsForAutotuning(
      debug_options, /*force_allow_register_spills=*/true);
  EXPECT_TRUE(
      debug_options.xla_gpu_fail_ptx_compilation_on_register_spilling());
}

}  // namespace
}  // namespace gpu
}  // namespace xla
