/* Copyright 2022 The TensorFlow Authors. All Rights Reserved.

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

#include <memory>
#include <utility>

#include "tensorflow/compiler/xla/service/gpu/nvptx_compiler.h"
#include "tensorflow/compiler/xla/tests/hlo_test_base.h"

namespace xla {
namespace gpu {

using GpuAotCompilationTest = HloTestBase;

TEST_F(GpuAotCompilationTest, LoadExecutableFromAotCompilation) {
  const absl::string_view hlo_string = R"(
HloModule Test

ENTRY main {
  a = f32[100, 200]{1,0} parameter(0)
  ROOT b = f32[100, 200]{0,1} copy(a)
}
)";
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          ParseAndReturnVerifiedModule(hlo_string));

  NVPTXCompiler compiler;
  TF_ASSERT_OK_AND_ASSIGN(se::Platform * platform,
                          se::MultiPlatformManager::PlatformWithName("cuda"));
  TF_ASSERT_OK_AND_ASSIGN(se::StreamExecutor * stream_exec,
                          platform->ExecutorForDevice(0));

  // Compile AOT.
  auto module_group = std::make_unique<HloModuleGroup>(std::move(module));
  AotCompilationOptions aot_options(compiler.PlatformId());
  aot_options.set_executor(stream_exec);
  TF_ASSERT_OK_AND_ASSIGN(
      std::vector<std::unique_ptr<AotCompilationResult>> aot_results,
      compiler.CompileAheadOfTime(std::move(module_group), aot_options));

  // Serialize-deserialize AOT compilation result.
  TF_ASSERT_OK_AND_ASSIGN(std::string serialized_aot_result,
                          aot_results[0]->SerializeAsString());
  TF_ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<AotCompilationResult> aot_result,
      compiler.LoadAotCompilationResult(serialized_aot_result));

  // Load Executable from AOT compilation result.
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<Executable> executable,
                          aot_result->LoadExecutable(&compiler, stream_exec));
}

TEST_F(GpuAotCompilationTest, AotCompilationWithoutGpuDevice) {
  const absl::string_view hlo_string = R"(
HloModule Test

ENTRY main {
  a = f32[100, 200]{1,0} parameter(0)
  ROOT b = f32[100, 200]{0,1} copy(a)
}
)";
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          ParseAndReturnVerifiedModule(hlo_string));

  NVPTXCompiler compiler;
  TF_ASSERT_OK_AND_ASSIGN(se::Platform * platform,
                          se::MultiPlatformManager::PlatformWithName("cuda"));
  TF_ASSERT_OK_AND_ASSIGN(se::StreamExecutor * stream_exec,
                          platform->ExecutorForDevice(0));

  auto module_group = std::make_unique<HloModuleGroup>(std::move(module));

  const stream_executor::DeviceDescription& device_description =
      stream_exec->GetDeviceDescription();
  stream_executor::CudaComputeCapability cuda_compute_capability =
      device_description.cuda_compute_capability();
  stream_executor::RocmComputeCapability rocm_compute_capability =
      device_description.rocm_compute_capability();

  // Stream executor is not passed as an option.
  GpuTargetConfig gpu_target_config;
  gpu_target_config.gpu_device_info = GetGpuDeviceInfo(stream_exec);
  gpu_target_config.gpu_version = cuda_compute_capability;
  gpu_target_config.platform_name = stream_exec->platform()->Name();

  AotCompilationOptions aot_options(compiler.PlatformId());
  aot_options.set_target_config(gpu_target_config);

  TF_ASSERT_OK_AND_ASSIGN(
      std::vector<std::unique_ptr<AotCompilationResult>> aot_results,
      compiler.CompileAheadOfTime(std::move(module_group), aot_options));

  // Serialize-deserialize AOT compilation result.
  TF_ASSERT_OK_AND_ASSIGN(std::string serialized_aot_result,
                          aot_results[0]->SerializeAsString());
  TF_ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<AotCompilationResult> aot_result,
      compiler.LoadAotCompilationResult(serialized_aot_result));

  // Load Executable from AOT compilation result.
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<Executable> executable,
                          aot_result->LoadExecutable(&compiler, stream_exec));
}

}  // namespace gpu
}  // namespace xla
