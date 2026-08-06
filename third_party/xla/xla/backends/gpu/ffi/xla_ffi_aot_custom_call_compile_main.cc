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

#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/strings/string_view.h"
#include "xla/tsl/platform/status_macros.h"
#include "xla/backends/gpu/target_config/target_config.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/parser/hlo_parser.h"
#include "xla/pjrt/pjrt_compiler.h"
#include "xla/service/compiled_module.h"
#include "xla/service/compiler.h"
#include "xla/service/gpu_topology.h"
#include "xla/stream_executor/cuda/cuda_platform_id.h"
#include "xla/tsl/platform/env.h"
#include "tsl/platform/init_main.h"

namespace xla::gpu {
namespace {

constexpr absl::string_view kHloText = R"hlo(
HloModule TestAddI32Module

ENTRY main {
  ROOT result = s32[4] custom-call(),
    custom_call_target="xla.gpu.test_write_42_aot",
    api_version=API_VERSION_TYPED_FFI
}
)hlo";

absl::Status CompileAndWriteExecutable(absl::string_view output_path) {
  ABSL_ASSIGN_OR_RETURN(
      stream_executor::GpuTargetConfigProto gpu_target_config_proto,
      GetGpuTargetConfig(GpuModel::H100_SXM));
  ABSL_ASSIGN_OR_RETURN(GpuTargetConfig gpu_target_config,
                   GpuTargetConfig::FromProto(gpu_target_config_proto));

  ABSL_ASSIGN_OR_RETURN(
      std::unique_ptr<Compiler> compiler,
      Compiler::GetForPlatform(stream_executor::cuda::kCudaPlatformId));

  AotCompilationOptions aot_options(compiler->PlatformId());
  aot_options.set_gpu_topology(
      GetSingleDeviceGpuTopology(CudaName(), gpu_target_config));

  ABSL_ASSIGN_OR_RETURN(std::unique_ptr<HloModule> hlo_module,
                   ParseAndReturnUnverifiedModule(kHloText, {}));

  ABSL_ASSIGN_OR_RETURN(
      std::vector<std::unique_ptr<CompiledModule>> aot_results,
      compiler->CompileAheadOfTime(std::move(hlo_module), aot_options));

  ABSL_ASSIGN_OR_RETURN(std::string serialized_executable,
                   aot_results[0]->SerializeAsString());

  return tsl::WriteStringToFile(tsl::Env::Default(), output_path,
                                serialized_executable);
}

}  // namespace
}  // namespace xla::gpu

int main(int argc, char** argv) {
  tsl::port::InitMain(argv[0], &argc, &argv);
  if (argc < 2) {
    LOG(FATAL) << "Usage: " << argv[0] << " <output_file>";
  }
  std::string output_path = argv[1];
  absl::Status status = xla::gpu::CompileAndWriteExecutable(output_path);
  QCHECK_OK(status) << "Failed to compile HLO to AOT executable";
  return 0;
}
