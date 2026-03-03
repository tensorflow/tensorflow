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

#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "absl/log/check.h"
#include "absl/strings/str_cat.h"
#include "xla/backends/gpu/target_config/target_config.h"
#include "xla/pjrt/pjrt_client.h"
#include "xla/pjrt/plugin/xla_gpu/xla_gpu_allocator_config.h"
#include "xla/pjrt/plugin/xla_gpu/xla_gpu_client_options.h"
#include "xla/pjrt/plugin/xla_gpu/xla_gpu_pjrt_client.h"
#include "xla/service/gpu/model/hlo_op_profile.pb.h"
#include "xla/service/gpu/model/hlo_op_profiles.h"
#include "xla/service/hlo_runner_interface.h"
#include "xla/service/hlo_runner_pjrt.h"
#include "xla/service/pjrt_gpu_utils.h"
#include "xla/stream_executor/device_description.h"
#include "xla/tools/matmul_perf_table_gen.h"
#include "tsl/platform/init_main.h"
#include "tsl/platform/path.h"

namespace xla::gpu {
namespace {

constexpr char kUsage[] = R"(
  Collect performance table for HLO dot op.
)";

std::string FilepathOutput(const std::string& sm_ver) {
  std::string output_directory;
  CHECK(tsl::io::GetTestUndeclaredOutputsDir(&output_directory));
  return tsl::io::JoinPath(output_directory,
                           absl::StrCat("dot-perf-table-", sm_ver, ".pbtxt"));
}

std::pair<std::unique_ptr<HloRunnerInterface>,
          stream_executor::DeviceDescription>
MakeRunnerAndGetDeviceDescription() {
  GpuAllocatorConfig gpu_config;
  gpu_config.kind = GpuAllocatorConfig::Kind::kDefault;
  gpu_config.preallocate = false;
  gpu_config.collective_memory_size = 0;
  GpuClientOptions options;
  options.allocator_config = std::move(gpu_config);
  options.use_tfrt_gpu_client = true;

  absl::StatusOr<std::unique_ptr<PjRtClient>> client =
      GetXlaPjrtGpuClient(options);
  CHECK_OK(client);
  GpuTargetConfig gpu_target_config = GetGpuTargetConfig(client->get());
  return {std::make_unique<HloRunnerPjRt>(*std::move(client)),
          gpu_target_config.device_description};
}

int RunPerfTableCollection(int argc, char** argv) {
  tsl::port::InitMain(kUsage, &argc, &argv);

  auto [runner, device_description] = MakeRunnerAndGetDeviceDescription();

  MatmulPerfTableGen::Config cfg;
  cfg.b_spec = {/*start=*/1, /*stop=*/4, /*step=*/0, /*factor=*/2};
  cfg.m_spec = {/*start=*/256, /*stop=*/4096, /*step=*/0, /*factor=*/2};
  cfg.n_spec = {/*start=*/256, /*stop=*/4096, /*step=*/0, /*factor=*/2};
  cfg.k_spec = {/*start=*/256, /*stop=*/4096, /*step=*/0, /*factor=*/2};
  cfg.dtypes = {
      {
          /*lhs_dtype=*/"bf16",
          /*rhs_dtype=*/"bf16",
          /*out_dtype=*/"bf16",
      },
      {
          /*lhs_dtype=*/"f32",
          /*rhs_dtype=*/"f32",
          /*out_dtype=*/"f32",
      },
  };
  cfg.output =
      FilepathOutput(HloOpProfiles::GetProfileName(device_description));
  MatmulPerfTableGen table_gen(runner.get(), &device_description,
                               std::move(cfg));

  DeviceHloInstructionProfiles result = table_gen.ComputeTable();
  absl::StatusOr<GemmPerfTable> compact_result =
      MatmulPerfTableGen::Compact(result);
  CHECK_OK(compact_result);
  CHECK_OK(table_gen.Dump(*compact_result));

  return 0;
}

}  // namespace
}  // namespace xla::gpu

int main(int argc, char** argv) {
  return xla::gpu::RunPerfTableCollection(argc, argv);
}
