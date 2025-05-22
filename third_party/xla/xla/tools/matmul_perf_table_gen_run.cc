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

#include <string>
#include <utility>
#include <vector>

#include "absl/log/check.h"
#include "absl/strings/str_cat.h"
#include "xla/service/gpu/model/hlo_op_profiles.h"
#include "xla/service/hlo_runner.h"
#include "xla/service/platform_util.h"
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

int RunPerfTableCollection(int argc, char** argv) {
  tsl::port::InitMain(kUsage, &argc, &argv);

  MatmulPerfTableGen::Config cfg;
  cfg.b_spec = {/*start=*/1, /*stop=*/8, /*step=*/0, /*factor=*/2};
  cfg.m_spec = {/*start=*/16, /*stop=*/4096, /*step=*/0, /*factor=*/2};
  cfg.n_spec = {/*start=*/16, /*stop=*/4096, /*step=*/0, /*factor=*/2};
  cfg.k_spec = {/*start=*/16, /*stop=*/4096, /*step=*/0, /*factor=*/2};
  cfg.dtypes = {
      {
          /*lhs_dtype=*/"bf16",
          /*rhs_dtype=*/"bf16",
          /*out_dtype=*/"bf16",
      },
  };

  {
    HloRunner runner(PlatformUtil::GetPlatform("cuda").value());
    const se::DeviceDescription& device_info =
        runner.backend().stream_executors()[0]->GetDeviceDescription();
    cfg.output = FilepathOutput(HloOpProfiles::GetProfileName(device_info));
  }
  MatmulPerfTableGen table_gen(std::move(cfg));

  xla::gpu::DeviceHloInstructionProfiles result = table_gen.ComputeTable();
  CHECK_OK(table_gen.Dump(result));

  return 0;
}

}  // namespace
}  // namespace xla::gpu

int main(int argc, char** argv) {
  return xla::gpu::RunPerfTableCollection(argc, argv);
}
