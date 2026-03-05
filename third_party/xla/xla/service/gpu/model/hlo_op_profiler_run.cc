/* Copyright 2023 The OpenXLA Authors.

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
#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"
#include "absl/strings/string_view.h"
#include "xla/backends/gpu/target_config/target_config.h"
#include "xla/debug_options_flags.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/pjrt/pjrt_client.h"
#include "xla/pjrt/plugin/xla_gpu/xla_gpu_allocator_config.h"
#include "xla/pjrt/plugin/xla_gpu/xla_gpu_client_options.h"
#include "xla/pjrt/plugin/xla_gpu/xla_gpu_pjrt_client.h"
#include "xla/service/gpu/model/hlo_op_profile.pb.h"
#include "xla/service/gpu/model/hlo_op_profiler.h"
#include "xla/service/gpu/model/hlo_op_profiles.h"
#include "xla/service/hlo_runner_interface.h"
#include "xla/service/hlo_runner_pjrt.h"
#include "xla/service/pjrt_gpu_utils.h"
#include "xla/stream_executor/device_description.h"
#include "xla/tsl/platform/env.h"
#include "xla/tsl/util/command_line_flags.h"
#include "xla/xla_data.pb.h"
#include "tsl/platform/init_main.h"
#include "tsl/platform/path.h"
#include "tsl/platform/protobuf.h"

namespace xla {
namespace gpu {
namespace {

constexpr absl::string_view kUsage = R"(
This tool measures clock cycles per operation on GPU.
)";

void WriteOutput(const DeviceHloInstructionProfiles& literal,
                 absl::string_view name) {
  std::string file_name;
  std::string output_directory;
  if (tsl::io::GetTestUndeclaredOutputsDir(&output_directory)) {
    file_name = tsl::io::JoinPath(
        output_directory,
        absl::StrFormat("profiles-%d-%s", tsl::Env::Default()->NowMicros(),
                        name));
    absl::StrAppend(&file_name, ".textproto");
  } else {
    file_name = tsl::io::GetTempFilename(absl::StrCat(name, ".textproto"));
  }
  VLOG(0) << "Writing output to " << file_name;
  CHECK_OK(tsl::WriteStringToFile(tsl::Env::Default(), file_name,
                                  tsl::LegacyUnredactedDebugString(literal)));
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

int RunProfiler(int argc, char** argv) {
  std::string output_file;
  std::vector<tsl::Flag> flag_list = {
      tsl::Flag("output_file", &output_file,
                "Output measurements protobuf to the destination file."),
  };
  // Allow setting flags as command line argument (in addition to XLA_FLAGS
  // environment variable).
  AppendDebugOptionsFlags(&flag_list);
  bool parse_ok = tsl::Flags::Parse(&argc, argv, flag_list);
  tsl::port::InitMain(kUsage.data(), &argc, &argv);
  if (!parse_ok) {
    LOG(QFATAL) << "Error parsing flags";
  }

  auto [runner, dev_info] = MakeRunnerAndGetDeviceDescription();
  HloOpProfiler profiler(runner.get(), &dev_info);
  VLOG(0) << dev_info.name() << " @ " << dev_info.clock_rate_ghz() << " GHz";

  HloInstructionProfileList instr_profiles;

  for (const PrimitiveType data_type : HloOpProfiler::AllSupportedDtypes()) {
    for (const HloOpcode op : HloOpProfiler::AllSupportedOps()) {
      if (HloOpProfiler::TooFastToMeasure().count(op) ||
          HloOpProfiler::Unsupported().count(op)) {
        continue;
      }
      auto result = profiler.MeasureClockCyclesPerOp(op, data_type);
      if (result.ok()) {
        instr_profiles.add_entries()->Swap(&*result);
      } else {
        LOG(ERROR) << result.status();
      }
    }
  }

  VLOG(1) << "\n" << instr_profiles.DebugString();

  DeviceHloInstructionProfiles device_profiles;
  device_profiles.mutable_entries()->insert(
      {HloOpProfiles::GetProfileName(dev_info), instr_profiles});
  if (!output_file.empty()) {
    WriteOutput(device_profiles, output_file);
  }

  return 0;
}

}  // namespace
}  // namespace gpu
}  // namespace xla

int main(int argc, char** argv) { return xla::gpu::RunProfiler(argc, argv); }
