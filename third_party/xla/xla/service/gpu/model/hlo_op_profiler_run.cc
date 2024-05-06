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

#include <string>
#include <vector>

#include "absl/log/log.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"
#include "absl/strings/string_view.h"
#include "xla/debug_options_flags.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/service/gpu/model/hlo_op_profile.pb.h"
#include "xla/service/gpu/model/hlo_op_profiler.h"
#include "xla/service/gpu/model/hlo_op_profiles.h"
#include "xla/service/hlo_runner.h"
#include "xla/service/platform_util.h"
#include "xla/stream_executor/device_description.h"
#include "xla/tsl/util/command_line_flags.h"
#include "xla/xla_data.pb.h"
#include "tsl/platform/env.h"
#include "tsl/platform/init_main.h"
#include "tsl/platform/path.h"
#include "tsl/platform/protobuf.h"
#include "tsl/platform/status.h"

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
    std::string filename = tsl::io::JoinPath(
        output_directory,
        absl::StrFormat("profiles-%d-%s", tsl::Env::Default()->NowMicros(),
                        name));
    file_name = absl::StrCat(filename, ".textproto");
  } else {
    file_name = tsl::io::GetTempFilename(absl::StrCat(name, ".textproto"));
  }
  VLOG(0) << "Writing output to " << file_name;
  TF_CHECK_OK(
      tsl::WriteStringToFile(tsl::Env::Default(), file_name,
                             tsl::LegacyUnredactedDebugString(literal)));
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

  HloRunner runner(PlatformUtil::GetPlatform("cuda").value());
  HloOpProfiler profiler(runner);
  const se::DeviceDescription& dev_info =
      runner.backend().stream_executors()[0]->GetDeviceDescription();
  VLOG(0) << dev_info.name() << " @ " << dev_info.clock_rate_ghz() << " GHz";

  const std::vector<PrimitiveType> dtypes = {
      S8, S16, S32, S64, U8, U16, U32, U64, F16, F32, F64, C64, C128,
  };
  const std::vector<HloOpcode> ops = {
      // Unary
      HloOpcode::kCbrt,
      HloOpcode::kCos,
      HloOpcode::kErf,
      HloOpcode::kExp,
      HloOpcode::kExpm1,
      HloOpcode::kLog,
      HloOpcode::kLog1p,
      HloOpcode::kLogistic,
      HloOpcode::kRsqrt,
      HloOpcode::kSin,
      HloOpcode::kSqrt,
      HloOpcode::kTanh,
      // Binary
      HloOpcode::kAdd,
      HloOpcode::kAtan2,
      HloOpcode::kDivide,
      HloOpcode::kMultiply,
      HloOpcode::kPower,
      HloOpcode::kSubtract,
  };

  HloInstructionProfileList instr_profiles;

  for (const PrimitiveType data_type : dtypes) {
    for (const HloOpcode op : ops) {
      auto result = profiler.MeasureClockCyclesPerOp(op, data_type);
      if (result.ok()) {
        instr_profiles.add_entries()->Swap(&*result);
      } else {
        LOG(ERROR) << result.status();
      }
    }
  }

  VLOG(1) << "\n" << instr_profiles;

  auto profile_name = HloOpProfiles::GetProfileName(&dev_info);
  DeviceHloInstructionProfiles device_profiles;
  device_profiles.mutable_entries()->insert({profile_name, instr_profiles});
  if (!output_file.empty()) {
    WriteOutput(device_profiles, output_file);
  }

  return 0;
}

}  // namespace
}  // namespace gpu
}  // namespace xla

int main(int argc, char** argv) { return xla::gpu::RunProfiler(argc, argv); }
