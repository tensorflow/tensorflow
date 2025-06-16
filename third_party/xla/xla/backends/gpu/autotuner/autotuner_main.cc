/* Copyright 2025 The OpenXLA Authors.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http:www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include <iostream>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/ascii.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "xla/backends/autotuner/autotuner.h"
#include "xla/backends/autotuner/profiler.h"
#include "xla/backends/gpu/autotuner/factory.h"
#include "xla/backends/gpu/autotuner/gpu_profiler.h"
#include "xla/debug_options_flags.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/hlo/parser/hlo_parser.h"
#include "xla/service/compiler.h"
#include "xla/service/platform_util.h"
#include "xla/stream_executor/cuda/cuda_platform_id.h"
#include "xla/stream_executor/platform.h"
#include "xla/stream_executor/platform_manager.h"
#include "xla/tsl/platform/env.h"
#include "xla/tsl/platform/errors.h"
#include "xla/tsl/platform/statusor.h"
#include "xla/tsl/util/command_line_flags.h"
#include "tsl/platform/init_main.h"

namespace {

const char* const kUsage = R"(
This tool autotunes an HLO module from a given HLO file and prints the autotuned module to stdout.

Usage:

  bazel run autotuner_main -- --hlo_file=path/to/hlo_module
)";

}  // namespace

namespace xla {

absl::StatusOr<std::unique_ptr<HloModule>> GetModule(
    const std::string& hlo_file) {
  std::string hlo_text;
  TF_RETURN_IF_ERROR(
      tsl::ReadFileToString(tsl::Env::Default(), hlo_file, &hlo_text));
  return ParseAndReturnUnverifiedModule(hlo_text);
}

absl::Status Autotune(HloModule& module) {
  TF_ASSIGN_OR_RETURN(std::string platform_name,
                      PlatformUtil::CanonicalPlatformName("gpu"));

  TF_ASSIGN_OR_RETURN(se::Platform * platform,
                      se::PlatformManager::PlatformWithName(
                          absl::AsciiStrToUpper(platform_name)));
  // TODO: b/407494793 - Add support for ROCM.
  if (platform->id() != se::cuda::kCudaPlatformId) {
    return absl::InternalError("Only CUDA is supported");
  }
  if (platform->VisibleDeviceCount() == 0) {
    return absl::InternalError("No devices found");
  }

  TF_ASSIGN_OR_RETURN(std::unique_ptr<Compiler> compiler,
                      xla::Compiler::GetForPlatform(platform));
  se::StreamExecutor* stream_executor = platform->ExecutorForDevice(0).value();
  Compiler::TargetConfig target_config(stream_executor);
  DebugOptions debug_options = GetDebugOptionsFromFlags();
  auto backends = gpu::GetAllGpuCodegenBackends(&target_config, &debug_options,
                                                compiler.get());

  auto profiler = gpu::GpuProfiler::Create(stream_executor, ProfileOptions());
  if (profiler == nullptr) {
    return absl::InternalError("Failed to create profiler");
  }

  TF_ASSIGN_OR_RETURN(std::unique_ptr<Autotuner> autotuner,
                      Autotuner::Create(std::move(backends), stream_executor,
                                        std::move(profiler), AutotuneConfig()));

  // TODO: b/407494793 - Expand the filter to include more instructions.
  auto should_autotune = [](const HloInstruction& instruction) -> bool {
    if ((instruction.opcode() == HloOpcode::kFusion &&
         instruction.fusion_kind() == HloInstruction::FusionKind::kCustom) ||
        instruction.opcode() == HloOpcode::kCustomCall) {
      return true;
    }
    return false;
  };

  return autotuner->Autotune(&module, should_autotune);
}

}  // namespace xla

int main(int argc, char* argv[]) {
  std::string hlo_file;
  std::vector<tsl::Flag> flag_list = {
      tsl::Flag("hlo_file", &hlo_file, "Path to the HLO file to autotune.")};

  const std::string usage_string =
      absl::StrCat(kUsage, "\n\n", tsl::Flags::Usage(argv[0], flag_list));
  bool parse_ok = tsl::Flags::Parse(&argc, argv, flag_list);
  if (!parse_ok) {
    LOG(QFATAL) << usage_string;
  }
  tsl::port::InitMain(usage_string.c_str(), &argc, &argv);
  auto module = xla::GetModule(hlo_file);
  CHECK_OK(module.status());
  CHECK_OK(xla::Autotune(*module.value()));
  std::cout << module.value()->ToString() << std::endl;
  return 0;
}
