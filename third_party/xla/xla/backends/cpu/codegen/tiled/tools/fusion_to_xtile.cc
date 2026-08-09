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

#include <cstdint>
#include <iostream>
#include <memory>
#include <string>
#include <vector>

#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/strings/string_view.h"
#include "llvm/Support/raw_ostream.h"
#include "xla/backends/cpu/codegen/fusion_compiler.h"
#include "xla/backends/cpu/codegen/tiled/tiled_fusion_emitter.h"
#include "xla/debug_options_flags.h"
#include "xla/hlo/ir/hlo_casting_utils.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_instructions.h"
#include "xla/service/gpu/backend_configs.pb.h"
#include "xla/service/gpu/model/block_level_parameters.h"
#include "xla/tools/hlo_module_loader.h"
#include "xla/tsl/platform/logging.h"
#include "xla/tsl/util/command_line_flags.h"
#include "xla/util.h"
#include "xla/xla.pb.h"
#include "xla/xla_data.pb.h"
#include "tsl/platform/init_main.h"

namespace xla::gpu {
namespace {

absl::Status RealMain(absl::string_view input_file) {
  ABSL_ASSIGN_OR_RETURN(std::unique_ptr<HloModule> hlo_module,
                   xla::LoadModuleFromFile(std::string(input_file)));

  const HloInstruction& fusion =
      *hlo_module->entry_computation()->root_instruction();
  ABSL_ASSIGN_OR_RETURN(auto gpu_config, fusion.backend_config<GpuBackendConfig>());
  const HloFusionInstruction* fusion_instr =
      Cast<HloFusionInstruction>(&fusion);
  const FusionBackendConfig& backend_config =
      gpu_config.fusion_backend_config();
  if (!backend_config.has_block_level_fusion_config()) {
    return absl::InvalidArgumentError(
        "Fusion backend config must have block_level_fusion_config.");
  }
  BlockLevelParameters block_level_parameters =
      BlockLevelParameters::FromBlockLevelFusionConfig(
          backend_config.block_level_fusion_config());

  auto mlir_context = cpu::FusionCompiler::CreateContext();

  VLOG(1) << "fusion instruction: " << fusion.ToString() << "\n";

  cpu::TiledEmissionResult result = cpu::EmitTiledFusionKernel(
      *mlir_context, *fusion_instr, /*buffer_assignment=*/nullptr,
      "wrapped_fusion", /*num_work_groups=*/block_level_parameters.num_ctas,
      block_level_parameters);
  if (!result.kernel.ok()) {
    return result.kernel.status();
  }
  result.kernel->source().module().print(llvm::outs());
  return absl::OkStatus();
}

}  // namespace
}  // namespace xla::gpu

int main(int argc, char** argv) {
  std::vector<tsl::Flag> flag_list;
  xla::AppendDebugOptionsFlags(&flag_list);
  const std::string kUsageString = tsl::Flags::Usage(argv[0], flag_list);
  bool parse_ok = tsl::Flags::Parse(&argc, argv, flag_list);
  tsl::port::InitMain(argv[0], &argc, &argv);
  if (!parse_ok) {
    // Print the usage using cerr to avoid truncation by LOG.
    std::cerr << kUsageString;
    return 1;
  }
  CHECK_GT(argc, 1) << "Must specify an input file";
  absl::Status status = xla::gpu::RealMain(argv[1]);
  if (!status.ok()) {
    // We don't return non-zero codes as some of the tests check the status.
    std::cerr << status << "\n";
  }
  return 0;
}
