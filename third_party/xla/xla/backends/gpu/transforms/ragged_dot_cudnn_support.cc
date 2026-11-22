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

#include "xla/backends/gpu/transforms/ragged_dot_cudnn_support.h"

#include <memory>

#include "absl/base/nullability.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "xla/backends/gpu/transforms/cudnn_fusion_compiler.h"
#include "xla/backends/gpu/transforms/ragged_dot_fusion_rewriter.h"
#include "xla/hlo/ir/hlo_casting_utils.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_instructions.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/layout_util.h"
#include "xla/service/decision.h"
#include "xla/service/gpu/cudnn_support_utils.h"
#include "xla/stream_executor/device_description.h"
#include "xla/stream_executor/dnn.h"
#include "xla/stream_executor/stream_executor.h"
#include "xla/tools/hlo_decomposer.h"
#include "xla/xla.pb.h"

namespace xla {
namespace gpu {

Decision IsRaggedDotSupportedByCuDnn(
    const HloInstruction& ragged_dot,
    const stream_executor::DeviceDescription& device_description,
    stream_executor::StreamExecutor* absl_nullable stream_exec) {
  const DebugOptions& debug_options =
      ragged_dot.GetModule()->config().debug_options();

  // Resolve the cuDNN handle the same way CuDnnFusionCompiler will: this honors
  // xla_gpu_cudnn_deviceless_compilation_mode (e.g. forced deviceless) and
  // returns nullptr for deviceless compilation.
  absl::StatusOr<stream_executor::dnn::DnnSupport* absl_nullable> dnn_support =
      ResolveCudnnSupport(debug_options, stream_exec,
                          device_description.dnn_version());
  if (!dnn_support.ok()) {
    VLOG(1) << "Ragged dot cuDNN probe: cannot resolve cuDNN support: "
            << dnn_support.status();
    return Decision::Forbid("cannot resolve cuDNN support: ")
           << dnn_support.status().message();
  }

  // ExtractInstructionIntoNewModule() uses a default module config, so carry
  // over the debug options: RaggedDotFusionRewriter and CuDnnFusionCompiler
  // both read them.
  std::unique_ptr<HloModule> probe_module =
      ExtractInstructionIntoNewModule(ragged_dot);
  probe_module->mutable_config().set_debug_options(debug_options);

  // This runs before layout assignment, so `ragged_dot` may still carry
  // arbitrary layouts. GpuLayoutAssignment pins the operands and the result of
  // a ragged dot (and of a cuDNN fusion) to the default layout, so normalize
  // the probe the same way to match the fusion cuDNN will actually see.
  HloComputation* probe_entry = probe_module->entry_computation();
  for (HloInstruction* instr : probe_entry->instructions()) {
    LayoutUtil::SetToDefaultLayout(instr->mutable_shape());
  }
  probe_module->mutable_config().SetComputationLayoutIfExists(
      probe_entry->ComputeProgramShape());

  // Build the fusion that the real pipeline would hand to cuDNN, by running
  // RaggedDotFusionRewriter on the scratch module. This is not recursive: the
  // rewriter does not consult this support check (that happens earlier, in
  // RaggedDotRewriter, via the callback), it unconditionally converts every
  // ragged dot it sees. We reuse the rewriter rather than hand-building a
  // fusion because its output depends on its rewrites (wgrad K/N padding for
  // TMA alignment, padding-tail masking, cumulative-sum group offsets), and
  // cuDNN support must be probed on that form. The scratch module keeps those
  // in-place rewrites away from the module being compiled.
  absl::StatusOr<bool> changed =
      RaggedDotFusionRewriter().Run(probe_module.get());
  if (!changed.ok()) {
    VLOG(1) << "Ragged dot cuDNN probe: fusion rewrite failed: "
            << changed.status();
    return Decision::Forbid("fusion rewrite failed: ")
           << changed.status().message();
  }
  for (const HloInstruction* instr : probe_entry->instructions()) {
    if (instr->opcode() != HloOpcode::kFusion) {
      continue;
    }
    // A null DnnSupport means deviceless: cuDNN answers from the device
    // description alone.
    absl::StatusOr<int> plan_count = CuDnnFusionCompiler::GetAvailablePlanCount(
        *dnn_support, device_description, *Cast<HloFusionInstruction>(instr));
    if (!plan_count.ok()) {
      VLOG(1) << "Ragged dot cuDNN probe: plan count failed: "
              << plan_count.status();
      return Decision::Forbid("plan count failed: ")
             << plan_count.status().message();
    }
    if (*plan_count <= 0) {
      return Decision::Forbid("cuDNN reports no available plans");
    }
    return Decision::Allow();
  }
  return Decision::Forbid("ragged dot did not produce a cuDNN fusion");
}

}  // namespace gpu
}  // namespace xla
