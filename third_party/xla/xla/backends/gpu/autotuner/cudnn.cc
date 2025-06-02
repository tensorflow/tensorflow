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

#include "xla/backends/gpu/autotuner/cudnn.h"

#include <memory>
#include <utility>
#include <vector>

#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "xla/autotuning.pb.h"
#include "xla/backends/autotuner/codegen_backend.h"
#include "xla/hlo/ir/hlo_casting_utils.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_instructions.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/hlo/utils/hlo_query.h"
#include "xla/service/algorithm_util.h"
#include "xla/service/compiler.h"
#include "xla/service/gpu/backend_configs.pb.h"
#include "xla/service/gpu/ir_emission_utils.h"
#include "xla/service/gpu/stream_executor_util.h"
#include "xla/service/gpu/transforms/cudnn_fusion_compiler.h"
#include "xla/stream_executor/cuda/cuda_compute_capability.h"
#include "xla/stream_executor/device_description.h"
#include "xla/stream_executor/stream_executor.h"
#include "xla/tsl/protobuf/dnn.pb.h"
#include "xla/xla.pb.h"

namespace xla {
namespace gpu {

namespace se = ::stream_executor;

using CudnnBackendConfig = stream_executor::dnn::AlgorithmProto;

bool IsSupportedByCudnn(const HloInstruction& instr,
                        se::StreamExecutor* stream_executor,
                        const DebugOptions& debug_options) {
  if (instr.opcode() != HloOpcode::kFusion) {
    return false;
  }

  if (!instr.has_backend_config() ||
      !instr.backend_config<GpuBackendConfig>()->has_fusion_backend_config() ||
      instr.backend_config<GpuBackendConfig>()
              ->fusion_backend_config()
              .kind() != kCuDnnFusionKind) {
    return false;
  }

  HloDotInstruction* dot =
      Cast<HloDotInstruction>(hlo_query::GetFirstInstructionWithOpcode(
          *instr.fused_instructions_computation(), HloOpcode::kDot));
  if (dot == nullptr) {
    return false;
  }
  if (dot->sparse_operands()) {
    return false;
  }
  if (!algorithm_util::IsSupportedByCudnn(
          dot->precision_config().algorithm())) {
    return false;
  }

  if (GetDnnVersionInfoOrDefault(stream_executor).major_version() < 9) {
    return false;
  }

  stream_executor::CudaComputeCapability compute_capability =
      stream_executor->GetDeviceDescription().cuda_compute_capability();
  if ((compute_capability.IsAtLeastAmpere() &&
       debug_options.xla_gpu_cudnn_gemm_fusion_level() > 1) ||
      (compute_capability.IsAtLeastBlackwell() &&
       debug_options.xla_gpu_cudnn_gemm_fusion_level() > 0)) {
    return true;
  }

  LOG(ERROR) << "YOLO5";

  return false;
}

absl::StatusOr<std::vector<std::unique_ptr<BackendConfig>>>
CudnnBackend::GetSupportedConfigs(
    const HloInstruction& instr,
    stream_executor::StreamExecutor* stream_executor) {
  if (!IsSupportedByCudnn(instr, stream_executor, debug_options())) {
    return absl::InvalidArgumentError("Cudnn backend is not supported.");
  }
  int plan_count = CuDnnFusionCompiler::GetAvailablePlanCount(
      *stream_executor, *DynCast<HloFusionInstruction>(&instr));
  std::vector<std::unique_ptr<BackendConfig>> configs;
  configs.reserve(plan_count);
  for (int plan_id = 0; plan_id < plan_count; ++plan_id) {
    auto config = std::make_unique<CudnnBackendConfig>();
    config->set_algo_id(plan_id);
    configs.push_back(std::move(config));
  }

  return configs;
}

absl::StatusOr<std::unique_ptr<BackendConfig>> CudnnBackend::GetDefaultConfig(
    const HloInstruction& instr) {
  // Default config would require stream_executor to check if the fusion is
  // supported by Cudnn.
  return absl::InvalidArgumentError(
      "Cudnn backend doesn't support getting a default config.");
}

}  // namespace gpu
}  // namespace xla
