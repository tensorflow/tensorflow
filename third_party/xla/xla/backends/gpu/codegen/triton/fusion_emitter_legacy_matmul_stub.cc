/* Copyright 2024 The OpenXLA Authors.

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

#include <optional>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "mlir/Interfaces/FunctionInterfaces.h"
#include "xla/backends/gpu/codegen/triton/fusion_emitter_legacy_matmul.h"
#include "xla/codegen/emitter_loc_op_builder.h"
#include "xla/hlo/ir/hlo_instructions.h"
#include "xla/hlo/utils/hlo_traversal.h"
#include "xla/service/gpu/launch_dimensions.h"
#include "xla/service/gpu/matmul_utils.h"
#include "xla/service/gpu/model/tiled_hlo_computation.h"
#include "xla/service/gpu/triton_fusion_analysis.h"
#include "xla/stream_executor/device_description.h"
#include "xla/stream_executor/gpu/tma_metadata.h"

namespace xla::gpu {

// Compute the launch dimensions for the given Triton MatMul.
absl::StatusOr<LaunchDimensions> GetMatMulLaunchDimensions(
    const TritonFusionAnalysis& analysis, const HloFusionAdaptor& fusion,
    const TritonGemmConfig& config, const se::DeviceDescription& device_info) {
  return absl::UnimplementedError("not supported for this build configuration");
}

absl::StatusOr<std::optional<stream_executor::gpu::TmaMetadata>> EmitMatMul(
    EmitterLocOpBuilder& builder, absl::string_view libdevice_path,
    const se::DeviceDescription& device_info,
    const HloFusionInstruction* fusion, mlir::FunctionOpInterface fn,
    const BlockLevelParameters&) {
  return absl::UnimplementedError("not supported for this build configuration");
}

}  // namespace xla::gpu
