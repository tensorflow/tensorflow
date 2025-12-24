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

#include <cstdint>
#include <vector>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "mlir/IR/MLIRContext.h"
#include "xla/backends/cpu/codegen/tiled/tiled_fusion_emitter.h"
#include "xla/codegen/kernel_definition.h"
#include "xla/codegen/mlir_kernel_source.h"
#include "xla/codegen/tiling/tiling_specification.h"
#include "xla/hlo/ir/hlo_instructions.h"
#include "xla/service/buffer_assignment.h"

namespace xla::cpu {

bool IsSupportedTilingType(PrimitiveType type) { return false; }

absl::StatusOr<std::vector<FlatTiling>> GetTilingIfSupported(
    mlir::MLIRContext& context, const HloFusionInstruction& fusion) {
  return absl::UnimplementedError("not supported for this build configuration");
}

absl::StatusOr<KernelDefinition<MlirKernelSource>> EmitTiledFusionKernel(
    mlir::MLIRContext& context, const HloFusionInstruction& fusion,
    const BufferAssignment* buffer_assignment, absl::string_view name,
    int64_t num_work_groups, absl::Span<const FlatTiling> tiling) {
  return absl::UnimplementedError("not supported for this build configuration");
}

}  // namespace xla::cpu
