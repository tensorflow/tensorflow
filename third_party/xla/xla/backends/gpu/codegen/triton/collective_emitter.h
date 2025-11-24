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

#ifndef XLA_BACKENDS_GPU_CODEGEN_TRITON_COLLECTIVE_EMITTER_H_
#define XLA_BACKENDS_GPU_CODEGEN_TRITON_COLLECTIVE_EMITTER_H_

#include <cstdint>
#include <optional>

#include "absl/container/flat_hash_map.h"
#include "absl/status/statusor.h"
#include "llvm/ADT/SmallVector.h"
#include "mlir/IR/Types.h"
#include "mlir/IR/Value.h"
#include "mlir/Interfaces/FunctionInterfaces.h"
#include "xla/backends/gpu/codegen/triton/emitter_helpers.h"
#include "xla/codegen/emitter_loc_op_builder.h"
#include "xla/codegen/tiling/tiled_hlo_instruction.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_instructions.h"
#include "xla/service/gpu/backend_configs.pb.h"
#include "xla/service/gpu/model/block_level_parameters.h"
#include "xla/stream_executor/device_description.h"
#include "xla/types.h"  // IWYU pragma: keep

namespace xla::gpu {

// Returns the block level fusion config for the collective kernel.
// For now only all-reduce is supported.
// If an std::nullopt is returned, it implies that the collective kernel is
// not supported and cannot be emitted.
absl::StatusOr<std::optional<xla::gpu::BlockLevelFusionConfig>>
GetCollectiveBlockLevelFusionConfig(const se::DeviceDescription& device_info,
                                    const HloFusionInstruction* fusion_instr);

// Sets the BlockLevelFusionConfig for a collective op inside the
// GpuBackendConfig for the fusion instruction.
// Returns true if the collective op is supported and the config is set.
// Returns false if the collective op is not supported. No backend config is set
// in this case.
// Returns an error in case of an internal error or invalid arguments.
absl::StatusOr<bool> TrySetGpuBackendConfigForCollective(
    const se::DeviceDescription& device_info,
    HloFusionInstruction* fusion_instr);

// Adds the metadata arguments to the function's argument list.
// For collective some extra metadata arguments are needed such as rank,
// and pointers to remote GPU buffers.
// The fn_arg_types is updated in place to add these.
// Returns the number of metadata arguments added or error.
absl::StatusOr<int32_t> AddCollectiveMetadataArguments(
    llvm::SmallVector<mlir::Type>& fn_arg_types, EmitterLocOpBuilder& b,
    const HloComputation* hlo_computation);

// Emits tiled XTile/Triton IR for a collective op.
// See [EmitTiledHloInstruction] for an overview of how this fits into the
// emitter.
absl::StatusOr<triton::TensorValue> EmitCollective(
    EmitterLocOpBuilder b, const HloFusionInstruction* fusion,
    const TiledHloInstruction& tiled_hlo_reduce,
    const BlockLevelParameters& block_level_parameters,
    mlir::FunctionOpInterface fn, mlir::Value pid,
    absl::flat_hash_map<const TiledHloInstruction*, triton::TensorValue>&
        values);

}  // namespace xla::gpu
#endif  // XLA_BACKENDS_GPU_CODEGEN_TRITON_COLLECTIVE_EMITTER_H_
