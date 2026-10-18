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
#include <vector>

#include "absl/base/nullability.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "llvm/ADT/SmallVector.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Types.h"
#include "mlir/Support/LLVM.h"
#include "stablehlo/dialect/StablehloOps.h"
#include "xla/backends/gpu/runtime/collective_params.h"
#include "xla/codegen/xtile/ir/xtile_ops.h"
#include "xla/codegen/xtile/xtile_config.pb.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_instructions.h"
#include "xla/service/gpu/backend_configs.pb.h"
#include "xla/service/gpu/launch_dimensions.h"
#include "xla/shape.h"
#include "xla/types.h"  // IWYU pragma: keep

namespace xla {
class DeviceAssignment;
class GpuTopology;
}  // namespace xla

namespace xla::gpu {

// Returns a tile such that each dimension is a power of two and that the total
// number of blocks does not exceed num_blocks. Returns the tile sizes in
// same order as the output shape. It picks tiles greedily from the most major
// dimension to the most minor dimension. Eg:
// - [1024, 1024] with num_blocks = 32 returns [32, 1024]
// - [8192, 16] with num_blocks = 32 returns [256, 16].
// - [5, 1024] with num_blocks = 32 returns [2, 128].
// Note that it can be that the number of blocks used is less than num_blocks.
// @pre num_blocks > 0.
llvm::SmallVector<int64_t> GreedyPowerOfTwoTiles(const Shape& output_shape,
                                                 int32_t num_blocks);

// Returns the block level fusion config for the collective kernel.
// For now only all-reduce is supported.
// If an std::nullopt is returned, it implies that the collective kernel is
// not supported and cannot be emitted.
absl::StatusOr<xla::xtile::BlockLevelFusionConfig>
GetCollectiveBlockLevelFusionConfig(
    const GpuTopology& gpu_topology, const HloFusionInstruction* fusion_instr,
    const DeviceAssignment* device_assignment = nullptr);

// Sets the BlockLevelFusionConfig for a collective op inside the
// GpuBackendConfig for the fusion instruction.
// Returns an error in case the collective configuration cannot be set.
absl::Status TrySetGpuBackendConfigForCollective(
    const GpuTopology& gpu_topology, HloFusionInstruction* fusion_instr,
    const DeviceAssignment* device_assignment = nullptr);

// Adds the metadata arguments to the function's argument list.
// For collective some extra metadata arguments are needed such as rank,
// and pointers to remote GPU buffers.
// The fn_arg_types is updated in place to add these.
// Returns the number of metadata arguments added or error.
absl::StatusOr<int32_t> AddCollectiveMetadataArguments(
    llvm::SmallVector<mlir::Type>& fn_arg_types, mlir::ImplicitLocOpBuilder& b,
    const HloComputation* hlo_computation);

// Version of [AddCollectiveMetadataArguments] that does the same for
// [emitters::KernelArgument] structure.
absl::StatusOr<std::vector<Shape>> GetCollectiveUnmanagedKernelArguments(
    const HloFusionInstruction* fusion);

// Rewrites stablehlo all-reduce op to a triton implementation.
mlir::LogicalResult RewriteAllReduce(mlir::stablehlo::AllReduceOp op,
                                     mlir::PatternRewriter& rewriter);

// Rewrites stablehlo all-gather op to a triton implementation.
mlir::LogicalResult RewriteAllGather(mlir::stablehlo::AllGatherOp op,
                                     mlir::PatternRewriter& rewriter);

// Rewrites stablehlo all-to-all op to a triton implementation.
mlir::LogicalResult RewriteAllToAll(mlir::stablehlo::AllToAllOp op,
                                    mlir::PatternRewriter& rewriter);

// Rewrites xtile reduce-scatter op to a triton implementation.
mlir::LogicalResult RewriteReduceScatter(xla::xtile::ReduceScatterOp op,
                                         mlir::PatternRewriter& rewriter);

// Returns the symmetric memory type for the scratch buffers of the collective
// kernel emitted for `instr` (a collective or a collective fusion), or an error
// if the collective is cross-host and cross-host symmetric memory collectives
// are not supported on `gpu_topology`.
absl::StatusOr<SymmetricMemoryType> GetSymmetricMemoryType(
    const GpuTopology& gpu_topology, const HloInstruction& instr,
    const DeviceAssignment* device_assignment = nullptr);
// Creates a CollectiveKernelSpec for a given collective or fusion instruction.
// `scratch_memory_type` is used for all scratch buffers.
absl::StatusOr<CollectiveKernelSpec> CreateCollectiveKernelSpec(
    const GpuTopology& gpu_topology, const HloInstruction* instr,
    const LaunchDimensions& launch_dimensions,
    SymmetricMemoryType scratch_memory_type,
    const DeviceAssignment* device_assignment = nullptr);

// Returns true if `rs` needs to be reshaped by `FlattenReduceScatterFusion`
// before tiling: either an output is not divisible by the replica group size
// along the scatter dimension, or its per-rank element count is not a power of
// two.
bool ShouldFlattenReduceScatter(
    const HloReduceScatterInstruction* absl_nonnull rs);

// Reshapes a reduce-scatter fusion so every operand is scattered along dim 0
// with output dim 0 equal to R (the number of ranks per group), and wraps the
// fusion inputs and outputs in bitcasts in the parent computation.
// For example, with R = 2, an operand `f32[2,4,8] -> f32[1,4,8]` is reshaped
// to `f32[4,16] -> f32[2,16]`.
absl::Status FlattenReduceScatterFusion(
    HloFusionInstruction* absl_nonnull fusion_instr);

// Returns the collective instruction (all-reduce, all-gather, all-to-all, or
// reduce-scatter) of a collective fusion computation, or nullptr if there is
// none.
const HloInstruction* GetCollectiveHero(const HloComputation* computation);
HloInstruction* GetMutableCollectiveHero(HloComputation* computation);

}  // namespace xla::gpu
#endif  // XLA_BACKENDS_GPU_CODEGEN_TRITON_COLLECTIVE_EMITTER_H_
