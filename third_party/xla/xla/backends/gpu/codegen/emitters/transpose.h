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
#ifndef XLA_BACKENDS_GPU_CODEGEN_EMITTERS_TRANSPOSE_H_
#define XLA_BACKENDS_GPU_CODEGEN_EMITTERS_TRANSPOSE_H_

#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <vector>

#include "absl/container/inlined_vector.h"
#include "absl/status/status.h"
#include "absl/types/span.h"
#include "llvm/ADT/SmallVector.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/AffineExpr.h"
#include "mlir/IR/AffineMap.h"
#include "mlir/IR/ImplicitLocOpBuilder.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Value.h"
#include "mlir/IR/ValueRange.h"
#include "mlir/Support/LLVM.h"
#include "xla/backends/gpu/codegen/emitters/emitter_base.h"
#include "xla/codegen/emitters/computation_partitioner.h"
#include "xla/hlo/analysis/indexing_map.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_instructions.h"
#include "xla/service/gpu/hlo_fusion_analysis.h"
#include "xla/service/gpu/ir_emission_utils.h"
#include "xla/service/gpu/launch_dimensions.h"
#include "xla/shape.h"
#include "xla/xla_data.pb.h"

namespace xla {
namespace gpu {

// Lowers kTranspose fusion to LLVM via MLIR using GPU's shared memory.

// Each thread block of `kWarpSize` x `kNumRows` threads
// transposes one tile: each thread copies kWarpSize/kNumRows elements from
// the input to a shared memory tile.

// This is similar to the following CUDA algorithm in TensorFlow:
// https://goo.gl/MStRV6.
class TransposeFusion : public EmitterBase {
 public:
  explicit TransposeFusion(const HloFusionAnalysis& analysis);
  LaunchDimensions launch_dimensions() const override;

  std::optional<IndexingMap> ComputeThreadIdToOutputIndexing(
      int64_t root_index, mlir::MLIRContext* mlir_context) const override;

  std::optional<IndexingMap> ComputeThreadIdToInputIndexing(
      int64_t root_index, int64_t hero_operand_index,
      mlir::MLIRContext* mlir_context) const override;

 protected:
  absl::Status EmitEntryFunction(
      const emitters::PartitionedComputations& computations,
      const emitters::CallTargetProvider& call_targets,
      mlir::func::FuncOp entry_function,
      const HloFusionInstruction& fusion) const override;

  std::vector<emitters::EpilogueSpecification> GetEpilogues(
      const HloFusionInstruction& fusion,
      mlir::MLIRContext* mlir_context) const override;

  struct WriteResult {
    // All output tensors of the fusion, with side outputs written to them.
    mlir::SmallVector<mlir::Value> updated_outputs;
    // Shared memory tiles for transpose heroes.
    mlir::ValueRange shmem_tensors;
  };

  WriteResult EmitWriteToShMemMlir(
      mlir::ImplicitLocOpBuilder& builder, mlir::func::FuncOp entry_function,
      const HloFusionInstruction& fusion,
      const emitters::PartitionedComputation& root_computation,
      const emitters::CallTargetProvider& call_target_provider,
      mlir::ValueRange output_args,
      mlir::ValueRange thread_and_block_ids) const;
  void EmitReadFromShMemMlir(
      mlir::ImplicitLocOpBuilder& builder, mlir::func::FuncOp entry_function,
      const HloFusionInstruction& fusion,
      const emitters::PartitionedComputations& computations,
      const WriteResult& written, mlir::ValueRange thread_and_block_ids) const;

 private:
  const HloFusionAnalysis& analysis_;

  IndexingMap GetIndexing(bool input, const xla::Shape& shape,
                          mlir::MLIRContext* ctx) const;
  IndexingMap GetSharedMemoryIndexing(bool read, mlir::MLIRContext* ctx) const;
  llvm::SmallVector<mlir::AffineExpr, 4> GetThreadOffsets(
      bool read, mlir::MLIRContext* ctx) const;
  bool MostMinorDimensionUnchanged() const;

  TransposeDescription transpose_;
  absl::InlinedVector<int64_t, 3> permutation_;
  std::vector<int64_t> input_shape_;
  std::vector<int64_t> block_sizes_;  // In input elements.
  std::vector<int64_t> output_block_sizes_;
  std::vector<int64_t> block_counts_;
  int vector_size_;
  int block_size_;
  int64_t base_block_size_;

  std::vector<const HloInstruction*> shmem_transposes_;
  std::vector<const HloInstruction*> shmem_transpose_roots_;
  std::vector<int> shmem_transpose_root_indices_;
  std::vector<const HloInstruction*> side_output_roots_;
  std::vector<int> side_output_root_indices_;
};

// Packed transpose is a more advanced version of the transpose emitter.
// It considers the canonical transpose described by TransposeSpec class,
// i.e. [T2, A, T1, B] -> [T1, A, T2, B] and tries to pack as many T1 rows into
// shared memory as possible.
//
// Let's describe the algorithm for a concrete example.
//   bf16 [640,100,6,1] - > bf16 [6,100,640,1]
//
// 1. Compute the vector size based on the bitwidth of the element type and the
//    width of the shared memory bank (32 bits):
//
//    vector_size = 32 bits / 16 bits = 2
//
// 2. Decide the shared memory size based on the vector size and the number of
//    banks.
//
//    shmem_size = 32 * vector_size = 64
//
// 3. Allocate shared memory
//
//    %shmem = xla.allocate_shared : tensor<64x64xbf16>
//
// 4. Compute the tile sizes to pack as many T1 rows as possible into the
//    columns of the shared memory tensor.
//
//    tile_size_t1 = min(t1, shmem_size) = min(6, 64) = 6
//    tile_size_a = shmem_size / tile_size_t1 = 64 / 6 = 10
//    tile_size_t2 = min(t2, shmem_size) = min(64, 64) = 64
//
//    populated_shmem_cols = tile_size_a * tile_size_t1 = 60
//    populated_shmem_rows = tile_size_t2 = 64
//
//    In this case we are packing 64 x 10 x 6 x bf16 tile into 64 x 60 x bf16
//    slice of shared memory.
//
// 5. Every GPU block gets a single 64 x 10 x 6 x bf16 tile.
//    The tile is read by `num_warps_per_block` warps.
//    Let's assume that there are 4 warps per block. In this case, on every
//    iteration each warp will read 10 x 6 x bf16 elements, i.e. every thread
//    (30 out of 32) performs a vector load of 2 x bf16 and stores it to the
//    shared memory. In total, there will be 16 iterations performed by each
//    block.
//
//    The following code snippet shows how the data is read from the input
//    tensor into the shared memory:
//
//    for I = 0 to CEIL(shmem_rows, num_warps_per_block):
//      for J = 0 to VECTOR_SIZE:
//        ROW = WARP_ID + NUM_WARPS * I
//        COL = LANE_ID * VECTOR_SIZE + J
//        SHMEM[ROW, COL] = INPUT[ROW, COL / 10, COL % 10]
//
//    After the data is read, xla_gpy.sync_threads will be inserted.
//
// 6. Each thread reads a VECTOR_SIZE x VECTOR_SIZE x bf16 tile from the shared
//    memory and performs the write of each of the columns of the tile.
//
//    for I = 0 to CEIL(shmem_cols, VECTOR_SIZE * num_warps_per_block):
//      VECTOR_2D = arith.constant dense<0>
//        : vector<VECTOR_SIZE x VECTOR_SIZE x bf16>
//      for J = 0 to VECTOR_SIZE:
//        VECTOR_2D[J, :] = SHMEM[LANE_ID * VECTOR_SIZE + J, I:I+2]
//      for J = 0 to VECTOR_SIZE:
//        for K = 0 to VECTOR_SIZE:
//          OUTPUT[(I + J) % 10, (I + J) / 10,
//                  LANE_ID * VECTOR_SIZE + K] =  VECTOR_2D[K, J]
class PackedTranspose : public EmitterBase {
 public:
  explicit PackedTranspose(const HloFusionAnalysis& analysis,
                           const TransposeSpec& spec,
                           absl::Span<const int64_t> output_block_tile,
                           int64_t num_warps);

  LaunchDimensions launch_dimensions() const override;

  std::optional<IndexingMap> ComputeThreadIdToOutputIndexing(
      int64_t root_index, mlir::MLIRContext* mlir_context) const override;

  std::optional<IndexingMap> ComputeThreadIdToInputIndexing(
      int64_t root_index, int64_t hero_operand_index,
      mlir::MLIRContext* mlir_context) const override;

 protected:
  absl::Status EmitEntryFunction(
      const emitters::PartitionedComputations& computations,
      const emitters::CallTargetProvider& call_targets,
      mlir::func::FuncOp entry_function,
      const HloFusionInstruction& fusion) const override;

  std::vector<emitters::EpilogueSpecification> GetEpilogues(
      const HloFusionInstruction& fusion,
      mlir::MLIRContext* mlir_context) const override;

  struct WriteResult {
    // All output tensors of the fusion, with side outputs written to them.
    mlir::SmallVector<mlir::Value> updated_outputs;
    // Shared memory tiles for transpose heroes.
    mlir::ValueRange shmem_tensors;
  };

  WriteResult EmitWriteToShMemMlir(
      mlir::ImplicitLocOpBuilder& builder, mlir::func::FuncOp entry_function,
      const HloFusionInstruction& fusion,
      const emitters::PartitionedComputation& root_computation,
      const emitters::CallTargetProvider& call_target_provider,
      mlir::ValueRange output_args,
      mlir::ValueRange thread_and_block_ids) const;

  void EmitReadFromShMemMlir(
      mlir::ImplicitLocOpBuilder& builder, mlir::func::FuncOp entry_function,
      const HloFusionInstruction& fusion,
      const emitters::PartitionedComputations& computations,
      const WriteResult& written, mlir::ValueRange thread_and_block_ids) const;

 private:
  IndexingMap GetInputIndexing(mlir::MLIRContext* ctx) const;
  IndexingMap GetShmemWriteIndexing(mlir::MLIRContext* ctx) const;

  IndexingMap GetShmemReadIndexing(mlir::MLIRContext* ctx) const;
  IndexingMap GetOutputIndexing(mlir::MLIRContext* ctx) const;

  const HloFusionAnalysis& analysis_;
  TransposeSpec spec_;

  // Tile sizes for the canonical input shape.
  std::vector<int64_t> output_tile_;

  // Tile sizes for the canonical output shape.
  std::vector<int64_t> input_tile_;

  // Block counts for the canonical output shape.
  std::vector<int64_t> block_counts_;

  // Vector size in elements.
  int64_t vector_size_;

  // Number of warps per block.
  int64_t num_warps_per_block_;

  // Tile sizes for the canonicalical dimensions
  // [T2, A, T1, 1] -> [T1, A, T2, 1].
  int64_t tile_size_t1_;
  int64_t tile_size_a_;
  int64_t tile_size_t2_;

  // Number of populated columns in the shared memory tensor.
  int64_t populated_shmem_cols_;

  // Number of populated rows in the shared memory tensor.
  int64_t populated_shmem_rows_;

  // Transpose instructions that require shared memory. Note that not all
  // transposes require shared memory, e.g. the ones with a large innermost
  // dimension.
  std::vector<const HloInstruction*> shmem_transposes_;

  // Roots that have shmem transposes as heroes.
  std::vector<const HloInstruction*> shmem_transpose_roots_;

  // Root indices for shmem_transpose_roots_.
  std::vector<int> shmem_transpose_root_indices_;

  // Roots that don't have a transpose hero.
  std::vector<const HloInstruction*> side_output_roots_;

  // Root indices for side_output_roots_.
  std::vector<int> side_output_root_indices_;
};

std::unique_ptr<EmitterBase> CreateTransposeFusion(
    const HloFusionAnalysis& analysis);

}  // namespace gpu
}  // namespace xla

#endif  // XLA_BACKENDS_GPU_CODEGEN_EMITTERS_TRANSPOSE_H_
