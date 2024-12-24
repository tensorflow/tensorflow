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
#ifndef XLA_SERVICE_GPU_FUSIONS_TRANSPOSE_MLIR_H_
#define XLA_SERVICE_GPU_FUSIONS_TRANSPOSE_MLIR_H_

#include <cstdint>
#include <optional>
#include <vector>

#include "absl/container/inlined_vector.h"
#include "absl/status/status.h"
#include "llvm/ADT/SmallVector.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/AffineExpr.h"
#include "mlir/IR/AffineMap.h"
#include "mlir/IR/ImplicitLocOpBuilder.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Value.h"
#include "mlir/IR/ValueRange.h"
#include "mlir/Support/LLVM.h"
#include "xla/hlo/analysis/indexing_map.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_instructions.h"
#include "xla/service/gpu/fusions/mlir/computation_partitioner.h"
#include "xla/service/gpu/fusions/mlir/mlir_fusion_emitter.h"
#include "xla/service/gpu/hlo_fusion_analysis.h"
#include "xla/service/gpu/ir_emission_utils.h"
#include "xla/service/gpu/launch_dimensions.h"
#include "xla/shape.h"

namespace xla {
namespace gpu {

// Lowers kTranspose fusion to LLVM via MLIR using GPU's shared memory.

// Each thread block of `kWarpSize` x `kNumRows` threads
// transposes one tile: each thread copies kWarpSize/kNumRows elements from
// the input to a shared memory tile.

// This is similar to the following CUDA algorithm in TensorFlow:
// https://goo.gl/MStRV6.
class MlirTransposeFusion : public MlirFusionEmitterBase {
 public:
  explicit MlirTransposeFusion(const HloFusionAnalysis& analysis);
  LaunchDimensions launch_dimensions() const override;

  std::optional<IndexingMap> ComputeThreadIdToOutputIndexing(
      int64_t root_index, mlir::MLIRContext* mlir_context) const override;

  std::optional<IndexingMap> ComputeThreadIdToInputIndexing(
      int64_t root_index, int64_t hero_operand_index,
      mlir::MLIRContext* mlir_context) const override;

 protected:
  absl::Status EmitEntryFunction(
      const mlir_converter::PartitionedComputations& computations,
      const mlir_converter::CallTargetProvider& call_targets,
      mlir::func::FuncOp entry_function,
      const HloFusionInstruction& fusion) const override;

  std::vector<mlir_converter::EpilogueSpecification> GetEpilogues(
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
      const mlir_converter::PartitionedComputation& root_computation,
      const mlir_converter::CallTargetProvider& call_target_provider,
      mlir::ValueRange output_args,
      mlir::ValueRange thread_and_block_ids) const;
  void EmitReadFromShMemMlir(
      mlir::ImplicitLocOpBuilder& builder, mlir::func::FuncOp entry_function,
      const HloFusionInstruction& fusion,
      const mlir_converter::PartitionedComputations& computations,
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

}  // namespace gpu
}  // namespace xla

#endif  // XLA_SERVICE_GPU_FUSIONS_TRANSPOSE_MLIR_H_
