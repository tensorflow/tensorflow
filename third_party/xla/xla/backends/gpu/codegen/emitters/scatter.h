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
#ifndef XLA_BACKENDS_GPU_CODEGEN_EMITTERS_SCATTER_H_
#define XLA_BACKENDS_GPU_CODEGEN_EMITTERS_SCATTER_H_

#include <cstdint>
#include <memory>
#include <optional>
#include <vector>

#include "absl/container/flat_hash_set.h"
#include "absl/status/status.h"
#include "llvm/ADT/SmallVector.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/ImplicitLocOpBuilder.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Value.h"
#include "mlir/IR/ValueRange.h"
#include "xla/backends/gpu/codegen/emitters/emitter_base.h"
#include "xla/codegen/emitters/computation_partitioner.h"
#include "xla/hlo/analysis/indexing_map.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_instructions.h"
#include "xla/service/gpu/hlo_fusion_analysis.h"
#include "xla/service/gpu/launch_dimensions.h"
#include "xla/shape.h"
#include "xla/stream_executor/device_description.h"
#include "xla/util.h"
#include "xla/xla_data.pb.h"

namespace xla {
namespace gpu {

class EmitterHelper;

// Full description of the scatter operation.
// The shape of the indices tensor is <num_slices x index_vector_length>.
// The shape of the updates tensor is <num_slices x slice_shape>.
struct ScatterDescription {
  const HloScatterInstruction* scatter;
  int64_t num_slices;
  int64_t index_vector_length;
  PrimitiveType elem_type;
  // The shape of the updates tensor
  Shape update_shape;
  llvm::SmallVector<int64_t, 2> slice_shape;
  llvm::SmallVector<int64_t, 2> output_shape;
};
ScatterDescription GetScatterDescription(const HloFusionAnalysis& analysis);

class ScatterFusion : public EmitterBase {
 public:
  explicit ScatterFusion(const HloFusionAnalysis& analysis,
                         const ScatterDescription& description,
                         int64_t vector_size);

  absl::Status EmitEntryFunction(
      const emitters::PartitionedComputations& computations,
      const emitters::CallTargetProvider& call_targets,
      mlir::func::FuncOp entry_function,
      const HloFusionInstruction& fusion) const override;

  LaunchDimensions launch_dimensions() const override {
    return LaunchDimensions(num_blocks_, num_warps_ * warp_size_);
  }

  std::optional<IndexingMap> ComputeThreadIdToOutputIndexing(
      int64_t root_index, mlir::MLIRContext* ctx) const override {
    // Since the access pattern to the output is not statically known, we cannot
    // compute the output->input indexing map.
    return std::nullopt;
  }

  std::optional<IndexingMap> ComputeThreadIdToInputIndexing(
      int64_t root_index, int64_t hero_operand_index,
      mlir::MLIRContext* ctx) const override;

 protected:
  virtual absl::Status EmitEntryFunctionImpl(
      mlir::ImplicitLocOpBuilder& b, const EmitterHelper& helper,
      const IndexingMap& updates_map, const IndexingMap& indices_map,
      mlir::ValueRange thread_and_block_ids,
      mlir::Value output_tensor) const = 0;

  virtual void ComputeIndexing(mlir::MLIRContext* ctx, IndexingMap* updates_map,
                               IndexingMap* indices_map) const = 0;

  std::vector<emitters::EpilogueSpecification> GetEpilogues(
      const HloFusionInstruction& fusion,
      mlir::MLIRContext* mlir_context) const final;

  const HloFusionAnalysis& analysis_;
  ScatterDescription description_;

  // The grid is {num_warps_ * WarpSize(), 1, 1, num_blocks_, 1, 1}.
  int64_t warp_size_;
  int64_t num_warps_;
  int64_t num_blocks_;

  // The number of elements that every thread will read from the updates tensor
  // and write to the output tensor.
  int64_t vector_size_;
};

// The distribution happens similarly to the loop emitter, but the iteration
// space corresponds to the shape of the updates tensor. In this case, GPU
// performs a grid-stride loop over the updates and every warp computes at what
// index to scatter an element(s) of the update.
class ScatterWithDistributedUpdates : public ScatterFusion {
 public:
  explicit ScatterWithDistributedUpdates(const HloFusionAnalysis& analysis,
                                         const ScatterDescription& description,
                                         int64_t vector_size);

 protected:
  absl::Status EmitEntryFunctionImpl(mlir::ImplicitLocOpBuilder& b,
                                     const EmitterHelper& helper,
                                     const IndexingMap& updates_map,
                                     const IndexingMap& indices_map,
                                     mlir::ValueRange thread_and_block_ids,
                                     mlir::Value output_tensor) const override;

  void ComputeIndexing(mlir::MLIRContext* ctx, IndexingMap* updates_map,
                       IndexingMap* indices_map) const override;
};

// Every warp will process one or more indices, i.e. there won't be two threads
// in a warp that scatter different indices at a time. In this case, every warp
// iterates its fraction of the indices, and then computes what updates to
// scatter.
// It implements the following algorithm:

/*
 %indices = -1
 %inbounds = false
 %acc = vector<num_iter x vector_size>

 // #indices_map
 %updated_accumulator, %updated_out
    = for %i = 0 to %num_indices_per_warp_  step %indices_vector_size_ {
        for %j = 0 to %indices_vector_size_  step 1 {
  %index = %i * %indices_vector_size_ + %j + %index_start(bl_x, th_x)
  %new_indices = PadWithZeros(ExtractOffsets(%indices_operand, %index))
  %indices_changed = EmitInequalityCheck(%new_indices, %indices)
  if (%indices_changed && %index != 0) {
    %output_tensor = WriteAccumulatorToOutput(%current_acc, %current_out);
  }
  if (%indices_changed) {
    %inbounds = EmitBoundsCheck(%new_indices, %slice_shape, %output_shape)
  }
  if (%inbounds) {
    if (%indices_changed) {
     // updates_map(%index)
     for %j = 0 to %num_slice_iterations_per_warp step 1 {
       for %k = 0 to %vector_size step 1 {
         %update_elem = GetUpdateElement
         %acc = %update_elem
       }
     }
    } else {
     // updates_map(%index)
     for %j = 0 to %num_slice_iterations_per_warp step 1 {
       for %k = 0 to %vector_size step 1 {
         %update_elem = GetUpdateElement
         %acc = Reduce(%update_elem, %acc)
       }
     }
    }
  }
}
%final_out = WriteAccumulatorToOutput(%updated_accumulator, %updated_out);
*/
class ScatterWithDistributedIndices : public ScatterFusion {
 public:
  explicit ScatterWithDistributedIndices(const HloFusionAnalysis& analysis,
                                         const ScatterDescription& description,
                                         int64_t vector_size,
                                         int64_t num_warps_per_slice,
                                         int64_t num_indices_per_warp,
                                         int64_t indices_vector_size);

 protected:
  void ComputeIndexing(mlir::MLIRContext* ctx, IndexingMap* updates_map,
                       IndexingMap* indices_map) const override;

  absl::Status EmitEntryFunctionImpl(mlir::ImplicitLocOpBuilder& b,
                                     const EmitterHelper& helper,
                                     const IndexingMap& updates_map,
                                     const IndexingMap& indices_map,
                                     mlir::ValueRange thread_and_block_ids,
                                     mlir::Value output_tensor) const override;

 private:
  // Creates a 2D vector to store the accumulated updates in each thread.
  mlir::Value InitializeAccumulator(mlir::ImplicitLocOpBuilder& b) const;

  // The number of warps that process a single slice of the update.
  int64_t num_warps_per_slice_;
  // The number of indices that every warp iterates over. This is a useful
  // setting, if we know that the indices tensor is sorted.
  int64_t num_indices_per_warp_;
  // Vector size for the indices operand.
  int64_t indices_vector_size_;
};

std::unique_ptr<ScatterFusion> CreateScatterFusion(
    const HloFusionAnalysis& analysis);

}  // namespace gpu
}  // namespace xla

#endif  // XLA_BACKENDS_GPU_CODEGEN_EMITTERS_SCATTER_H_
