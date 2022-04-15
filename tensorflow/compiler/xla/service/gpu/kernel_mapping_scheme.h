/* Copyright 2019 The TensorFlow Authors. All Rights Reserved.

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

#ifndef TENSORFLOW_COMPILER_XLA_SERVICE_GPU_KERNEL_MAPPING_SCHEME_H_
#define TENSORFLOW_COMPILER_XLA_SERVICE_GPU_KERNEL_MAPPING_SCHEME_H_

#include "absl/container/inlined_vector.h"
#include "absl/types/span.h"
#include "llvm/IR/Value.h"
#include "tensorflow/compiler/xla/service/hlo_instruction.h"
#include "tensorflow/compiler/xla/service/llvm_ir/ir_array.h"
#include "tensorflow/compiler/xla/service/llvm_ir/loop_emitter.h"
#include "tensorflow/compiler/xla/util.h"

namespace xla {
namespace gpu {

using Vector3 = std::array<int64_t, 3>;

// Describes tiling used by the kernel.
//
// Used by reductions and 021 transpose algorithm. Both algorithms operate over
// "logical" 3D views over input arrays, hence tiling and number of threads
// information has only 3 dimensions.
//
// In the presence of virtual threadIdx/blockIdx scaling, all accessors are
// "logical", unless otherwise specified.
class TilingScheme {
 public:
  enum { DimZ = 0, DimY, DimX, DimTot };

  enum IndexingOrder {
    // Thread reads consecutive elements.
    LinearIndexingX,
    // Thread reads strided elements while keeping memory coalescing.
    StridedIndexingX,
  };

  TilingScheme(Vector3 dims_in_elems, Vector3 tile_sizes, Vector3 num_threads,
               IndexingOrder indexing_order, int vector_size,
               int scaling_factor)
      : dims_in_elems_(dims_in_elems),
        tile_sizes_(tile_sizes),
        num_threads_(num_threads),
        indexing_order_(indexing_order),
        vector_size_(vector_size),
        thread_id_virtual_scaling_(scaling_factor) {
    CHECK_EQ(tile_sizes[2] % vector_size_, 0);
  }

  static std::string IndexingOrderToString(IndexingOrder order) {
    switch (order) {
      case LinearIndexingX:
        return "linear";
      case StridedIndexingX:
        return "strided";
    }
  }

  std::string ToString() const {
    return absl::StrJoin(
        {absl::StrFormat("dims_in_elems = {%s}",
                         absl::StrJoin(dims_in_elems_, ", ")),
         absl::StrFormat("tile_sizes = {%s}", absl::StrJoin(tile_sizes_, ", ")),
         absl::StrFormat("num_threads = {%s}",
                         absl::StrJoin(num_threads_, ", ")),
         absl::StrFormat("indexing_order = %s",
                         IndexingOrderToString(indexing_order_)),
         absl::StrFormat("vector_size = %d", vector_size_)},
        ", ");
  }

  // Number of elements in each dimension (Z/Y/X respectively).
  absl::Span<const int64_t> GetDimsInElems() const { return dims_in_elems_; }

  Vector3 GetDimsInBlocks() const {
    return {GetDimInBlock(0), GetDimInBlock(1), GetDimInBlock(2)};
  }

  // Number of blocks required to "cover" the given dimension.
  int64_t GetDimInBlock(int d) const {
    return CeilOfRatio(dims_in_elems_[d], GetBlockTileSizeFor(d));
  }

  // Tile size for a given dimensions per thread.
  //
  // Equals to the number of iterations in the loop each tile will make.
  int64_t GetTileSizeFor(int d) const { return tile_sizes_.at(d); }

  // Tile size for a given dimension per entire thread block.
  int64_t GetBlockTileSizeFor(int d) const {
    return num_threads_.at(d) * tile_sizes_.at(d);
  }

  // Number of threads in given dimension.
  int64_t GetNumThreadsFor(int d) const { return num_threads_.at(d); }

  // Number of logical threads per block.
  int64_t GetNumThreadsPerBlock() const {
    return GetNumThreadsFor(0) * GetNumThreadsFor(1) * GetNumThreadsFor(2);
  }

  // Number of logical blocks.
  int64_t GetNumberOfBlocks() const {
    return GetDimInBlock(0) * GetDimInBlock(1) * GetDimInBlock(2);
  }

  // Number of physical blocks launched (with scaling applied).
  int64_t GetNumberOfBlocksPhysical() const {
    return CeilOfRatio(GetNumberOfBlocks(), thread_id_virtual_scaling_);
  }

  // Number of physical threads per block launched (with scaling applied).
  int64_t GetNumThreadsPerBlockPhysical() const {
    return GetNumThreadsPerBlock() * thread_id_virtual_scaling_;
  }

  IndexingOrder GetIndexingOrder() const { return indexing_order_; }
  int GetVectorSize() const { return vector_size_; }

  // Scaling factor for transforming physical threadId to logical.
  int GetThreadIdScalingFactor() const { return thread_id_virtual_scaling_; }

 private:
  // The number of elements in each dimension.
  const Vector3 dims_in_elems_;

  // The number of elements for each dimension of a tile.
  const Vector3 tile_sizes_;

  // Number of threads implicitly assigned to each dimension.
  const Vector3 num_threads_;

  const IndexingOrder indexing_order_;

  // Vector size for dimension X.
  const int vector_size_;

  // Scaling apply to transform physical threadIdx into logical.
  const int64_t thread_id_virtual_scaling_ = 1;
};

class ReductionCodegenInfo {
 public:
  explicit ReductionCodegenInfo(TilingScheme mapping_scheme,
                                int num_partial_results, bool is_row_reduction,
                                bool is_race_free)
      : tiling_scheme_(mapping_scheme),
        num_partial_results_(num_partial_results),
        is_row_reduction_(is_row_reduction),
        is_race_free_(is_race_free) {
    if (num_partial_results > 1) {
      CHECK_EQ(num_partial_results,
               mapping_scheme.GetTileSizeFor(TilingScheme::DimX));
    }
  }

  const TilingScheme& GetTilingScheme() const { return tiling_scheme_; }

  int GetNumPartialResults() const { return num_partial_results_; }
  bool IsRaceFree() const { return is_race_free_; }

 private:
  friend class ReductionCodegenState;

  const TilingScheme tiling_scheme_;
  int num_partial_results_;
  bool is_row_reduction_;
  bool is_race_free_;
};

class ReductionCodegenState {
 public:
  struct ReductionCalculationState {
    llvm::GlobalVariable* shared_cache;
    llvm::Value* initial_value;
    llvm::AllocaInst* partial_result_address;
    llvm::AllocaInst* input_address;
    llvm_ir::ElementGenerator input_gen;
  };

  explicit ReductionCodegenState(
      const ReductionCodegenInfo& reduction_codegen_info)
      : reduction_codegen_info_(reduction_codegen_info) {}

  const TilingScheme& GetTilingScheme() const {
    return reduction_codegen_info_.tiling_scheme_;
  }

  int GetNumPartialResults() const {
    return reduction_codegen_info_.num_partial_results_;
  }

  bool IsRowReduction() const {
    return reduction_codegen_info_.is_row_reduction_;
  }

  bool IsRaceFree() const { return reduction_codegen_info_.IsRaceFree(); }

  const ReductionCalculationState& GetCalculationStateFor(
      const HloInstruction* instruction, int operand_idx) const {
    const ReductionOpState& op_state = state_.at(instruction);
    CHECK_LT(operand_idx, op_state.size());
    return op_state[operand_idx];
  }

  void SetCalculationStateFor(
      const ReductionCalculationState& calculation_state,
      const HloInstruction* instruction, int operand_idx) {
    ReductionOpState& op_state = state_[instruction];
    CHECK_EQ(operand_idx, op_state.size());
    op_state.push_back(calculation_state);
  }

 private:
  ReductionCodegenInfo reduction_codegen_info_;

  // One state per reduction operand.
  using ReductionOpState = absl::InlinedVector<ReductionCalculationState, 2>;

  // HloInstruction -> operand_idx -> cache
  absl::flat_hash_map<const HloInstruction*, ReductionOpState> state_;
};

}  // end namespace gpu
}  // end namespace xla

#endif  // TENSORFLOW_COMPILER_XLA_SERVICE_GPU_KERNEL_MAPPING_SCHEME_H_
