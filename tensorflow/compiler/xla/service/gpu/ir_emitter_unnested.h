/* Copyright 2018 The TensorFlow Authors. All Rights Reserved.

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

#ifndef TENSORFLOW_COMPILER_XLA_SERVICE_GPU_IR_EMITTER_UNNESTED_H_
#define TENSORFLOW_COMPILER_XLA_SERVICE_GPU_IR_EMITTER_UNNESTED_H_

#include "absl/container/inlined_vector.h"
#include "tensorflow/compiler/xla/service/gpu/ir_emitter.h"
#include "tensorflow/compiler/xla/service/gpu/kernel_mapping_scheme.h"
#include "tensorflow/compiler/xla/service/gpu/sequential_thunk.h"
#include "tensorflow/compiler/xla/service/gpu/thunk.h"
#include "tensorflow/compiler/xla/service/gpu/thunk_emitter.h"
#include "tensorflow/compiler/xla/service/hlo_computation.h"
#include "tensorflow/compiler/xla/service/llvm_ir/kernel_support_library.h"

namespace xla {
namespace gpu {

// Emits LLVM IR for an "unnested computation".
//
// An unnested computation is an HloComputation which you run by executing one
// or more kernels for each HloInstruction it contains.  Examples of unnested
// computations:
//
//  - An HloModule's root computation,
//  - The body of an HLO while loop,
//  - The true/false computation of an HLO conditional.
//
// Note the opportunity for confusion -- the while loop's computation is nested
// within the root computation, but it's emitted using IrEmitterUnnested!  Don't
// think about it too hard.
//
// Examples of things that are not unnested computations:
//
//  - The reducer of a kReduce HLO.  This is emitted using IrEmitterNested.
//  - The body of a fusion node.  IrEmitterUnenested emits the relevant code
//    within a kernel function using FusedIrEmitter.  (FusedIrEmitter is not
//    really an IrEmitter, but is more an "IR generator generator".)
//
class IrEmitterUnnested : public IrEmitter,
                          private ThunkEmitter::EmissionContext {
 public:
  // A function object to generate code to process one element in a tile.
  //
  // hlo: the instruction for which the code is generated for.
  // index: the index for the first output element of the current thread.
  // y_loc: The y coordinate within a tile.
  // x_loc: The x coordinate within a tile.
  // kernel_info: Other information to support the kernel code generation.
  // x_iter_num: When a thread process N elements in the X dimension, x_iter_num
  //             has a value of 0..N-1 to identify the element being process.
  using EmitElementFunction = std::function<void(
      const llvm_ir::IrArray::Index& index, llvm::Value* y_loc,
      llvm::Value* x_loc, int64 x_iter_num)>;

  using ConstantGenerator = std::function<llvm::Value*(int64)>;

  // A function to generate the code to emit the entire tile.
  using TileElementGenerator = std::function<void(
      llvm::Value* y, llvm::Value* x, const llvm_ir::IrArray::Index& index,
      const string& loop_name, llvm::Value* tile_height,
      llvm::Value* tile_width, KernelSupportLibrary* ksl)>;

  IrEmitterUnnested(const HloModuleConfig& hlo_module_config,
                    const HloComputation* hlo_computation,
                    IrEmitterContext* ir_emitter_context);
  IrEmitterUnnested(const IrEmitterUnnested&) = delete;
  IrEmitterUnnested& operator=(const IrEmitterUnnested&) = delete;

  // Transfers the ownship of thunk_sequence_ out.
  std::unique_ptr<ThunkSequence> ConsumeThunkSequence() {
    return std::move(thunk_sequence_);
  }

  Status DefaultAction(HloInstruction* hlo) override;

  // IrEmitterUnnested handles the following instructions differently from
  // IrEmitter. It also mixes in some special handling for custom kernels
  // via the ThunkEmitter.
  Status HandleCopy(HloInstruction* copy) override;
  Status HandleConditional(HloInstruction* conditional) override;
  Status HandleConvolution(HloInstruction* convolution) override;
  Status HandleCustomCall(HloInstruction* custom_call) override;
  Status HandleDot(HloInstruction* dot) override;
  Status HandleFft(HloInstruction* fft) override;
  Status HandleFusion(HloInstruction* fusion) override;
  Status HandleGetTupleElement(HloInstruction* get_tuple_element) override;
  Status HandleReduce(HloInstruction* reduce) override;
  Status HandleSelectAndScatter(HloInstruction* instruction) override;
  Status HandleTuple(HloInstruction* tuple) override;
  Status HandleWhile(HloInstruction* xla_while) override;
  Status HandleInfeed(HloInstruction* xla_infeed) override;
  Status HandleOutfeed(HloInstruction* outfeed) override;
  Status HandleRng(HloInstruction* random) override;
  Status HandleRngGetAndUpdateState(HloInstruction* rng_state) override;
  Status HandleScatter(HloInstruction* scatter) override;
  Status HandleSelect(HloInstruction* select) override;
  Status HandleSort(HloInstruction* sort) override;
  Status HandleTriangularSolve(HloInstruction* hlo) override;
  Status HandleTupleSelect(HloInstruction* tuple_select) override;
  Status HandleAllReduce(HloInstruction* crs) override;
  Status HandleAfterAll(HloInstruction* after_all) override;
  Status HandleReplicaId(HloInstruction* hlo) override;
  Status HandleCollectivePermute(HloInstruction* hlo) override;

  Status EmitTargetElementLoop(
      const HloInstruction& hlo,
      const llvm_ir::ElementGenerator& body_emitter) override;

  // Same as `EmitTargetElementLoop`, but in given `thunk` rather than
  // `LastThunk()`.
  Status EmitTargetElementLoopInThunk(
      const HloInstruction& hlo, const llvm_ir::ElementGenerator& body_emitter,
      KernelThunk* thunk);

  // Emits LLVM global variables corresponding to constant instructions.
  Status EmitConstantGlobals();

 private:
  // Add a owning Thunk object to the thunk sequence.
  void AddThunkToThunkSequence(std::unique_ptr<Thunk> thunk) override {
    thunk_sequence_->emplace_back(std::move(thunk));
  }

  // A convenient helper for calling BufferAssignment::GetUniqueSlice.
  StatusOr<BufferAllocation::Slice> MaybeGetAllocationSlice(
      const HloInstruction& hlo, const ShapeIndex& index) const override {
    return ir_emitter_context_->buffer_assignment().GetUniqueSlice(&hlo, index);
  }

  BufferAllocation::Slice GetAllocationSlice(
      const HloInstruction& hlo, const ShapeIndex& index = {}) const {
    return MaybeGetAllocationSlice(hlo, index).ConsumeValueOrDie();
  }

  int64 ByteSizeOf(const Shape& shape) const override {
    return llvm_ir::ByteSizeOf(
        shape, ir_emitter_context_->llvm_module()->getDataLayout());
  }

  const se::Platform* platform() const override {
    return ir_emitter_context_->platform();
  }

  // Builds the prototype of the IR kernel for `inst` and adds it to the module.
  // This kernel takes as arguments pointers to the given buffer allocations.
  llvm::Function* BuildKernelPrototype(
      const HloInstruction& inst,
      absl::Span<const BufferAllocation* const> args);

  // Helper for writing extra outputs from inside a reduce kernel.
  Status EmitExtraOutputsForReduce(
      const HloInstruction* unnested_hlo, const llvm_ir::IrArray::Index& index,
      bool use_linear_index,
      absl::Span<const std::pair<llvm_ir::ElementGenerator, ShapeIndex>>
          extra_output_gens);

  // Generates code for reduction to contiguous dimensions.
  //
  // output_instructions: Output instructions in the computation: instruction
  // itself if it's not a fusion, fusion root if fusion is not multi-output, and
  // elements of the fusion multi-output tuple otherwise.
  Status EmitReductionFromOrToContiguousDimensions(
      HloInstruction* unnested_hlo,
      absl::Span<HloInstruction* const> output_instructions);

  // Computes the KernelMappingScheme for the reduce HLO and indicates whether
  // the reduction is a row reduction. For an un-fused reduce op, unnested_hlo
  // and first_reduce are the same instruction. For a kInput fusion,
  // unnested_hlo is the fusion instruction while first_reduce is the first
  // reduce op.
  ReductionCodegenInfo ComputeReductionCodegenInfo(
      const HloInstruction* unnested_hlo, const HloInstruction* first_reduce);

  // Generates code for input-fusible slices.
  //
  // Prerequisite: ROOT is either a slice or a tuple of slices. The input shapes
  // of all ROOT slices need to be the same while their output shapes can be
  // different. On the other hand, the input ranges of slices can be
  // overlapping. Further generalization/specialization when the needs are seen
  // in the future.
  Status EmitInputFusibleNonStridedSlices(HloInstruction* unnested_hlo);

  void EmitElementForInputFusibleSlices(
      HloInstruction* unnested_hlo,
      const llvm_ir::IrArray::Index& slice_input_index);

  // Emits code for an in-place scatter, modifying `thunk`s launch dimensions in
  // the process. `scatter` may be fused, scatter indices are taken from
  // `scatter_indices_gen`, updates from`updates_gen`. The output buffer is
  // expected to have the operand values in it already. If unique_indices
  // is false, we will use an atomic update. Using false for unique_indices
  // is safe only when it is guaranteed that there are no duplicate
  // indices.
  // When using unique_indices=true, it is the caller's responsibility to
  // ensure there is no overlap.
  Status EmitScatter(Thunk* thunk, HloInstruction* scatter,
                     const llvm_ir::ElementGenerator& scatter_indices_gen,
                     const llvm_ir::ElementGenerator& updates_gen);

  // Returns true if a 0-2-1 tiling algorithm is already used to emit the kernel
  // for the hlo instruction.
  bool CheckAndEmitHloWithTile021(HloInstruction* hlo);

  // Emits a kernel for the hlo instruction using a 0-2-1 tiling algorithm and
  // sets the corresponding launch dimensions. This is a helper to support
  // the implementation of CheckAndEmitHloWithTile021.
  void EmitHlo021Tile(HloInstruction* hlo, Thunk* kernel_thunk,
                      absl::Span<const int64> reduced_output_dims,
                      absl::Span<const int64> tiled_param_ids);

  // Emits a kernel for the hlo instruction using the given kernel mapping
  // scheme.
  //
  // Returns lane_id as an LLVM value.
  llvm::Value* EmitTilingKernel(
      const KernelMappingScheme& mapping_scheme, llvm::Type* index_ty,
      const TileElementGenerator& tile_element_generator);

  // Emits code to process a tensor element in a tile for the given kCopy HLO
  // that performs a 0-2-1 transpose.
  void EmitTileElementForCopy(
      HloInstruction* hlo, const llvm_ir::IrArray::Index& index,
      const KernelMappingScheme& mapping_scheme, llvm::Value* y_loc,
      llvm::Value* x_loc, int64 x_iter_num,
      absl::Span<llvm::Value* const> param_shmem_buffers);

  // Emits code to process a tensor element in a tile for the given kLoop fusion
  // HLO containing parameters that are 0-2-1 transpose of its outputs.
  void EmitTileElementForFusion(
      HloInstruction* hlo, const llvm_ir::IrArray::Index& index,
      const KernelMappingScheme& mapping_scheme, llvm::Value* y_loc,
      llvm::Value* x_loc, int64 x_iter_num,
      absl::Span<llvm::Value* const> param_shmem_buffers);

  // Emits code to process a tensor element in a tile for the given input hlo
  // that is either a unnested kReduce or a kInput fusion.
  //
  // Calculates and stores the temporary reduction value in the corresponding
  // alloca.
  void EmitTileElementForReduction(
      HloInstruction* unnested_hlo, const Shape& reduction_operand_shape,
      absl::Span<HloInstruction* const> output_instructions,
      const llvm_ir::IrArray::Index& index,
      const ReductionCodegenInfo& kernel_info,
      absl::Span<HloComputation* const> reducers, int64 x_iter_num);

  // Prepares for the code generation for a tile block of a reduction kernel.
  //
  // Create accumulator alloca's, populate them with initial values, and store
  // inside reduction_info.
  void EmitPrologueForReduction(
      HloInstruction* unnested_hlo, ReductionCodegenInfo* reduction_info,
      absl::Span<HloInstruction* const> reduce_instructions,
      llvm::Type* index_type);

  void EmitPrologueForOneReduction(HloInstruction* unnested_hlo,
                                   HloInstruction* reduce_inst, int reduce_idx,
                                   ReductionCodegenInfo* kernel_info,
                                   GpuElementalIrEmitter* elemental_emitter);

  // Wraps up the code generation for a tile block of a reduction kernel: write
  // the calculated output into the output tensor.
  void EmitEpilogueForReduction(
      HloInstruction* unnested_hlo, const ReductionCodegenInfo& reduction_info,
      absl::Span<const HloInstruction* const> reduce_instructions,
      absl::Span<const ShapeIndex> reduction_output_shape_indices,
      absl::Span<HloComputation* const> reducers, llvm::Value* lane_id);

  // For each reducer, emits the shuffle-down loop to accumulate the partial
  // result to the global result.
  void EmitFullWarpShuffleDownLoopForAllReduces(
      absl::Span<HloComputation* const> reducers,
      absl::Span<llvm::AllocaInst* const> partial_result_addresses);

  // Returns a KernelThunk that invokes the kernel emitted for `inst`. The
  // caller needs to make sure `inst` outlives the lifetime of the returned
  // Thunk object. The kernel implementation will be unrolled if unroll_factor
  // is greater than one. 'implements_whole_instruction' specifies whether this
  // KernelThunk implements the whole 'inst' HloInstruction. In some cases
  // 'inst' will be implemented by a sequence of Thunks.
  std::unique_ptr<KernelThunk> BuildKernelThunk(
      const HloInstruction* inst, bool implements_whole_instruction,
      int unroll_factor = 1);

  // Returns a thunk that, given a reduce or select-and-scatter op, initializes
  // its memory to the appropriate initial value.
  StatusOr<std::unique_ptr<Thunk>> BuildInitializerThunk(
      HloInstruction* hlo, const ShapeIndex& index = {});

  // Returns a WhileThunk that invokes thunk sequences for 'condition' and
  // 'body' sub-computations of while instruction 'hlo'.
  std::unique_ptr<Thunk> BuildWhileThunk(const HloInstruction* hlo);

  // Returns a ForThunk which executes 'loop_limit' invocations of a thunk
  // sequence from the 'body' sub-computation of the while instruction 'hlo'.
  std::unique_ptr<Thunk> BuildForThunk(const HloInstruction* hlo,
                                       const int64 loop_limit);

  // Returns a ConditionalThunk which executes the thunk sequence for the
  // 'branch_computation' corresponding to the predicate/branch_index of the
  // given conditional instruction.
  std::unique_ptr<Thunk> BuildConditionalThunk(const HloInstruction* hlo);

  Status Postprocess(HloInstruction* hlo) override;

  // Returns the last generated thunk.
  Thunk* LastThunk() const { return thunk_sequence_->back().get(); }

  // The thunk sequence this IrEmitter generates for the input computation.
  std::unique_ptr<ThunkSequence> thunk_sequence_;

  // The HloComputation that this IrEmitter emits code for.
  const HloComputation* hlo_computation_;
};

}  // namespace gpu
}  // namespace xla

#endif  // TENSORFLOW_COMPILER_XLA_SERVICE_GPU_IR_EMITTER_UNNESTED_H_
