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

#include "tensorflow/compiler/xla/service/gpu/ir_emitter.h"
#include "tensorflow/compiler/xla/service/gpu/thunk.h"
#include "tensorflow/compiler/xla/service/llvm_ir/kernel_tiling.h"

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
class IrEmitterUnnested : public IrEmitter {
 public:
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
  // IrEmitter.
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
  Status HandleSelect(HloInstruction* select) override;
  Status HandleTupleSelect(HloInstruction* tuple_select) override;
  Status HandleCrossReplicaSum(HloInstruction* crs) override;
  Status HandleAfterAll(HloInstruction* gen_token) override;

  Status EmitTargetElementLoop(
      const HloInstruction& hlo,
      const llvm_ir::ElementGenerator& body_emitter) override;

  // Same as `EmitTargetElementLoop`, but in given `thunk` rather than
  // `LastThunk()`.
  Status EmitTargetElementLoopInThunk(
      const HloInstruction& hlo, const llvm_ir::ElementGenerator& body_emitter,
      KernelThunk* thunk);

 private:
  // Builds the appropriate thunk for the instruction hlo and returns the owning
  // pointer to it. The caller needs to make sure `inst` outlives the lifetime
  // of the returned Thunk object.
  std::unique_ptr<Thunk> BuildThunk(const HloInstruction* hlo);

  // Builds the prototype of the IR kernel for `inst` and adds it to the module.
  // This kernel takes as arguments pointers to the given buffer allocations.
  llvm::Function* BuildKernelPrototype(
      const HloInstruction& inst,
      tensorflow::gtl::ArraySlice<const BufferAllocation*> args);

  // Helper for writing extra outputs from inside a reduce kernel.
  Status EmitExtraOutputsForReduce(
      const HloInstruction* reduce, const llvm_ir::IrArray::Index& index,
      tensorflow::gtl::ArraySlice<
          std::pair<llvm_ir::ElementGenerator, ShapeIndex>>
          extra_output_gens);

  // EmitColumnReduction and EmitRowReduction emit code for column and row
  // reduction of a matrix and/or 3D tensor. Row and column reduction have
  // different memory access pattern, so for performance their implementations
  // are significantly different.
  //
  // Emits code that reduces a matrix of shape [height x width] to a vector of
  // [width]. Other parameters have the same meaning as those of
  // `EmitReductionToVector`. Note that input shape might not be
  // [height x width], but can be bitcast to [height x width] with "height"
  // being the major dimension.
  Status EmitColumnReduction(
      int64 height, int64 width, HloInstruction* reduce,
      const Shape& input_shape,
      tensorflow::gtl::ArraySlice<llvm_ir::ElementGenerator> input_gens,
      tensorflow::gtl::ArraySlice<llvm_ir::ElementGenerator> init_value_gens,
      tensorflow::gtl::ArraySlice<HloComputation*> reducers,
      tensorflow::gtl::ArraySlice<ShapeIndex> reduce_output_shapes,
      tensorflow::gtl::ArraySlice<
          std::pair<llvm_ir::ElementGenerator, ShapeIndex>>
          extra_output_gens);

  // Emits code that reduces a 3D tensor of shape [depth x height x width] to a
  // vector of shape [height]. Other parameters have the same meaning as those
  // of `EmitReductionToVector`. Note that input shape might not be
  // [depth x height x width], but can be bitcast to [depth x height x width]
  // with "depth" being the most major dimension.
  Status EmitRowReduction(
      int64 depth, int64 height, int64 width, HloInstruction* reduce,
      const Shape& input_shape,
      tensorflow::gtl::ArraySlice<llvm_ir::ElementGenerator> input_gens,
      tensorflow::gtl::ArraySlice<llvm_ir::ElementGenerator> init_value_gens,
      tensorflow::gtl::ArraySlice<HloComputation*> reducers,
      tensorflow::gtl::ArraySlice<ShapeIndex> reduce_output_shapes,
      tensorflow::gtl::ArraySlice<
          std::pair<llvm_ir::ElementGenerator, ShapeIndex>>
          extra_output_gens);

  // Emits code that reduces a tensor of arbitrary rank to a scalar.
  Status EmitReductionToScalar(
      HloInstruction* reduce, const Shape& input_shape,
      tensorflow::gtl::ArraySlice<llvm_ir::ElementGenerator> input_gens,
      tensorflow::gtl::ArraySlice<llvm_ir::ElementGenerator> init_value_gens,
      tensorflow::gtl::ArraySlice<HloComputation*> reducers,
      tensorflow::gtl::ArraySlice<ShapeIndex> reduce_output_shapes,
      tensorflow::gtl::ArraySlice<
          std::pair<llvm_ir::ElementGenerator, ShapeIndex>>
          extra_output_gens);

  // Figures out whether `reduce` is a row or column reduction, and which
  // dimensions to reduce, and calls either `EmitRowReduction` or
  // `EmitColumnReduction` as appropriate. `input_shape` is the shape of the
  // input array, which is the operand of the Reduce instruction if unfused or
  // of the Fusion instruction if fused. `input_gen` and `init_value_gen`
  // generate elements of the input and the initial value. Other parameters mean
  // the same as for `HandleReduce`.
  //
  // Multiple reduces can be emitted in the same loop, assuming they have the
  // same input and output shapes, and the same reduce dimensions.
  //
  // extra_output_gens can contain extra generators for intermediate outputs.
  // These must have the same shape as the reduce input as they are computed
  // when the reduce inputs are being read.
  //
  // Prerequisite: `IsReductionToVector(*reduce)`
  Status EmitReductionToVector(
      HloInstruction* reduce, const Shape& input_shape,
      tensorflow::gtl::ArraySlice<llvm_ir::ElementGenerator> input_gens,
      tensorflow::gtl::ArraySlice<llvm_ir::ElementGenerator> init_value_gens,
      tensorflow::gtl::ArraySlice<int64> dimensions_to_reduce,
      tensorflow::gtl::ArraySlice<HloComputation*> reducers,
      tensorflow::gtl::ArraySlice<ShapeIndex> reduce_output_shapes,
      tensorflow::gtl::ArraySlice<
          std::pair<llvm_ir::ElementGenerator, ShapeIndex>>
          extra_output_gens);

  // Returns true if a 0-2-1 tiling algorithm is already used to emit the kernel
  // for the hlo instruction.
  bool CheckAndEmitHloWithTile021(HloInstruction* hlo);
  // Emits a kernel for the hlo instruction using a 0-2-1 tiling algorithm and
  // returns the launch dimensions for the kernel. This is a helper to support
  // the implementation of CheckAndEmitHloWithTile021.
  LaunchDimensions EmitHlo021Tile(
      HloInstruction* hlo,
      tensorflow::gtl::ArraySlice<int64> reduced_output_dims,
      tensorflow::gtl::ArraySlice<int64> tiled_param_ids);
  // Generates the IrArray for each output of hlo and returns the number of
  // outputs.
  int ConstructIrArrayForOutputs(const HloInstruction& hlo,
                                 std::vector<llvm_ir::IrArray>* output_arrays);
  // Generates the IrArray for each input of hlo and returns the number of
  // inputs.
  int ConstructIrArrayForInputs(const HloInstruction& hlo,
                                std::vector<llvm_ir::IrArray>* param_arrays);
  // For each output of the `hlo` instruction, constructs the reduced shape for
  // the output with the given `reduced_output_dims` and cast the original
  // output IrArray element in `output_arrays` to the reduced shape. Returns
  // the number of outputs.
  int ConstructOutputReducedShapeAndCastOutputIrArrayToShape(
      const HloInstruction& hlo,
      const std::vector<llvm_ir::IrArray>& output_arrays,
      tensorflow::gtl::ArraySlice<int64> reduced_output_dims,
      std::vector<Shape>* output_reduced_shapes,
      std::vector<llvm_ir::IrArray>* output_in_reduced_shape_arrays);
  // For each input of the `hlo` instruction, checks its value in
  // `param_buffers` to find out whether the input has a reduced shape. If the
  // input has a reduced shape, constructs the reduced shape for the input and
  // casts the original input IrArray in `param_arrays` to the reduced shape.
  // Return the total number of inputs.
  int ConstructInputReducedShapeAndCastInputIrArrayToShape(
      const HloInstruction& hlo,
      const std::vector<llvm_ir::IrArray>& param_arrays,
      const std::vector<llvm::Value*>& param_buffers,
      tensorflow::gtl::ArraySlice<int64> reduced_output_dims,
      std::vector<Shape>* param_reduced_shapes,
      std::vector<llvm_ir::IrArray>* param_in_reduced_shape_arrays);

  // Returns a KernelThunk that invokes the kernel emitted for `inst`. The
  // caller needs to make sure `inst` outlives the lifetime of the returned
  // Thunk object. The kernel implementation will be unrolled if unroll_factor
  // is greater than one. 'implements_whole_instruction' specifies whether this
  // KernelThunk implements the whole 'inst' HloInstruction. In some cases
  // 'inst' will be implemented by a sequence of Thunks.
  std::unique_ptr<KernelThunk> BuildKernelThunk(
      const HloInstruction* inst, bool implements_whole_instruction,
      int unroll_factor = 1);

  // Returns a FftThunk that calls cuFFT to implement `inst`.
  std::unique_ptr<Thunk> BuildFftThunk(const HloInstruction* inst);

  // Returns a GemmThunk that calls gemm to implement `inst`. The caller needs
  // to make sure `inst` outlives the lifetime of the returned Thunk object.
  std::unique_ptr<Thunk> BuildGemmThunk(const HloInstruction* inst);

  // Returns a thunk that, given a reduce or select-and-scatter op, initializes
  // its memory to the appropriate initial value.
  StatusOr<std::unique_ptr<Thunk>> BuildInitializerThunk(
      const HloInstruction* hlo, const ShapeIndex& index = {});

  // Returns a thunk that calls host-to-device cuMemcpy to implement `inst`.
  std::unique_ptr<Thunk> BuildHostToDeviceCopyThunk(const HloInstruction* inst);

  // Returns a thunk that calls device-to-device cuMemcpy to implement `inst`.
  std::unique_ptr<Thunk> BuildDeviceToDeviceCopyThunk(
      const HloInstruction* inst);

  // Returns an InfeedThunk that performs a host-to-device memcpy to implement
  // `inst`.
  std::unique_ptr<Thunk> BuildInfeedThunk(const HloInstruction* inst);

  // Returns an OutfeedThunk that performs a device-to-host memcpy to implement
  // `inst`.
  std::unique_ptr<Thunk> BuildOutfeedThunk(const HloInstruction* inst);

  // Returns a WhileThunk that invokes thunk sequences for 'condition' and
  // 'body' sub-computations of while instruction 'hlo'.
  std::unique_ptr<Thunk> BuildWhileThunk(const HloInstruction* hlo);

  // Returns a ForThunk which executes 'loop_limit' invocations of a thunk
  // sequence from the 'body' sub-computation of the while instruction 'hlo'.
  std::unique_ptr<Thunk> BuildForThunk(const HloInstruction* hlo,
                                       const int64 loop_limit);

  // Returns a ConditionalThunk that executes the thunk sequence for
  // 'true_computation' or 'false_computation' depending on the value of the
  // predicate in the given conditional instruction.
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
