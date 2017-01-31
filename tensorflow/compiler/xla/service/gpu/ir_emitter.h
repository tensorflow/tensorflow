/* Copyright 2017 The TensorFlow Authors. All Rights Reserved.

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

// An XLA HLO graph may contain multiple computations. These computations
// fall into two types, nested and unnested. We translate each nested
// computation (e.g. the computation operand of a Map operator) to a device
// function. For each unnested computation composed of top-level
// HloInstructions, we generate a CUDA kernel for each HloInstruction.
//
// This file declares classes that translate an XLA HLO graph to LLVM IR for
// GPUs. IrEmitterNested emits LLVM IR for nested computations, and
// IrEmitterUnnested for unnested computations. The logic of emitting LLVM IR
// for each individual HloInstruction is largely the same between these two
// classes. Therefore, we implement the common logic in the Handle* functions in
// the superclass IrEmitter.

#ifndef TENSORFLOW_COMPILER_XLA_SERVICE_GPU_IR_EMITTER_H_
#define TENSORFLOW_COMPILER_XLA_SERVICE_GPU_IR_EMITTER_H_

#include <functional>
#include <map>
#include <memory>
#include <utility>
#include <vector>

#include "external/llvm/include/llvm/IR/Function.h"
#include "external/llvm/include/llvm/IR/IRBuilder.h"
#include "external/llvm/include/llvm/IR/Value.h"
#include "tensorflow/compiler/xla/service/buffer_assignment.h"
#include "tensorflow/compiler/xla/service/dfs_hlo_visitor_with_default.h"
#include "tensorflow/compiler/xla/service/gpu/elemental_ir_emitter.h"
#include "tensorflow/compiler/xla/service/gpu/hlo_to_ir_bindings.h"
#include "tensorflow/compiler/xla/service/gpu/ir_emitter_context.h"
#include "tensorflow/compiler/xla/service/gpu/kernel_thunk.h"
#include "tensorflow/compiler/xla/service/gpu/thunk.h"
#include "tensorflow/compiler/xla/service/hlo_computation.h"
#include "tensorflow/compiler/xla/service/hlo_instruction.h"
#include "tensorflow/compiler/xla/service/llvm_ir/ir_array.h"
#include "tensorflow/compiler/xla/service/llvm_ir/llvm_loop.h"
#include "tensorflow/compiler/xla/service/llvm_ir/loop_emitter.h"
#include "tensorflow/compiler/xla/statusor.h"
#include "tensorflow/compiler/xla/types.h"
#include "tensorflow/compiler/xla/xla_data.pb.h"
#include "tensorflow/core/lib/core/stringpiece.h"
#include "tensorflow/core/lib/gtl/array_slice.h"
#include "tensorflow/core/platform/types.h"

namespace xla {
namespace gpu {

// This class is the top-level API for the XLA HLO --> LLVM IR compiler.
// It implements the DfsHloVisitor interface and emits an LLVM IR program that
// implements the input HLO graph.
//
// Note: if `T` is a subclass of `IrEmitter` and a handler is not overridden in
//       either `IrEmitter` or `T`, the handler in `DfsHloVisitorWithDefault`
//       calls `T::DefaultAction`.
class IrEmitter : public DfsHloVisitorWithDefault {
 public:
  IrEmitter(const IrEmitter&) = delete;
  IrEmitter& operator=(const IrEmitter&) = delete;

  // The following methods implement the DfsHloVisitorWithDefault interface.
  Status DefaultAction(HloInstruction* hlo) override;
  Status HandleConstant(HloInstruction* constant,
                        const Literal& literal) override;
  Status HandleBitcast(HloInstruction* bitcast) override;
  Status HandleGetTupleElement(HloInstruction* get_tuple_element,
                               HloInstruction* operand) override;
  Status HandleDot(HloInstruction* dot, HloInstruction* lhs,
                   HloInstruction* rhs) override;
  Status HandleConvolution(HloInstruction* convolution, HloInstruction* lhs,
                           HloInstruction* rhs, const Window& window) override;
  Status HandleCrossReplicaSum(HloInstruction* crs) override;
  Status HandleInfeed(HloInstruction* infeed) override;
  Status HandleOutfeed(HloInstruction* outfeed) override;
  Status HandleSort(HloInstruction* sort, HloInstruction* operand) override;
  Status HandleSend(HloInstruction* send) override;
  Status HandleRecv(HloInstruction* recv) override;
  Status HandleParameter(HloInstruction* parameter) override;
  Status HandleReduce(HloInstruction* reduce, HloInstruction* arg,
                      HloInstruction* init_value,
                      tensorflow::gtl::ArraySlice<int64> dimensions,
                      HloComputation* function) override;
  Status HandleTuple(
      HloInstruction* tuple,
      tensorflow::gtl::ArraySlice<HloInstruction*> operands) override;
  Status HandleSelect(HloInstruction* select, HloInstruction* pred,
                      HloInstruction* on_true,
                      HloInstruction* on_false) override;
  Status HandleFusion(HloInstruction* fusion) override;
  Status HandleCall(HloInstruction* call,
                    tensorflow::gtl::ArraySlice<HloInstruction*> operands,
                    HloComputation* computation) override;
  Status HandleCustomCall(HloInstruction* custom_call,
                          tensorflow::gtl::ArraySlice<HloInstruction*> operands,
                          tensorflow::StringPiece custom_call_target) override;
  Status HandleRng(HloInstruction* random,
                   RandomDistribution /*distribution*/) override;

  Status FinishVisit(HloInstruction* root) override { return Status::OK(); }

 protected:
  // Constructs an IrEmitter with the given IrEmitter context.
  // ir_emitter_context is owned by the caller and should outlive the IrEmitter
  // object.
  explicit IrEmitter(const HloModuleConfig& hlo_module_config,
                     IrEmitterContext* ir_emitter_context, bool is_nested);

  // A convenient helper for calling HloToIrBindings::GetIrArray.
  llvm_ir::IrArray GetIrArray(const HloInstruction& inst) {
    return bindings_.GetIrArray(inst);
  }
  // A convenient helper for calling HloToIrBindings::GetBasePointer.
  llvm::Value* GetBasePointer(const HloInstruction& inst) const {
    return bindings_.GetBasePointer(inst);
  }
  // A convenient helper for calling BufferAssignment::GetAllocationIndex.
  BufferAllocation::Index GetAllocationIndex(const HloInstruction& hlo) const {
    return ir_emitter_context_->buffer_assignment()
        .GetUniqueTopLevelAllocation(&hlo)
        .ConsumeValueOrDie()
        ->index();
  }

  // Emit a singlethreaded or multithreaded loop that computes every element in
  // the result of the given HLO instruction. This produces a series of nested
  // loops (e.g. one for each dimension of the `hlo`'s shape). The body of the
  // inner-most loop is provided by the body_emitter function.
  virtual Status EmitTargetElementLoop(
      const HloInstruction& hlo,
      const llvm_ir::ElementGenerator& body_emitter) = 0;

  // Emits a call in IR to the given nested computation with the given operands
  // and output. If no IR function has been previously emitted for the
  // computation, also emits such a function.
  Status EmitCallToNestedComputation(
      const HloComputation& nested_computation,
      tensorflow::gtl::ArraySlice<llvm::Value*> operands, llvm::Value* output);

  // Emits an atomic operation that implements `nested_computation` in the
  // sequentially consistent memory model. `output_address` and `source_address`
  // are the arguments of the nested computation. For example,
  // atomicAdd(output_address, *source_address).
  Status EmitAtomicOperationForNestedComputation(
      const HloComputation& nested_computation, llvm::Value* output_address,
      llvm::Value* source_address);

  GpuElementalIrEmitter::NestedComputer GetNestedComputer() {
    return std::bind(&IrEmitter::ComputeNestedElement, this,
                     std::placeholders::_1, std::placeholders::_2);
  }

  IrEmitterContext* ir_emitter_context_;

  // The following fields track the IR emission state. According to LLVM memory
  // management rules, their memory is owned by the module.
  llvm::IRBuilder<> ir_builder_;

  // Mapping from HLO to its underlying LLVM value.
  HloToIrBindings bindings_;

  // Hlo configuration data used during code generation.
  const HloModuleConfig& hlo_module_config_;

 private:
  // Emits a series of nested loops for iterating over an operand array in the
  // dot operation. Loops are constructed in major to minor dimension layout
  // order. No loop is emitted for the given reduction_dimension. The function
  // returns an IrArray index for the given operand_array containing the indvars
  // of the loops. All dimensions of the index are filled except for the
  // reduction dimension. name_suffix is the string to append to the names of
  // LLVM constructs (eg, basic blocks) constructed by this method.
  llvm_ir::IrArray::Index EmitOperandArrayLoopNest(
      const llvm_ir::IrArray& operand_array, int64 reduction_dimension,
      tensorflow::StringPiece name_suffix, llvm_ir::ForLoopNest* loop_nest);

  // A helper method for EmitAtomicOperationForNestedComputation. Certain
  // computations, such as floating-point addition and integer maximization, can
  // be simply implemented using an LLVM atomic instruction. If "computation" is
  // one of this kind, emits code to do that and returns true; otherwise,
  // returns false.
  bool MaybeEmitSpecialAtomicOperation(const HloComputation& computation,
                                       llvm::Value* output_address,
                                       llvm::Value* source_address);

  StatusOr<llvm::Value*> ComputeNestedElement(
      const HloComputation& computation,
      tensorflow::gtl::ArraySlice<llvm::Value*> parameter_elements);

  // Emits an atomic operation that implements `nested_computation` in the
  // sequentially consistent memory model. `output_address` and `source_address`
  // are the arguments of the nested computation. For example,
  // atomicAdd(output_address, *source_address).
  StatusOr<llvm::Function*> EmitAtomicFunctionForNestedComputation(
      const HloComputation& nested_computation, llvm::Type* element_ir_type);

  // Map nested computations to emitted IR functions. This serves as a cache so
  // that IrEmitter does not emit multiple functions for the same
  // HloComputation.
  std::map<const HloComputation*, llvm::Function*> computation_to_ir_function_;
};

// Emits LLVM IR for unnested computations. Each HloInstruction is translated to
// a separate CUDA kernel. These kernels are inserted into the resultant module
// sorted in reverse postorder of the XLA HLO graph.
class IrEmitterUnnested : public IrEmitter {
 public:
  IrEmitterUnnested(const HloModuleConfig& hlo_module_config,
                    const HloComputation* hlo_computation,
                    bool has_hybrid_result,
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
  Status HandleCopy(HloInstruction* copy, HloInstruction* operand) override;
  Status HandleConvolution(HloInstruction* convolution, HloInstruction* lhs,
                           HloInstruction* rhs, const Window& window) override;
  Status HandleDot(HloInstruction* dot, HloInstruction* lhs_instruction,
                   HloInstruction* rhs_instruction) override;
  Status HandleFusion(HloInstruction* fusion) override;
  Status HandleGetTupleElement(HloInstruction* get_tuple_element,
                               HloInstruction* operand) override;
  Status HandleReduce(HloInstruction* reduce, HloInstruction* arg,
                      HloInstruction* init_value,
                      tensorflow::gtl::ArraySlice<int64> dimensions,
                      HloComputation* function) override;
  Status HandleSelectAndScatter(HloInstruction* instruction) override;
  Status HandleTuple(
      HloInstruction* tuple,
      tensorflow::gtl::ArraySlice<HloInstruction*> operands) override;
  Status HandleWhile(HloInstruction* xla_while, HloInstruction* init,
                     HloComputation* condition, HloComputation* body) override;
  Status HandleRng(HloInstruction* random,
                   RandomDistribution distribution) override;
  Status HandleSelect(HloInstruction* select, HloInstruction* pred,
                      HloInstruction* on_true,
                      HloInstruction* on_false) override;

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
  llvm::Function* BuildKernelPrototype(
      const HloInstruction& inst,
      tensorflow::gtl::ArraySlice<const HloInstruction*> escaped_hlos);

  // Emits the base pointers for `hlo` and its operands. `io_hlos` will store
  // all input/output HLOs among `hlo` and its operands.
  llvm::Function* EmitBasePointersForHloAndItsOperands(
      const HloInstruction& hlo, std::vector<const HloInstruction*>* io_hlos);

  // EmitColumnReduction and EmitRowReduction emit code for column and row
  // reduction of a matrix and/or 3D tensor. Row and column reduction have
  // different memory access pattern, so for performance their implementations
  // are significantly different.
  //
  // Emits code that reduces a matrix of shape [height x width] to a vector of
  // [width]. Other parameters have the same meaning as those of
  // `EmitReductionToVector`. Note that input shape might not be
  // [height x width], but can be bitcast to [height x weight] with "height"
  // being the major dimension.
  Status EmitColumnReduction(int64 height, int64 width, HloInstruction* reduce,
                             const Shape& input_shape,
                             const llvm_ir::ElementGenerator& input_gen,
                             const llvm_ir::ElementGenerator& init_value_gen,
                             HloComputation* reducer);

  // Emits code that reduces a 3D tensor of shape [depth x height x width] to a
  // vector of shape [height]. Other parameters have the same meaning as those
  // of `EmitReductionToVector`. Note that input shape might not be
  // [depth x height x width], but can be bitcast to [depth x height x weight]
  // with "depth" being the most major dimension.
  Status EmitRowReduction(int64 depth, int64 height, int64 width,
                          HloInstruction* reduce, const Shape& input_shape,
                          const llvm_ir::ElementGenerator& input_gen,
                          const llvm_ir::ElementGenerator& init_value_gen,
                          HloComputation* reducer);

  // Figures out whether `reduce` is a row or column reduction, and which
  // dimensions to reduce, and calls either `EmitRowReduction` or
  // `EmitColumnReduction` as appropriate. `input_shape` is the shape of the
  // input array, which is the operand of the Reduce instruction if unfused or
  // of the Fusion instruction if fused. `input_gen` and `init_value_gen`
  // generate elements of the input and the initial value. Other parameters mean
  // the same as for `HandleReduce`.
  //
  // Prerequisite: `IsReductionToVector(*reduce)`
  Status EmitReductionToVector(
      HloInstruction* reduce, const Shape& input_shape,
      const llvm_ir::ElementGenerator& input_gen,
      const llvm_ir::ElementGenerator& init_value_gen,
      tensorflow::gtl::ArraySlice<int64> dimensions_to_reduce,
      HloComputation* reducer);

  // Emits code to initialize buffer of `inst` in given `thunk`.
  Status EmitInitializer(const HloInstruction* inst, KernelThunk* thunk);

  // Returns a KernelThunk that invokes the kernel emitted for `inst`. The
  // caller needs to make sure `inst` outlives the lifetime of the returned
  // Thunk object.
  std::unique_ptr<Thunk> BuildKernelThunk(const HloInstruction* inst);

  // Returns a ConvolutionThunk that calls DNN to implement `inst`.
  std::unique_ptr<Thunk> BuildConvolutionThunk(const HloInstruction* inst);

  // Returns a GemmThunk that calls gemm to implement `inst`. The caller needs
  // to make sure `inst` outlives the lifetime of the returned Thunk object.
  std::unique_ptr<Thunk> BuildGemmThunk(const HloInstruction* inst);

  // Returns a CopyThunk that calls host-to-device cuMemcpy to implement `inst`.
  std::unique_ptr<Thunk> BuildCopyThunk(const HloInstruction* inst);

  // Returns a WhileThunk that invokes thunk sequences for 'condition' and
  // 'body' sub-computations of while instruction 'hlo'.
  std::unique_ptr<Thunk> BuildWhileThunk(const HloInstruction* hlo);

  // Returns a ForThunk which executes 'loop_limit' invocations of a thunk
  // sequence from the 'body' sub-computation of the while instruction 'hlo'.
  std::unique_ptr<Thunk> BuildForThunk(const HloInstruction* hlo,
                                       const int64 loop_limit);

  Status Postprocess(HloInstruction* hlo) override;

  // Returns the last generated thunk.
  Thunk* LastThunk() const { return thunk_sequence_->back().get(); }

  // The thunk sequence this IrEmitter generates for the input computation.
  std::unique_ptr<ThunkSequence> thunk_sequence_;

  // The HloComputation that this IrEmitter emits code for.
  const HloComputation* hlo_computation_;

  // Whether this computation will produce a hybrid result, that is the
  // computation produces a ShapedBuffer.
  bool has_hybrid_result_;
};

// Emits LLVM IR for a nested computation to the resultant function.
class IrEmitterNested : public IrEmitter {
 public:
  // Constructs an LLVM IR emitter for a nested HLO computation. `function` is
  // the containing IR function this emitter produces IR to. See
  // IrEmitter::IrEmitter for the meanings of other arguments.
  IrEmitterNested(const HloModuleConfig& hlo_module_config,
                  const HloComputation& nested_computation,
                  IrEmitterContext* ir_emitter_context);
  IrEmitterNested(const IrEmitterNested&) = delete;
  IrEmitterNested& operator=(const IrEmitterNested&) = delete;

  // Overrides the default empty implementation. Binds the given instruction
  // "parameter" with the parameter of the IR function.
  Status HandleParameter(HloInstruction* parameter) override;

  llvm::Function* GetEmittedFunction() const { return emitted_function_; }

  Status EmitTargetElementLoop(
      const HloInstruction& hlo,
      const llvm_ir::ElementGenerator& body_emitter) override;

 private:
  llvm::Function* EmitBasePointersForNestedComputation(
      const HloComputation& nested_computation,
      std::vector<const HloInstruction*>* io_hlos);

  llvm::Function* emitted_function_;
};

}  // namespace gpu
}  // namespace xla

#endif  // TENSORFLOW_COMPILER_XLA_SERVICE_GPU_IR_EMITTER_H_
