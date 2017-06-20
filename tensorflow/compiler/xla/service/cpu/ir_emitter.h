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

#ifndef TENSORFLOW_COMPILER_XLA_SERVICE_CPU_IR_EMITTER_H_
#define TENSORFLOW_COMPILER_XLA_SERVICE_CPU_IR_EMITTER_H_

#include <stddef.h>
#include <map>
#include <string>
#include <unordered_map>
#include <vector>

#include "external/llvm/include/llvm/ADT/Triple.h"
#include "external/llvm/include/llvm/IR/Function.h"
#include "external/llvm/include/llvm/IR/IRBuilder.h"
#include "external/llvm/include/llvm/IR/Module.h"
#include "external/llvm/include/llvm/IR/Value.h"
#include "tensorflow/compiler/xla/service/buffer_assignment.h"
#include "tensorflow/compiler/xla/service/dfs_hlo_visitor_with_default.h"
#include "tensorflow/compiler/xla/service/hlo_computation.h"
#include "tensorflow/compiler/xla/service/hlo_instruction.h"
#include "tensorflow/compiler/xla/service/hlo_module_config.h"
#include "tensorflow/compiler/xla/service/llvm_ir/alias_analysis.h"
#include "tensorflow/compiler/xla/service/llvm_ir/ir_array.h"
#include "tensorflow/compiler/xla/service/llvm_ir/loop_emitter.h"
#include "tensorflow/compiler/xla/service/name_uniquer.h"
#include "tensorflow/compiler/xla/statusor.h"
#include "tensorflow/compiler/xla/types.h"
#include "tensorflow/compiler/xla/xla_data.pb.h"
#include "tensorflow/core/lib/core/stringpiece.h"
#include "tensorflow/core/lib/gtl/array_slice.h"
#include "tensorflow/core/platform/macros.h"
#include "tensorflow/core/platform/types.h"

namespace xla {
namespace cpu {

// This class is the top-level API for the XLA HLO --> LLVM IR compiler.  It
// implements the DfsHloVisitor interface and emits HLO computations as LLVM IR
// functions.
class IrEmitter : public DfsHloVisitorWithDefault {
 public:
  // Create a new LLVM IR emitter.
  //
  // hlo_module: the HLO module we are emitting IR for.
  // assignment: a BufferAssignment from which we know which temporary buffers
  //             are used by the HLO nodes.
  // llvm_module: the LLVM module to emit IR into.
  // hlo_to_profile_idx: the mapping from HLO to its index in the profiling
  //                     array.
  IrEmitter(const HloModule& hlo_module, const BufferAssignment& assignment,
            llvm::Module* llvm_module,
            const std::unordered_map<const HloInstruction*, size_t>*
                hlo_to_profile_idx);
  ~IrEmitter() override;

  // Emit and return the given HLO computation as an LLVM IR
  // function.
  //
  // function_name_prefix is the desired name of the function. If the name is
  // not unique among already emitted functions then a suffix is appended to
  // make the name unique.
  //
  // is_entry_computation indicates that this is the entry computation of the
  // HLO module.
  //
  // If 'instruction_order' is not NULL, then the HLO instructions are emitted
  // in the given order.  In this case, 'instruction_order' must be a
  // topological sort of the set of nodes accessible from the root of the
  // computation.
  StatusOr<llvm::Function*> EmitComputation(
      HloComputation* computation, const string& function_name_prefix,
      bool is_entry_computation,
      std::vector<const HloInstruction*>* instruction_order);

 protected:
  //
  // The following methods implement the DfsHloVisitor interface.
  //
  // Default action which emits code for most operations. Operations which are
  // special in some way are handled explicitly in HandleFoo methods.
  Status DefaultAction(HloInstruction* hlo_instruction) override;

  Status HandleBitcast(HloInstruction* bitcast) override;
  Status HandleConstant(HloInstruction* constant,
                        const Literal& literal) override;
  Status HandleCopy(HloInstruction* copy, HloInstruction* operand) override;
  Status HandleGetTupleElement(HloInstruction* get_tuple_element,
                               HloInstruction* operand) override;
  Status HandleSelect(HloInstruction* select, HloInstruction* pred,
                      HloInstruction* on_true,
                      HloInstruction* on_false) override;
  Status HandleDot(HloInstruction* dot, HloInstruction* lhs,
                   HloInstruction* rhs) override;
  Status HandleConvolution(HloInstruction* convolution, HloInstruction* lhs,
                           HloInstruction* rhs, const Window& window) override;
  Status HandleCrossReplicaSum(HloInstruction* crs) override;
  Status HandleInfeed(HloInstruction* infeed) override;
  Status HandleOutfeed(HloInstruction* infeed) override;
  Status HandleSort(HloInstruction* sort, HloInstruction* operand) override;
  Status HandleParameter(HloInstruction* parameter) override;
  Status HandleReduce(HloInstruction* reduce, HloInstruction* arg,
                      HloInstruction* init_value,
                      tensorflow::gtl::ArraySlice<int64> dimensions,
                      HloComputation* function) override;
  Status HandleReduceWindow(HloInstruction* reduce_window,
                            HloInstruction* operand, const Window& window,
                            HloComputation* function) override;
  Status HandleSelectAndScatter(HloInstruction* instruction) override;
  Status HandleSend(HloInstruction* send) override;
  Status HandleSlice(HloInstruction* slice,
                     HloInstruction* /*operand*/) override;
  Status HandleDynamicSlice(HloInstruction* dynamic_slice,
                            HloInstruction* /*operand*/,
                            HloInstruction* /*start_indices*/) override;
  Status HandleDynamicUpdateSlice(HloInstruction* dynamic_update_slice,
                                  HloInstruction* /*operand*/,
                                  HloInstruction* /*update*/,
                                  HloInstruction* /*start_indices*/) override;
  Status HandleRecv(HloInstruction* recv) override;
  Status HandlePad(HloInstruction* pad) override;
  Status HandleTuple(
      HloInstruction* tuple,
      tensorflow::gtl::ArraySlice<HloInstruction*> operands) override;
  Status HandleMap(
      HloInstruction* map,
      tensorflow::gtl::ArraySlice<HloInstruction*> operands,
      HloComputation* function,
      tensorflow::gtl::ArraySlice<HloInstruction*> static_operands) override;
  Status HandleFusion(HloInstruction* fusion) override;
  Status HandleCall(HloInstruction* call) override;
  Status HandleCustomCall(HloInstruction* custom_call,
                          tensorflow::gtl::ArraySlice<HloInstruction*> operands,
                          tensorflow::StringPiece custom_call_target) override;
  Status HandleWhile(HloInstruction* xla_while) override;
  Status FinishVisit(HloInstruction* root) override;

  Status Preprocess(HloInstruction* hlo) override;
  Status Postprocess(HloInstruction* visited) override;

 private:
  // Private helper to initialize an IR function for the computation.
  void InitializeIrFunction(const string& function_name,
                            bool is_entry_computation);

  // Convenience function to generate a GEP into the profile counter parameter
  // which would correspond to the index for a given HLO.
  llvm::Value* GetProfileCounterFor(const HloInstruction* hlo);

  // Convenience function to get the IR Value emitted previously for the given
  // hlo. Make sure to call it only when you're certain a value *was* emitted -
  // if not found, this will log a fatal error.
  llvm::Value* GetEmittedValueFor(const HloInstruction* hlo);

  // Convenience function to get an IrArray representing the given hlo.
  llvm_ir::IrArray GetIrArrayForOp(const HloInstruction* hlo);

  // Augments IrArray with aliasing information.
  void AddAliasingInformationToIrArray(const HloInstruction& hlo,
                                       llvm_ir::IrArray* array) {
    alias_analysis_.AddAliasingInformationToIrArray(hlo, array);
  }

  // Convenience function to get the IR type matching the given shape.
  llvm::Type* IrShapeType(const Shape& shape);

  // Get the llvm::Value* that represents the "retval" argument of the
  // computation function being emitted by this emitter.
  llvm::Argument* GetResultArgument();

  // Get the llvm::Value* that represents the "prof_counters" argument of the
  // computation function being emitted by this emitter.
  llvm::Argument* GetProfileCountersArgument();

  // Get the xla::ExecutableRunOptions that represents the "run_options"
  // argument of the computation function being emitted by this emitter.
  llvm::Value* GetExecutableRunOptionsArgument();

  // Get the llvm::Value* that represents the "temps" argument of the
  // computation function being emitted by this emitter.
  llvm::Value* GetTempBuffersArgument();

  // Emits code that computes the address of the given temporary buffer to the
  // function. target_shape is the shape of this temporary buffer.
  // The returned Value's type is a pointer to element_type.
  llvm::Value* EmitTempBufferPointer(const BufferAllocation::Slice& slice,
                                     const Shape& target_shape);

  // Emits a function into the current module. This can be used for
  // computations embedded inside other computations, such as the
  // function that a map operation applies.
  StatusOr<llvm::Function*> EmitFunction(
      HloComputation* function,  // The function to emit.
      tensorflow::StringPiece
          function_name_suffix);  // Used for LLVM IR register names.

  // Methods that emit a function call.
  // Parameters:
  //   function - The LLVM function to call.
  //   return_shape - The return shape of the HLO computation that was used to
  //     make the function.  Not the same as the return type of the function
  //     in LLVM, since we use output parameters for the return type.
  //   element_count - number of elements to return (array form only).
  //   parameter_addresses - pointers to be passed to the function as
  //     parameters.
  //   name - used for LLVM IR register names.

  // Emits a function call, returning a scalar, often an element of a larger
  // array.  Returns a Value for the scalar element returned by the function.
  llvm::Value* EmitElementFunctionCall(
      llvm::Function* function, const Shape& return_shape,
      tensorflow::gtl::ArraySlice<llvm::Value*> parameter_addresses,
      tensorflow::StringPiece name);

  // Array function call emitter.  Stores the function's result into a supplied
  // buffer.
  // Parameters:
  //   function - The LLVM function to call.
  //   parameter_addresses - pointers to be passed to the function as
  //     parameters.
  //   return_value - pointer to a buffer where the call result is stored.

  void EmitArrayFunctionCallInto(
      llvm::Function* function,
      tensorflow::gtl::ArraySlice<llvm::Value*> parameter_addresses,
      llvm::Value* return_value, tensorflow::StringPiece name);

  // Array function call emitter.  Returns a Value for the function's return
  // value buffer address. The return value buffer is alloca'ed by this
  // function.
  llvm::Value* EmitArrayFunctionCall(
      llvm::Function* function, const Shape& return_shape, int64 element_count,
      tensorflow::gtl::ArraySlice<llvm::Value*> parameter_addresses,
      tensorflow::StringPiece name);

  // Verifies that the element types of all of the given operand instructions
  // match and are of one of the given supported types.
  Status ElementTypesSameAndSupported(
      const HloInstruction& instruction,
      tensorflow::gtl::ArraySlice<const HloInstruction*> operands,
      tensorflow::gtl::ArraySlice<PrimitiveType> supported_types);

  // Emit IR to perform a computation for every element in the given target op.
  // This produces a series of nested loops (one for each dimension of the op's
  // shape). The body of the inner-most loop is provided by the body_emitter
  // function.
  //
  // TODO(jingyue): target_op should be a `const HloInstruction*`.
  Status EmitTargetElementLoop(
      HloInstruction* target_op,
      const llvm_ir::ElementGenerator& element_generator);

  // Emits a memcpy from the source instruction's result value to the
  // destination's.  Both source and destination must have an entry in the
  // emitted_value_ table.
  Status EmitMemcpy(const HloInstruction& source,
                    const HloInstruction& destination);

  // Emit IR to compute the target address of the buffer for the given op.
  // The returned Value is a pointer to a IR type that represents the op's
  // element type.
  StatusOr<llvm::Value*> EmitTargetAddressForOp(const HloInstruction* op);

  // Structurizes "array_elements" into an MD array that represents "shape".
  // This is a recursive function, and "dimension_index" indicates the index of
  // the current dimension that the function is considering (0 means the
  // most-minor dimension).
  llvm::Constant* CreateInitializerForConstantArray(
      const std::vector<llvm::Constant*>& array_elements, const Shape& shape,
      int64 dimension_index);

  // Name of the computation entry function. This function serves as the
  // top-level "main" of the computation and will be invoked by the JIT.
  string entry_function_name_;

  // Assignment of the temporary buffers needed by the computation and their
  // shape information.
  const BufferAssignment& assignment_;

  // The LLVM module into which IR will be emitted.
  llvm::Module* module_;

  // The target architecture.
  llvm::Triple::ArchType arch_type_;

  // Used to produce unique names for generated functions.
  NameUniquer name_uniquer_;

  // Map containing all previously emitted computations.
  std::map<HloComputation*, llvm::Function*> emitted_functions_;

  // Map containing all previously emitted thread-local temporary buffers.
  std::map<std::pair<llvm::Function*, BufferAllocation::Slice>,
           llvm::AllocaInst*>
      thread_local_buffers_;

  // The following fields track the IR emission state. According to LLVM memory
  // management rules, their memory is owned by the module.
  llvm::Function* compute_function_;
  llvm::IRBuilder<> ir_builder_;

  // Maps HLOs to their index into the profile counter array.
  const std::unordered_map<const HloInstruction*, size_t>* hlo_to_profile_idx_;

  // Maps HLOs to Values emitted for them.
  std::unordered_map<const HloInstruction*, llvm::Value*> emitted_value_;

  llvm_ir::AliasAnalysis alias_analysis_;

  // This struct contains all the state needed to emit instructions for
  // profiling a computation.
  class ProfilingState {
   public:
    ProfilingState()
        : is_entry_computation_(false),
          use_rdtscp_(false),
          prof_counters_(nullptr) {}
    ProfilingState(bool is_entry_computation, bool use_rdtscp,
                   llvm::Argument* prof_counters)
        : is_entry_computation_(is_entry_computation),
          use_rdtscp_(use_rdtscp),
          prof_counters_(prof_counters) {}

    // Record the cycle counter before an HLO executes.
    void RecordCycleStart(llvm::IRBuilder<>* ir_builder, HloInstruction* hlo);
    // Record the number of cycles it took for an HLO to execute.
    void RecordCycleDelta(llvm::IRBuilder<>* ir_builder, HloInstruction* hlo,
                          llvm::Value* prof_counter);
    // Record the number of cycles it took for the entire computation to
    // execute.
    void RecordCompleteComputation(llvm::IRBuilder<>* ir_builder,
                                   llvm::Value* prof_counter);

    // Convenience function to generate a call to an intrinsic which reads the
    // CPU cycle counter.
    llvm::Value* ReadCycleCounter(llvm::IRBuilder<>* ir_builder);

    // Store the cycle counter delta to the per-HLO profile counter.
    void UpdateProfileCounter(llvm::IRBuilder<>* ir_builder,
                              llvm::Value* prof_counter, llvm::Value* cycle_end,
                              llvm::Value* cycle_start);

   private:
    // Is this IrEmitter for a top-level computation?
    bool is_entry_computation_;

    // Should we use the x86-specific rdtscp or the generic readcyclecounter
    // intrinsic?
    bool use_rdtscp_;

    // The argument which corresponds to the profile counter buffer.
    llvm::Argument* prof_counters_;

    // The first read cycle counter in the program.
    llvm::Value* first_read_cycle_start_ = nullptr;

    // The last read cycle counter in the program.
    llvm::Value* last_read_cycle_end_ = nullptr;

    // An alloca used to hold the output of the aux value returned by the rdtscp
    // intrinsic.
    llvm::Value* aux_i8ptr_ = nullptr;

    // Maps HLOs to the value the cycle counter contained right before the HLO
    // began to execute.
    std::unordered_map<const HloInstruction*, llvm::Value*> cycle_starts_;
  };

  ProfilingState profiling_state_;

  // Given a load instruction and a shape or buffer size, annotate the load's
  // result with the alignment required by the shape or size.
  void AttachAlignmentMetadataForLoad(llvm::LoadInst* load, const Shape& shape);
  void AttachAlignmentMetadataForLoad(llvm::LoadInst* load, int64 buffer_size);

  // Given a load instruction and a shape or buffer size, annotate the load's
  // result with the dereferenceable bytes required by the shape / buffer size.
  void AttachDereferenceableMetadataForLoad(llvm::LoadInst* load,
                                            const Shape& shape);
  void AttachDereferenceableMetadataForLoad(llvm::LoadInst* load,
                                            int64 buffer_size);

  // Calculate the alignment of a buffer allocated for a given shape.
  int MinimumAlignmentForShape(const Shape& shape);

  // Calculate the alignment of a buffer allocated for a given primitive type.
  int MinimumAlignmentForPrimitiveType(PrimitiveType primitive_type);

  // Calculate the alignment of a buffer with a particular size.
  int MinimumAlignmentForBufferSize(int64 buffer_size);

  // Returns the number of bytes within the shape.
  int64 ByteSizeOf(const Shape& shape) const;

  const HloModuleConfig& hlo_module_config_;

  TF_DISALLOW_COPY_AND_ASSIGN(IrEmitter);
};

}  // namespace cpu
}  // namespace xla

#endif  // TENSORFLOW_COMPILER_XLA_SERVICE_CPU_IR_EMITTER_H_
