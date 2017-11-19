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

#include "llvm/ADT/Triple.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/Value.h"
#include "llvm/Target/TargetMachine.h"
#include "tensorflow/compiler/xla/service/buffer_assignment.h"
#include "tensorflow/compiler/xla/service/cpu/external_constant_pool.h"
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
#include "tensorflow/core/lib/gtl/flatmap.h"
#include "tensorflow/core/platform/macros.h"
#include "tensorflow/core/platform/types.h"

namespace xla {
namespace cpu {

// Wraps an llvm::TargetMachine and parses out some information that feeds into
// code LLVM IR generation decisions.
//
// Ideally we'd be able to use llvm::TargetTransformInfo here (since its
// interface is pretty much a perfect fit for our use case), but obtaining an
// instance of llvm::TargetTransformInfo outside an LLVM pass pipeline without
// super-ugly hacks is difficult.
//
// TODO(b/66049221): See if the LLVM community will be receptive to exposing an
// API that lets us directly create and use llvm::TargetTransformInfo instances
// outside of a pass manager.
class TargetMachineFeatures {
 public:
  TargetMachineFeatures(llvm::TargetMachine* target_machine)
      : target_machine_(target_machine) {}

  // Return the vectorization factor, which is the number of bytes of data
  // explicitly vectorized routines will try to process at once.
  int vectorization_factor_in_bytes() const {
    // Ideally this should be a function of the cache line size (which we can
    // get from llvm::TargetTransformInfo::getCacheLineSize) of the target
    // machine.  Guess a value of 128 bytes for now.
    return 128;
  }

  // Return the size of the largest register size in bytes.  We need to pass in
  // "function" since llvm functions can contain annotations for specializing
  // them to specific micro-architectures (though currently XLA does not use
  // this functionality).
  //
  // Ideally we should have been able to use
  // llvm::TargetTransformInfo::getRegisterBitWidth(true) here.
  unsigned largest_register_size_in_bytes(llvm::Function* function);

 private:
  unsigned largest_register_size_in_bytes_impl(llvm::Function* function) const;

  tensorflow::gtl::FlatMap<llvm::Function*, int>
      largest_register_size_in_bytes_;
  llvm::TargetMachine* target_machine_;
};

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
  // entry_computation_profile_idx: the index in the profiling array
  //                                for the entry computation.
  // external_constant_pool: if non-null, points to an ExternalConstantPool
  //                         instance into which the Ir emitter can spill
  //                         constants.
  IrEmitter(
      const HloModule& hlo_module, const BufferAssignment& assignment,
      llvm::Module* llvm_module,
      std::unordered_map<const HloInstruction*, size_t> hlo_to_profile_idx,
      tensorflow::gtl::optional<size_t> entry_computation_profile_idx,
      llvm::TargetMachine* target_machine,
      ExternalConstantPool* external_constant_pool);
  ~IrEmitter() override;

  // Emit and return the given HLO computation as an LLVM IR
  // function.
  //
  // function_name_prefix is the desired name of the function. If the name is
  // not unique among already emitted functions then a suffix is appended to
  // make the name unique.
  //
  // 'is_top_level_computation' has the following meanings for each CPU backend:
  // *) sequential: indicates that this is the entry computation of the HLO
  //    module.
  // *) parallel: indices that this is the callee of a kCall HLO in the entry
  //    computation of the HLO module.
  //
  // If 'instruction_order' is not NULL, then the HLO instructions are emitted
  // in the given order.  In this case, 'instruction_order' must be a
  // topological sort of the set of nodes accessible from the root of the
  // computation.
  StatusOr<llvm::Function*> EmitComputation(
      HloComputation* computation, const string& function_name_prefix,
      bool is_top_level_computation,
      std::vector<const HloInstruction*>* instruction_order);

  llvm::IRBuilder<>* ir_builder() { return &ir_builder_; }

  // Emits a call to `computation` with scalar arguments `arguments`.
  StatusOr<llvm::Value*> EmitScalarCall(
      PrimitiveType return_type, HloComputation* computation,
      const std::vector<llvm::Value*>& arguments, tensorflow::StringPiece name);

 protected:
  //
  // The following methods implement the DfsHloVisitor interface.
  //
  // Default action which emits code for most operations. Operations which are
  // special in some way are handled explicitly in HandleFoo methods.
  Status DefaultAction(HloInstruction* hlo) override;

  Status HandleBitcast(HloInstruction* bitcast) override;
  Status HandleConstant(HloInstruction* constant) override;
  Status HandleCopy(HloInstruction* copy) override;
  Status HandleGetTupleElement(HloInstruction* get_tuple_element) override;
  Status HandleSelect(HloInstruction* select) override;
  Status HandleDot(HloInstruction* dot) override;
  Status HandleConvolution(HloInstruction* convolution) override;
  Status HandleBatchNormTraining(HloInstruction* batch_norm_training) override;
  Status HandleBatchNormGrad(HloInstruction* batch_norm_grad) override;
  Status HandleCrossReplicaSum(HloInstruction* crs) override;
  Status HandleInfeed(HloInstruction* infeed) override;
  Status HandleOutfeed(HloInstruction* outfeed) override;
  Status HandleSort(HloInstruction* sort) override;
  Status HandleParameter(HloInstruction* parameter) override;
  Status HandleReduce(HloInstruction* reduce) override;
  Status HandleReduceWindow(HloInstruction* reduce_window) override;
  Status HandleSelectAndScatter(HloInstruction* select_and_scatter) override;
  Status HandleSend(HloInstruction* send) override;
  Status HandleSendDone(HloInstruction* send_done) override;
  Status HandleSlice(HloInstruction* slice) override;
  Status HandleDynamicSlice(HloInstruction* dynamic_slice) override;
  Status HandleDynamicUpdateSlice(
      HloInstruction* dynamic_update_slice) override;
  Status HandleRecv(HloInstruction* recv) override;
  Status HandleRecvDone(HloInstruction* recv_done) override;
  Status HandlePad(HloInstruction* pad) override;
  Status HandleTuple(HloInstruction* tuple) override;
  Status HandleMap(HloInstruction* map) override;
  Status HandleFusion(HloInstruction* fusion) override;
  Status HandleCall(HloInstruction* call) override;
  Status HandleCustomCall(HloInstruction* custom_call) override;
  Status HandleWhile(HloInstruction* xla_while) override;
  Status HandleConcatenate(HloInstruction* concatenate) override;
  Status FinishVisit(HloInstruction* root) override;

  Status Preprocess(HloInstruction* hlo) override;
  Status Postprocess(HloInstruction* hlo) override;

 private:
  // Private helper to initialize an IR function for the computation.
  void InitializeIrFunction(const string& function_name);

  // Convenience function to generate a GEP into the profile counter parameter
  // which would correspond to the index for a given HLO.
  llvm::Value* GetProfileCounterFor(const HloInstruction& hlo);

  // Convenience function to generate a GEP into the profile counter parameter
  // corresponding to the index for the entry computation.  Returns nullptr if
  // profiling the entry computation is disabled.
  llvm::Value* GetProfileCounterForEntryComputation();

  // Gets the IR Value emitted previously for the given hlo.
  //
  // Prefer calling GetIrArrayFor if the value you're reading is a buffer,
  // because GetIrArrayFor annotates buffer's loads/stores with noalias
  // metadata.
  //
  // Make sure to call this only when you're certain a value *was* emitted - if
  // not found, this will log a fatal error.
  llvm::Value* GetEmittedValueFor(const HloInstruction* hlo);

  // Gets an IrArray representing the given hlo.
  llvm_ir::IrArray GetIrArrayFor(const HloInstruction* hlo);

  // Gets a list of IrArrays, one for each of hlo's operands.
  std::vector<llvm_ir::IrArray> GetIrArraysForOperandsOf(
      const HloInstruction* hlo);

  // Augments IrArray with aliasing information.
  void AddAliasingInformationToIrArray(const HloInstruction& hlo,
                                       llvm_ir::IrArray* array) {
    alias_analysis_.AddAliasingInformationToIrArray(hlo, array);
  }

  // Convenience function to get the IR type matching the given shape.
  llvm::Type* IrShapeType(const Shape& shape);

  // Returns an array of compute function parameter types.
  std::vector<llvm::Type*> GetComputeFunctionParams();

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

  // Emit ir to read and return the ir value for the dynamic loop bound at
  // 'offset' from the "dynamic_loop_bounds" argument of the computation
  // function being emitted by this emitter.
  llvm::Value* GetDynamicLoopBound(const int64 offset);

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
      llvm::Value* return_value_buffer, tensorflow::StringPiece name);

  // Array function call emitter.  Returns a Value for the function's return
  // value buffer address. The return value buffer is alloca'ed by this
  // function.
  llvm::Value* EmitArrayFunctionCall(
      llvm::Function* function, const Shape& return_shape, int64 element_count,
      tensorflow::gtl::ArraySlice<llvm::Value*> parameter_addresses,
      tensorflow::StringPiece name);

  // Returns an array of compute function call arguments.
  std::vector<llvm::Value*> GetArrayFunctionCallArguments(
      tensorflow::gtl::ArraySlice<llvm::Value*> parameter_addresses,
      llvm::Value* return_value_buffer, tensorflow::StringPiece name);

  // Emits a call to a runtime fork/join function which dispatches parallel
  // calls to 'parallel_function' (and joins threads before returning).
  Status EmitParallelForkJoin(
      tensorflow::gtl::ArraySlice<llvm::Value*> parameter_addresses,
      llvm::Value* output_address, HloComputation* computation,
      llvm::Function* parallel_function);

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
  // desc is an optional human-readable string that's added to the loop name in
  // IR.  Regardless of whether desc is provided, target_op->name() is included
  // in the loop name.
  //
  // TODO(jingyue): target_op should be a `const HloInstruction*`.
  Status EmitTargetElementLoop(
      HloInstruction* target_op,
      const llvm_ir::ElementGenerator& element_generator);
  Status EmitTargetElementLoop(
      HloInstruction* target_op, tensorflow::StringPiece desc,
      const llvm_ir::ElementGenerator& element_generator);

  // Emit IR to perform a computation for every element in a partition/slice of
  // 'target_shape'. The loop bounds for the outer-dimension partitions are
  // passed into the compute function as a runtime argument (accessible from
  // GetDynamicLoopBound).
  Status EmitParallelTargetElementLoop(
      const Shape& target_shape,
      const llvm_ir::ElementGenerator& element_generator,
      tensorflow::StringPiece loop_name, llvm_ir::IrArray* target_array);

  // Emits a memcpy from the source instruction's result value to the
  // destination's.  Both source and destination must have an entry in the
  // emitted_value_ table.
  Status EmitMemcpy(const HloInstruction& source,
                    const HloInstruction& destination);

  // Emits IR to compute the target address of the buffer for the given op.
  // After calling this function, you can get a pointer to this buffer by
  // calling GetIrArrayForOp or GetEmittedValueFor.
  Status EmitTargetAddressForOp(const HloInstruction* op);

  // Structurizes "array_elements" into an MD array that represents "shape".
  // This is a recursive function, and "dimension_index" indicates the index of
  // the current dimension that the function is considering (0 means the
  // most-minor dimension).
  llvm::Constant* CreateInitializerForConstantArray(
      const std::vector<llvm::Constant*>& array_elements, const Shape& shape,
      int64 dimension_index);

  // Tries to codegen a reduction operation using vectorized instructions.
  // Returns true if successful, and false on failure.  On failure, sets
  // "failure_reason" to a string describing why it could not vectorize the
  // reduction.
  //
  // TODO(sanjoy): Some of the things we do here can be abstracted out into
  // concepts that generalize over other vectorizable operations.  We should
  // consider pulling out these abstractions into a VectorizingIrEmitter or
  // something similar.
  StatusOr<bool> EmitVectorizedReduce(
      HloInstruction* reduce, HloInstruction* arg, HloInstruction* init_value,
      tensorflow::gtl::ArraySlice<int64> dimensions, HloComputation* function,
      string* failure_reason);

  // We'd like to keep one or two one cache-line's worth of data in registers
  // without generating IR with illegal (e.g. excessively large or
  // non-power-of-two) vector types.  We do this by introducing a layer of
  // abstraction: we introduce a high level vector-like concept called a
  // "sharded vector" that models data paralleism, and is mapped to a sequence
  // scalar and vector llvm::Value s.
  //
  // For example, we can represent 29 f32 elements by a sharded vector mapped to
  // a sequence of LLVM values of types [<16 x f32>, <8 x f32>, <4 x f32>, f32].
  // Note that the last element is scalar.
  //
  // There is no requirement on the ordering or the uniqueness of the elements
  // mapped to sharded vectors -- we allow repeated elements, and we allow
  // elements to appear in any order.
  using ShardedVector = std::vector<llvm::Value*>;

  // A sharded vector type is the element-wise llvm::Type's of some
  // ShardedVector.
  using ShardedVectorType = std::vector<llvm::Type*>;

  // Create a sharded vector type corresponding to a "element_count" long
  // sequence of "element_type" values.
  ShardedVectorType CreateShardedVectorType(PrimitiveType element_type,
                                            unsigned element_count);

  // Emit LLVM IR to store the sharded vector "value_to_store" to
  // "store_address".
  void EmitShardedVectorStore(llvm::Value* store_address,
                              const ShardedVector& value_to_store,
                              const int alignment,
                              const llvm_ir::IrArray& containing_array);

  using ReductionGenerator = std ::function<llvm::Value*(
      llvm::IRBuilder<>*, llvm::Value*, llvm::Value*)>;

  // Tries to match the reduction function "function" to a known reduction
  // pattern.  Returns a non-null ReductionGenerator on a successful match,
  // which can be used to generate the LLVM IR corresponding to said reduction.
  // On failure, this stores a reason string into "failure_reason".
  ReductionGenerator MatchReductionGenerator(HloComputation* function,
                                             string* failure_reason) const;

  // Emits the inner loop nest that runs the reduction.  Helper function for
  // EmitVectorizedReduce.
  StatusOr<ShardedVector> EmitInnerLoopForVectorizedReduction(
      const ReductionGenerator& reduction_generator,
      const llvm_ir::IrArray::Index& output_index,
      const ShardedVectorType& accumulator_type, HloInstruction* init_value,
      HloInstruction* arg, tensorflow::gtl::ArraySlice<int64> dimensions,
      unsigned element_alignment);

  // Tries to emit a fast concatenate operation using memcpy.  Returns true if
  // successful, and false on failure.  On failure, sets "failure_reason" to a
  // string describing why it could not emit a fast concatenate.
  StatusOr<bool> EmitFastConcatenate(
      HloInstruction* concatenate,
      tensorflow::gtl::ArraySlice<HloInstruction*> operands,
      string* failure_reason);

  // Emits LLVM IR to transfer "element_count" elements of type "primitive_type"
  // from the address "source" to the address "target".
  void EmitTransferElements(llvm::Value* target, llvm::Value* source,
                            int64 element_count, PrimitiveType primitive_type,
                            const llvm_ir::IrArray& target_array,
                            const llvm_ir::IrArray& source_array);

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
  std::unordered_map<const HloInstruction*, size_t> hlo_to_profile_idx_;
  const tensorflow::gtl::optional<size_t> entry_computation_profile_idx_;

  // Maps HLOs to Values emitted for them.
  std::unordered_map<const HloInstruction*, llvm::Value*> emitted_value_;

  llvm_ir::AliasAnalysis alias_analysis_;

  // The number of root instruction outer dimensions used in parallel loop
  // emission (EmitParallelTargetElementLoop).
  int64 num_dynamic_loop_bounds_ = 0;

  // Returns whether the given instruction should be emitted as a parallel loop.
  bool ShouldEmitParallelLoopFor(const HloInstruction& op) const {
    // Emit parallel loop for root instruction if dynamic outer-dimension loop
    // bounds were specified.
    return num_dynamic_loop_bounds_ > 0 &&
           op.parent()->root_instruction() == &op;
  }

  // This struct contains all the state needed to emit instructions for
  // profiling a computation.
  class ProfilingState {
   public:
    ProfilingState()
        : is_top_level_computation_(false),
          use_rdtscp_(false),
          prof_counters_(nullptr) {}
    ProfilingState(bool is_top_level_computation, bool use_rdtscp,
                   llvm::Argument* prof_counters)
        : is_top_level_computation_(is_top_level_computation),
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
    bool is_top_level_computation_;

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

  enum class XfeedKind {
    kInfeed,
    kOutfeed,
  };

  // Emit IR to transfer between a {infeed,outfeed} buffer and an in-program
  // address.
  Status EmitXfeedTransfer(XfeedKind kind, const Shape& shape,
                           llvm::Value* program_buffer_address);

  const HloModuleConfig& hlo_module_config_;

  const bool parallel_cpu_backend_;

  bool is_top_level_computation_;

  TargetMachineFeatures target_machine_features_;

  int64 external_global_constant_counter_ = 0;
  ExternalConstantPool* external_constant_pool_;

  TF_DISALLOW_COPY_AND_ASSIGN(IrEmitter);
};

}  // namespace cpu
}  // namespace xla

#endif  // TENSORFLOW_COMPILER_XLA_SERVICE_CPU_IR_EMITTER_H_
