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

#include <functional>
#include <map>
#include <memory>
#include <ostream>
#include <string>
#include <utility>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/Value.h"
#include "llvm/TargetParser/Triple.h"
#include "mlir/IR/MLIRContext.h"  // from @llvm-project
#include "tensorflow/compiler/xla/hlo/ir/dfs_hlo_visitor_with_default.h"
#include "tensorflow/compiler/xla/hlo/ir/hlo_computation.h"
#include "tensorflow/compiler/xla/service/buffer_assignment.h"
#include "tensorflow/compiler/xla/service/cpu/ir_function.h"
#include "tensorflow/compiler/xla/service/cpu/target_machine_features.h"
#include "tensorflow/compiler/xla/service/hlo_module_config.h"
#include "tensorflow/compiler/xla/service/llvm_ir/alias_analysis.h"
#include "tensorflow/compiler/xla/service/llvm_ir/fused_ir_emitter.h"
#include "tensorflow/compiler/xla/service/llvm_ir/ir_array.h"
#include "tensorflow/compiler/xla/service/llvm_ir/ir_builder_mixin.h"
#include "tensorflow/compiler/xla/service/llvm_ir/loop_emitter.h"
#include "tensorflow/compiler/xla/service/name_uniquer.h"
#include "tensorflow/compiler/xla/statusor.h"
#include "tensorflow/compiler/xla/xla_data.pb.h"

namespace xla {
namespace cpu {
// This class is the top-level API for the XLA HLO --> LLVM IR compiler.  It
// implements the DfsHloVisitor interface and emits HLO computations as LLVM IR
// functions.
class IrEmitter : public DfsHloVisitorWithDefault,
                  public IrBuilderMixin<IrEmitter> {
  friend class CpuElementalIrEmitter;

 public:
  using GeneratorForOperandIrArrays =
      std::function<std::vector<llvm_ir::IrArray>()>;

  // Create a new LLVM IR emitter.
  //
  // hlo_module: the HLO module we are emitting IR for.
  // assignment: a BufferAssignment from which we know which buffers are used by
  //             the HLO nodes.
  // mlir_context: the MLIR context used for IR emission.
  // llvm_module: the LLVM module to emit IR into. It's built using the LLVM
  //              context inside of mlir_context.
  // instruction_to_profile_idx: the mapping from HLO instructions to their
  //              index in the profiling array.
  // computation_to_profile_idx: the mapping from HLO computations to their
  //              index in the profiling array.
  // computation_transitively_contains_custom_call: the mapping from HLO
  //   computations to whether or not they transitively contain a custom-call
  //   instruction. All computations in the module must have a key in this
  //   map.
  // emit_code_for_msan: whether emitted code should be compatible with msan.
  IrEmitter(mlir::MLIRContext* mlir_context, const HloModule& hlo_module,
            const BufferAssignment& assignment, llvm::Module* llvm_module,
            absl::flat_hash_map<const HloInstruction*, int64_t>
                instruction_to_profile_idx,
            absl::flat_hash_map<const HloComputation*, int64_t>
                computation_to_profile_idx,
            absl::flat_hash_map<const HloComputation*, bool>
                computation_transitively_contains_custom_call,
            const TargetMachineFeatures* target_machine,
            bool emit_code_for_msan);
  ~IrEmitter() override = default;

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
  //
  // If 'allow_reassociation' is true, the fast-math reassociation flag will
  // be enabled in the function's body. This is used when emitting reducers.
  StatusOr<llvm::Function*> EmitComputation(
      HloComputation* computation, const std::string& function_name_prefix,
      bool is_top_level_computation,
      absl::Span<HloInstruction* const> instruction_order,
      bool allow_reassociation);

  llvm::IRBuilder<>* b() { return &b_; }

  // builder() is for IrBuilderMixin.
  llvm::IRBuilder<>* builder() { return &b_; }

  // Emit an LLVM global variable for every constant buffer allocation.
  Status EmitConstantGlobals();

 protected:
  //
  // The following methods implement the DfsHloVisitor interface.
  //
  // Default action which emits code for most operations. Operations which are
  // special in some way are handled explicitly in HandleFoo methods.
  Status DefaultAction(HloInstruction* hlo) override;

  Status HandleAllToAll(HloInstruction* instruction) override;
  Status HandleBitcast(HloInstruction* bitcast) override;
  Status HandleConstant(HloInstruction* constant) override;
  Status HandleCopy(HloInstruction* copy) override;
  Status HandleGetTupleElement(HloInstruction* get_tuple_element) override;
  Status HandleSelect(HloInstruction* select) override;
  Status HandleDot(HloInstruction* dot) override;
  Status HandleConvolution(HloInstruction* convolution) override;
  Status HandleFft(HloInstruction* fft) override;
  Status HandleAllReduce(HloInstruction* crs) override;
  Status HandleCollectivePermute(HloInstruction* crs) override;
  Status HandleInfeed(HloInstruction* instruction) override;
  Status HandleOutfeed(HloInstruction* outfeed) override;
  Status HandleSort(HloInstruction* hlo) override;
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
  Status HandleFusion(HloInstruction* fusion) override;
  Status HandleCall(HloInstruction* call) override;
  Status HandleCustomCall(HloInstruction* custom_call) override;
  Status HandleWhile(HloInstruction* xla_while) override;
  Status HandleConcatenate(HloInstruction* concatenate) override;
  Status HandleConditional(HloInstruction* conditional) override;
  Status HandleScatter(HloInstruction* scatter) override;
  Status HandleAfterAll(HloInstruction* after_all) override;
  Status HandleAddDependency(HloInstruction* add_dependency) override;
  Status HandlePartitionId(HloInstruction* hlo) override;
  Status HandleReplicaId(HloInstruction* hlo) override;
  Status HandleRng(HloInstruction* rng) override;
  Status HandleRngGetAndUpdateState(HloInstruction* rng_state) override;
  Status FinishVisit(HloInstruction* root) override;

  Status Preprocess(HloInstruction* hlo) override;
  Status Postprocess(HloInstruction* hlo) override;

  // A convenient helper for calling BufferAssignment::GetUniqueSlice.
  BufferAllocation::Slice GetAllocationSlice(
      const HloInstruction& hlo, const ShapeIndex& index = {}) const {
    return assignment_.GetUniqueSlice(&hlo, index).value();
  }

 private:
  Status HandleSliceToDynamic(HloInstruction* hlo);
  Status HandlePadToStatic(HloInstruction* hlo);
  Status HandleTopK(HloInstruction* hlo);
  Status HandleAllReduceSingleReplica(HloInstruction* crs);
  Status HandleAllReduceMultipleReplica(HloInstruction* crs);

  // Private helper to initialize an IR function for the computation.
  void InitializeIrFunction(const std::string& function_name);

  // Emits the copying epilogue for the function,
  // where it copies the returned value to the reserved alloca.
  // This is only necessary for thread-local functions.
  // Note that since the call graph is flattened, if the same function is
  // called in both thread-local and non-thread-local it would be codegen'd
  // twice, and we would know whether it's thread-local at codegen time.
  void EmitThreadLocalFunctionEpilogue(HloComputation* computation);

  // Convenience functions to generate a GEP into the profile counter parameter
  // which would correspond to the index for a given HLO instruction or
  // computation.
  llvm::Value* GetProfileCounterFor(const HloInstruction& instruction);
  llvm::Value* GetProfileCounterFor(const HloComputation& computation);

  // Helper function template for the implementation of the above two functions.
  template <typename T>
  llvm::Value* GetProfileCounterCommon(
      const T& hlo,
      const absl::flat_hash_map<const T*, int64_t>& profile_index_map);

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

  // Bind all argument IrArrays of `fusion` to `fused_emitter`.
  void BindFusionArguments(const HloInstruction* fusion,
                           FusedIrEmitter* fused_emitter);

  // Augments IrArray with aliasing information.
  void AddAliasingInformationToIrArray(const HloInstruction& hlo,
                                       llvm_ir::IrArray* array) {
    alias_analysis_.AddAliasingInformationToIrArray(hlo, array);
  }

  // Convenience function to get the IR type matching the given shape.
  llvm::Type* IrShapeType(const Shape& shape);

  // Get the llvm::Value* that represents the "prof_counters" argument of the
  // computation function being emitted by this emitter.
  llvm::Value* GetProfileCountersArgument();

  // Get the llvm::Value* that represents the "status" argument of the
  // computation function being emitted by this emitter.
  llvm::Value* GetStatusArgument();

  // Get the xla::ExecutableRunOptions that represents the "run_options"
  // argument of the computation function being emitted by this emitter.
  llvm::Value* GetExecutableRunOptionsArgument();

  // Get the llvm::Value* that represents the "buffer_table" argument of the
  // computation function being emitted by this emitter.
  llvm::Value* GetBufferTableArgument();

  // Get the llvm::BasicBlock that contains the return instruction.
  llvm::BasicBlock* GetReturnBlock();

  // Emits code to check the state of the status object being threaded through
  // each computation and return early if it's in an error state.
  void EmitEarlyReturnIfErrorStatus();

  // Helper for EmitBufferPointer.
  llvm::Value* EmitGlobalBufferPointer(const BufferAllocation::Slice& slice,
                                       const Shape& target_shape);

  // Helper for EmitBufferPointer.
  llvm::Value* EmitThreadLocalBufferPointer(
      const BufferAllocation::Slice& slice, const Shape& target_shape);

  // Emits code that computes the address of the given buffer allocation slice.
  llvm::Value* EmitBufferPointer(const BufferAllocation::Slice& slice,
                                 const Shape& target_shape);

  // Emits a call to a thread local function (e.g. to the computation nested
  // within a reduce or a map).  Thread local callees (by definition) only write
  // to and read from thread local allocations.
  // Supports only functions returning scalars or tuples of scalars.
  //
  // `parameters` holds the *scalar values* that need to be passed to the
  // callee.  The return value is the scalar returned by the callee.
  std::vector<llvm::Value*> EmitThreadLocalCall(
      const HloComputation& callee, absl::Span<llvm::Value* const> parameters,
      absl::string_view name, bool is_reducer);

  // Similar to EmitThreadLocal, yet assumes that the function returns a scalar.
  llvm::Value* EmitScalarReturningThreadLocalCall(
      const HloComputation& callee, absl::Span<llvm::Value* const> parameters,
      absl::string_view name);

  // Emits a call to a "global" function (e.g. to the computation nested within
  // a kWhile or a kCall).  Buffer assignment unabiguously assigns buffers to
  // the parameters and return values for these computations so there is no need
  // to explicitly pass parameters or return results.
  void EmitGlobalCall(const HloComputation& callee, absl::string_view name);

  // Returns the buffer to which a global call to `callee` would have written
  // its result.
  llvm::Value* GetBufferForGlobalCallReturnValue(const HloComputation& callee);

  // Verifies that the element types of all of the given operand instructions
  // match and are of one of the given supported types.
  Status ElementTypesSameAndSupported(
      const HloInstruction& instruction,
      absl::Span<const HloInstruction* const> operands,
      absl::Span<const PrimitiveType> supported_types);

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
      HloInstruction* target_op, absl::string_view desc,
      const llvm_ir::ElementGenerator& element_generator);

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
      int64_t dimension_index);

  // Tries to codegen a reduction operation using vectorized instructions.
  // Returns true if successful, and false on failure.  On failure, sets
  // "failure_reason" to a string describing why it could not vectorize the
  // reduction.
  //
  // TODO(sanjoy): Some of the things we do here can be abstracted out into
  // concepts that generalize over other vectorizable operations.  We should
  // consider pulling out these abstractions into a VectorizingIrEmitter or
  // something similar.
  StatusOr<bool> EmitVectorizedReduce(HloInstruction* reduce,
                                      HloInstruction* arg,
                                      HloInstruction* init_value,
                                      absl::Span<const int64_t> dimensions,
                                      HloComputation* function,
                                      std::string* failure_reason);

  // We'd like to keep one or two one cache-line's worth of data in registers
  // without generating IR with illegal (e.g. excessively large or
  // non-power-of-two) vector types.  We do this by introducing a layer of
  // abstraction: we introduce a high level vector-like concept called a
  // "sharded vector" that models data parallelism, and is mapped to a sequence
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
                              llvm::Align alignment,
                              const llvm_ir::IrArray& containing_array);

  using ReductionGenerator = std ::function<llvm::Value*(
      llvm::IRBuilder<>*, llvm::Value*, llvm::Value*)>;

  // Tries to match the reduction function "function" to a known reduction
  // pattern.  Returns a non-null ReductionGenerator on a successful match,
  // which can be used to generate the LLVM IR corresponding to said reduction.
  // On failure, this stores a reason string into "failure_reason".
  ReductionGenerator MatchReductionGenerator(HloComputation* function,
                                             std::string* failure_reason) const;

  // Emits the inner loop nest that runs the reduction.  Helper function for
  // EmitVectorizedReduce.
  StatusOr<ShardedVector> EmitInnerLoopForVectorizedReduction(
      const ReductionGenerator& reduction_generator,
      const llvm_ir::IrArray::Index& output_index,
      const ShardedVectorType& accumulator_type, HloInstruction* init_value,
      HloInstruction* arg, absl::Span<const int64_t> dimensions,
      llvm::Align element_alignment);

  // Tries to emit a fast concatenate operation using memcpy.  Returns true if
  // successful, and false on failure.  On failure, sets "failure_reason" to a
  // string describing why it could not emit a fast concatenate.
  StatusOr<bool> EmitFastConcatenate(HloInstruction* concatenate,
                                     absl::Span<HloInstruction* const> operands,
                                     std::string* failure_reason);

  // Emits LLVM IR to transfer "element_count" elements of type "primitive_type"
  // from the address "source" to the address "target".
  void EmitTransferElements(llvm::Value* target, llvm::Value* source,
                            int64_t element_count, PrimitiveType primitive_type,
                            const llvm_ir::IrArray& target_array,
                            const llvm_ir::IrArray& source_array);

  // Emits printing during the execution.
  llvm::Value* EmitPrintf(absl::string_view fmt,
                          absl::Span<llvm::Value* const> arguments);
  llvm::Value* EmitPrintfToStderr(absl::string_view fmt,
                                  absl::Span<llvm::Value* const> arguments);

  // Emits a call to a non-variadic function `func_name` with arguments
  // `arguments` assuming C calling convention.
  llvm::Value* EmitCallToFunc(
      std::string func_name, const std::vector<llvm::Value*>& arguments,
      llvm::Type* return_type, bool does_not_throw = true,
      bool only_accesses_arg_memory = false,
      bool only_accesses_inaccessible_mem_or_arg_mem = false);

  // Assignment of the buffers needed by the computation and their shape
  // information.
  const BufferAssignment& assignment_;

  // The LLVM module into which IR will be emitted.
  llvm::Module* module_;

  // The target architecture.
  llvm::Triple::ArchType arch_type_;

  // Used to produce unique names for generated functions.
  NameUniquer name_uniquer_;

  struct ComputationToEmit {
    const HloComputation* computation;
    bool allow_reassociation;

    bool operator==(const ComputationToEmit& other) const {
      return computation == other.computation &&
             allow_reassociation == other.allow_reassociation;
    }

    template <typename H>
    friend H AbslHashValue(H h, const ComputationToEmit& c) {
      return H::combine(std::move(h), c.computation, c.allow_reassociation);
    }
    friend std::ostream& operator<<(std::ostream& os,
                                    const ComputationToEmit& c) {
      return os << c.computation->name() << ", " << c.allow_reassociation;
    }
  };

  // Map containing all previously emitted computations.
  absl::flat_hash_map<ComputationToEmit, llvm::Function*> emitted_functions_;

  // Map containing all previously emitted thread-local temporary buffers.
  std::map<std::pair<llvm::Function*, BufferAllocation::Slice>, llvm::Value*>
      thread_local_buffers_;

  // The following fields track the IR emission state. According to LLVM memory
  // management rules, their memory is owned by the module (Note that IrFunction
  // creates the encapsulated llvm::Function s.t. it is added to the llvm
  // module's function list).
  std::unique_ptr<IrFunction> compute_function_;
  llvm::IRBuilder<> b_;
  mlir::MLIRContext* mlir_context_;
  bool allow_reassociation_;

  // The buffer allocation slice for the root of the computation being compiled.
  // Only relevant for thread local computations.
  BufferAllocation::Slice computation_root_allocation_;

  // Maps the buffer allocation slices for the parameters to the computation
  // being compiled to their parameter numbers.  Only relevant for thread local
  // computations.
  absl::flat_hash_map<BufferAllocation::Index, int64_t>
      computation_parameter_allocations_;

  // Maps HLO instructions to their index into the profile counter array.
  const absl::flat_hash_map<const HloInstruction*, int64_t>
      instruction_to_profile_idx_;

  // Maps HLO computations to their index into the profile counter array.
  const absl::flat_hash_map<const HloComputation*, int64_t>
      computation_to_profile_idx_;

  // Maps HLO computations to whether they contain a custom-call instruction
  // (either directly, or transitively by e.g. calling another computation that
  // does).
  const absl::flat_hash_map<const HloComputation*, bool>
      computation_transitively_contains_custom_call_;

  // Accessor for the custom-call mapping that enforces the precondition that
  // all computations must have a key in the map.
  bool ComputationTransitivelyContainsCustomCall(
      const HloComputation* computation) const {
    auto it = computation_transitively_contains_custom_call_.find(computation);
    CHECK(it != computation_transitively_contains_custom_call_.cend())
        << "Must provide 'contains CustomCall' annotation for all computations "
           "in the module";
    return it->second;
  }

  // Maps HLOs to Values emitted for them.
  absl::flat_hash_map<const HloInstruction*, llvm::Value*> emitted_value_;

  llvm_ir::AliasAnalysis alias_analysis_;

  // The number of outer dimensions of the root instruction's shape that
  // will be partitioned when emitting parallel loops. (See
  // ParallelLoopEmitter).
  int64_t num_dynamic_loop_bounds_ = 0;

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
    ProfilingState() : use_rdtscp_(false) {}
    explicit ProfilingState(bool use_rdtscp) : use_rdtscp_(use_rdtscp) {}

    // Record the cycle counter before an HLO executes.
    void RecordCycleStart(llvm::IRBuilder<>* b, HloInstruction* hlo);
    // Record the number of cycles it took for an HLO to execute.
    void RecordCycleDelta(llvm::IRBuilder<>* b, HloInstruction* hlo,
                          llvm::Value* prof_counter);
    // Record the number of cycles it took for the entire computation to
    // execute.
    void RecordCompleteComputation(llvm::IRBuilder<>* b,
                                   llvm::Value* prof_counter);

    // Convenience function to generate a call to an intrinsic which reads the
    // CPU cycle counter.
    llvm::Value* ReadCycleCounter(llvm::IRBuilder<>* b);

    // Store the cycle counter delta to the per-HLO profile counter.
    void UpdateProfileCounter(llvm::IRBuilder<>* b, llvm::Value* prof_counter,
                              llvm::Value* cycle_end, llvm::Value* cycle_start);

   private:
    // Should we use the x86-specific rdtscp or the generic readcyclecounter
    // intrinsic?
    bool use_rdtscp_;

    // The first read cycle counter in the program.
    llvm::Value* first_read_cycle_start_ = nullptr;

    // The last read cycle counter in the program.
    llvm::Value* last_read_cycle_end_ = nullptr;

    // Maps HLOs to the value the cycle counter contained right before the HLO
    // began to execute.
    absl::flat_hash_map<const HloInstruction*, llvm::Value*> cycle_starts_;
  };

  ProfilingState profiling_state_;

  class TracingState {
   public:
    TracingState() : enabled_(false) {}
    void set_enabled(bool value) { enabled_ = value; }
    void EmitTracingStart(llvm::IRBuilder<>* b, HloInstruction* hlo,
                          llvm::Value* run_options);
    void EmitTracingEnd(llvm::IRBuilder<>* b, HloInstruction* hlo,
                        llvm::Value* run_options);

   private:
    bool enabled_;
    // Maps from HLO to the activity id returned by xprof::TraceMe.
    absl::flat_hash_map<const HloInstruction*, llvm::Value*> activity_ids_;
  };
  TracingState tracing_state_;

  // Given a load instruction and a shape or buffer size, annotate the load's
  // result with the alignment required by the shape or size.
  void AttachAlignmentMetadataForLoad(llvm::LoadInst* load, const Shape& shape);
  void AttachAlignmentMetadataForLoad(llvm::LoadInst* load,
                                      int64_t buffer_size);

  // Given a load instruction and a shape or buffer size, annotate the load's
  // result with the dereferenceable bytes required by the shape / buffer size.
  void AttachDereferenceableMetadataForLoad(llvm::LoadInst* load,
                                            const Shape& shape);
  void AttachDereferenceableMetadataForLoad(llvm::LoadInst* load,
                                            int64_t buffer_size);

  // Calculate the alignment of a buffer allocated for a given shape.
  int MinimumAlignmentForShape(const Shape& shape);

  // Calculate the alignment of a buffer allocated for a given primitive type.
  int MinimumAlignmentForPrimitiveType(PrimitiveType primitive_type);

  // Returns the number of bytes within the shape.
  int64_t ByteSizeOf(const Shape& shape) const;

  enum class XfeedKind {
    kInfeed,
    kOutfeed,
  };

  // Emit IR to transfer between a {infeed,outfeed} buffer and an in-program
  // address.
  Status EmitXfeedTransfer(XfeedKind kind, const Shape& shape,
                           llvm::Value* program_buffer_address);

  // Returns a ConstExpr bitcast.
  llvm::Constant* EmitGlobalForLiteral(const Literal& literal);

  const HloModuleConfig& hlo_module_config_;

  bool is_top_level_computation_;

  const TargetMachineFeatures& target_machine_features_;

  struct LiteralPtrHashFunctor {
    size_t operator()(const Literal* literal) const {
      return absl::HashOf(*literal);
    }
  };

  struct LiteralPtrEqualityFunctor {
    bool operator()(const Literal* lhs, const Literal* rhs) const {
      return *lhs == *rhs;
    }
  };

  absl::flat_hash_map<const Literal*, llvm::Constant*, LiteralPtrHashFunctor,
                      LiteralPtrEqualityFunctor>
      emitted_literals_;

  absl::flat_hash_map<BufferAllocation::Index, llvm::Constant*>
      constant_buffer_to_global_;

  std::vector<const HloComputation*> thread_local_computations_;
  std::vector<const HloComputation*> global_computations_;

  bool emit_code_for_msan_;

  IrEmitter(const IrEmitter&) = delete;
  IrEmitter& operator=(const IrEmitter&) = delete;
};

}  // namespace cpu
}  // namespace xla

#endif  // TENSORFLOW_COMPILER_XLA_SERVICE_CPU_IR_EMITTER_H_
