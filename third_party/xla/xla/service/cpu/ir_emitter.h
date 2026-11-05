/* Copyright 2017 The OpenXLA Authors.

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

#ifndef XLA_SERVICE_CPU_IR_EMITTER_H_
#define XLA_SERVICE_CPU_IR_EMITTER_H_

#include <stddef.h>

#include <cstdint>
#include <functional>
#include <map>
#include <memory>
#include <optional>
#include <ostream>
#include <stack>
#include <string>
#include <utility>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "llvm/IR/Attributes.h"
#include "llvm/IR/BasicBlock.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/Value.h"
#include "xla/backends/cpu/codegen/target_machine_features.h"
#include "xla/hlo/ir/dfs_hlo_visitor_with_default.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/literal.h"
#include "xla/service/buffer_assignment.h"
#include "xla/service/cpu/elemental_ir_emitter.h"
#include "xla/service/cpu/ir_function.h"
#include "xla/service/hlo_module_config.h"
#include "xla/service/llvm_ir/alias_analysis.h"
#include "xla/service/llvm_ir/fused_ir_emitter.h"
#include "xla/service/llvm_ir/ir_array.h"
#include "xla/service/llvm_ir/ir_builder_mixin.h"
#include "xla/service/llvm_ir/loop_emitter.h"
#include "xla/service/name_uniquer.h"
#include "xla/xla_data.pb.h"

namespace xla {
namespace cpu {

// Forward declare emitter for XLA:CPU thunks.
class IrEmitter2;

bool IsNativeConvertSupportedOnTargetCPU(std::string feature_string);

// This class is the top-level API for the XLA HLO --> LLVM IR compiler.  It
// implements the DfsHloVisitor interface and emits HLO computations as LLVM IR
// functions.
// NOTE: A lot of functionality in this class (e.g. ElementTypesSameAndSupported
// helper function) is duplicated by ThunkEmitter and IrEmitter2. These two
// classes are part of the new runtime and will eventually replace IrEmitter.
class IrEmitter : public DfsHloVisitorWithDefault,
                  public IrBuilderMixin<IrEmitter> {
  class ElementalIrEmitter;

 public:
  using GeneratorForOperandIrArrays =
      std::function<std::vector<llvm_ir::IrArray>()>;

  // Create a new LLVM IR emitter.
  //
  // hlo_module: the HLO module we are emitting IR for.
  // assignment: a BufferAssignment from which we know which buffers are used by
  //             the HLO nodes.
  // llvm_module: the LLVM module to emit IR into.
  IrEmitter(const HloModule& hlo_module, const BufferAssignment& assignment,
            llvm::Module* llvm_module,
            const TargetMachineFeatures* target_machine,
            absl::flat_hash_map<BufferAllocation::Slice, int64_t>
                slice_to_buffer_table_index = {});
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
  //
  // If 'allow_reassociation' is true, the fast-math reassociation flag will
  // be enabled in the function's body. This is used when emitting reducers.
  absl::StatusOr<llvm::Function*> EmitComputation(
      const HloComputation* computation, absl::string_view function_name_prefix,
      bool is_top_level_computation,
      absl::Span<HloInstruction* const> instruction_order,
      bool allow_reassociation,
      absl::Span<const llvm::Attribute::AttrKind> function_attributes = {});

  llvm::IRBuilderBase* b() { return current_builder_; }
  const llvm::IRBuilderBase* b() const { return current_builder_; }
  // builder() is for IrBuilderMixin.
  llvm::IRBuilderBase* builder() { return current_builder_; }
  const llvm::IRBuilderBase* builder() const { return current_builder_; }

  IrFunction* compute_function() { return &compute_function_.top(); }

  // Used by IrEmitter
  void PushComputeFunction(const std::string& function_name,
                           llvm::Function::LinkageTypes linkage,
                           const HloModuleConfig& module_config,
                           llvm::Module* llvm_module) {
    compute_function_.emplace(function_name, linkage, module_config,
                              llvm_module, b());
  }

  // Used by IrEmitter2
  void PushComputeFunction(llvm::IRBuilderBase* b, llvm::Module* llvm_module,
                           llvm::Function* function,
                           llvm::BasicBlock* return_block) {
    function->getEntryBlock().getTerminator()->eraseFromParent();
    b->SetInsertPoint(&function->getEntryBlock());
    compute_function_.emplace(b, llvm_module, function, return_block);
  }

  void PopComputeFunction() {
    // At this point, the compute function destructor adds a branch to the
    // return block.
    compute_function_.pop();
  }

  // Emit LLVM global variable for a small constant buffer allocation.
  absl::Status EmitSmallConstantGlobals();

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

  // Returns true if given computation has been emitted.
  bool is_computation_emitted(const HloComputation& callee,
                              bool allow_reassociation) {
    return emitted_functions_.contains({&callee, allow_reassociation});
  }

  const TargetMachineFeatures& target_machine_features() const {
    return target_machine_features_;
  }

  const BufferAssignment& assignment() const { return assignment_; }

  // IRBuilderGuard is a RAII class that temporarily replaces the IRBuilder.
  // This is convenient for reusing the same logic with a different builder.
  class IRBuilderGuard {
   public:
    IRBuilderGuard() = default;
    explicit IRBuilderGuard(IrEmitter* ir_emitter, llvm::IRBuilderBase* builder)
        : ir_emitter_(ir_emitter),
          original_builder_(ir_emitter->current_builder_) {
      ir_emitter_->current_builder_ = builder;
    }

    IRBuilderGuard(IRBuilderGuard&& other) = delete;
    IRBuilderGuard& operator=(IRBuilderGuard&& other) = delete;

    ~IRBuilderGuard() {
      if (ir_emitter_ != nullptr) {
        ir_emitter_->current_builder_ = original_builder_;
      }
    }

   private:
    IrEmitter* ir_emitter_ = nullptr;
    llvm::IRBuilderBase* original_builder_ = nullptr;
  };

  // WithBuilder is a convenience function that creates and returns a
  // IRBuilderGuard for the current IrEmitter.
  [[nodiscard]] IRBuilderGuard WithBuilder(llvm::IRBuilderBase& builder) {
    return IRBuilderGuard(this, &builder);
  }

  absl::StatusOr<llvm::Function*> EmitNestedComputation(
      const HloComputation& callee, absl::string_view name, bool is_reducer);

 protected:
  friend class IrEmitter2;

  // Emit an LLVM global variable for every constant buffer allocation.
  absl::Status EmitConstantGlobals(std::optional<size_t> max_size_bytes);

  //
  // The following methods implement the DfsHloVisitor interface.
  //
  // Default action which emits code for most operations. Operations which are
  // special in some way are handled explicitly in HandleFoo methods.
  absl::Status DefaultAction(HloInstruction* hlo) override;

  absl::Status HandleAllGather(HloInstruction* instruction) override;
  absl::Status HandleAllToAll(HloInstruction* instruction) override;
  absl::Status HandleBitcast(HloInstruction* bitcast) override;
  absl::Status HandleConstant(HloInstruction* constant) override;
  absl::Status HandleCopy(HloInstruction* copy) override;
  absl::Status HandleGetTupleElement(
      HloInstruction* get_tuple_element) override;
  absl::Status HandleSelect(HloInstruction* select) override;
  absl::Status HandleDot(HloInstruction* dot) override;
  absl::Status HandleConvolution(HloInstruction* convolution) override;
  absl::Status HandleFft(HloInstruction* fft) override;
  absl::Status HandleAllReduce(HloInstruction* crs) override;
  absl::Status HandleReduceScatter(HloInstruction* crs) override;
  absl::Status HandleCollectivePermute(HloInstruction* crs) override;
  absl::Status HandleInfeed(HloInstruction* instruction) override;
  absl::Status HandleOutfeed(HloInstruction* outfeed) override;
  absl::Status HandleSort(HloInstruction* hlo) override;
  absl::Status HandleParameter(HloInstruction* parameter) override;
  absl::Status HandleReduce(HloInstruction* reduce) override;
  absl::Status HandleReduceWindow(HloInstruction* reduce_window) override;
  absl::Status HandleSend(HloInstruction* send) override;
  absl::Status HandleSendDone(HloInstruction* send_done) override;
  absl::Status HandleSlice(HloInstruction* slice) override;
  absl::Status HandleDynamicSlice(HloInstruction* dynamic_slice) override;
  absl::Status HandleDynamicUpdateSlice(
      HloInstruction* dynamic_update_slice) override;
  absl::Status HandleRecv(HloInstruction* recv) override;
  absl::Status HandleRecvDone(HloInstruction* recv_done) override;
  absl::Status HandlePad(HloInstruction* pad) override;
  absl::Status HandleTuple(HloInstruction* tuple) override;
  absl::Status HandleFusion(HloInstruction* fusion) override;
  absl::Status HandleCall(HloInstruction* call) override;
  absl::Status HandleCustomCall(HloInstruction* custom_call) override;
  absl::Status HandleWhile(HloInstruction* xla_while) override;
  absl::Status HandleConcatenate(HloInstruction* concatenate) override;
  absl::Status HandleConditional(HloInstruction* conditional) override;
  absl::Status HandleScatter(HloInstruction* scatter) override;
  absl::Status HandleAfterAll(HloInstruction* after_all) override;
  absl::Status HandleGetDimensionSize(HloInstruction* get_size) override;
  absl::Status HandleSetDimensionSize(HloInstruction* set_size) override;
  absl::Status HandleAddDependency(HloInstruction* add_dependency) override;
  absl::Status HandlePartitionId(HloInstruction* hlo) override;
  absl::Status HandleReplicaId(HloInstruction* hlo) override;
  absl::Status HandleRng(HloInstruction* rng) override;
  absl::Status HandleRngBitGenerator(HloInstruction* rng) override;
  absl::Status HandleRngGetAndUpdateState(HloInstruction* rng_state) override;
  absl::Status HandleBatchNormGrad(HloInstruction* batch_norm_grad) override;
  absl::Status HandleBatchNormTraining(
      HloInstruction* batch_norm_training) override;
  absl::Status HandleStochasticConvert(HloInstruction* instruction) override;
  absl::Status FinishVisit(HloInstruction* root) override;

  absl::Status Preprocess(HloInstruction* hlo) override;
  absl::Status Postprocess(HloInstruction* hlo) override;

  absl::Status HandlePad(HloInstruction* pad,
                         const llvm_ir::IrArray& operand_array,
                         const llvm_ir::IrArray& padding_value_array,
                         const llvm_ir::IrArray& output_array);

  // A convenient helper for calling BufferAssignment::GetUniqueSlice.
  BufferAllocation::Slice GetAllocationSlice(
      const HloInstruction& hlo, const ShapeIndex& index = {}) const {
    return assignment_.GetUniqueSlice(&hlo, index).value();
  }

 private:
  // Private helper to initialize an IR function for the computation.
  void InitializeIrFunction(const std::string& function_name);

  // Emits the copying epilogue for the function,
  // where it copies the returned value to the reserved alloca.
  // This is only necessary for thread-local functions.
  // Note that since the call graph is flattened, if the same function is
  // called in both thread-local and non-thread-local it would be codegen'd
  // twice, and we would know whether it's thread-local at codegen time.
  void EmitThreadLocalFunctionEpilogue(const HloComputation* computation);

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

  // Get the llvm::Value* that represents the "buffer_table" argument of the
  // computation function being emitted by this emitter.
  llvm::Value* GetBufferTableArgument();

  // Helper for EmitBufferPointer.
  llvm::Value* EmitGlobalBufferPointer(const BufferAllocation::Slice& slice,
                                       const Shape& target_shape);

  // Helper for EmitBufferPointer.
  llvm::Value* EmitThreadLocalBufferPointer(
      const BufferAllocation::Slice& slice, const Shape& target_shape);

  // Emits code that computes the address of the given buffer allocation slice.
  llvm::Value* EmitBufferPointer(const BufferAllocation::Slice& slice,
                                 const Shape& target_shape);

  // Similar to EmitThreadLocal, yet assumes that the function returns a scalar.
  llvm::Value* EmitScalarReturningThreadLocalCall(
      const HloComputation& callee, absl::Span<llvm::Value* const> parameters,
      absl::string_view name);

  // Emits a call to a "global" function (e.g. to the computation nested within
  // a kWhile or a kCall).  Buffer assignment unabiguously assigns buffers to
  // the parameters and return values for these computations so there is no need
  // to explicitly pass parameters or return results.
  void EmitGlobalCall(const HloComputation& callee, absl::string_view name);

  // Returns an error if `callee` is not a global computation, i.e. if it is
  // called from an embedded computation such as a sort comparator.
  absl::Status CheckGlobalCallee(const HloComputation& callee);

  // Returns the buffer to which a global call to `callee` would have written
  // its result.
  llvm::Value* GetBufferForGlobalCallReturnValue(const HloComputation& callee);

  // Verifies that the element types of all of the given operand instructions
  // match and are of one of the given supported types.
  absl::Status ElementTypesSameAndSupported(
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
  absl::Status EmitTargetElementLoop(
      const HloInstruction* target_op, absl::string_view desc,
      const llvm_ir::ElementGenerator& element_generator,
      std::optional<llvm_ir::IrArray> result_array_opt);

  // Emits a memcpy from the source instruction's result value to the
  // destination's.  Both source and destination must have an entry in the
  // emitted_value_ table.
  absl::Status EmitMemcpy(const HloInstruction& source,
                          const HloInstruction& destination);

  // Emits IR to compute the target address of the buffer for the given op.
  // After calling this function, you can get a pointer to this buffer by
  // calling GetIrArrayForOp or GetEmittedValueFor.
  absl::Status EmitTargetAddressForOp(const HloInstruction* op);

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
  absl::StatusOr<bool> EmitVectorizedReduce(
      HloInstruction* reduce, HloInstruction* arg, HloInstruction* init_value,
      absl::Span<const int64_t> dimensions, HloComputation* function,
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

  using ReductionGenerator = std::function<llvm::Value*(
      llvm::IRBuilderBase*, llvm::Value*, llvm::Value*)>;

  // Tries to match the reduction function "function" to a known reduction
  // pattern.  Returns a non-null ReductionGenerator on a successful match,
  // which can be used to generate the LLVM IR corresponding to said reduction.
  // On failure, this stores a reason string into "failure_reason".
  ReductionGenerator MatchReductionGenerator(HloComputation* function,
                                             std::string* failure_reason) const;

  // Emits the inner loop nest that runs the reduction.  Helper function for
  // EmitVectorizedReduce.
  absl::StatusOr<ShardedVector> EmitInnerLoopForVectorizedReduction(
      const ReductionGenerator& reduction_generator,
      const llvm_ir::IrArray::Index& output_index,
      const ShardedVectorType& accumulator_type, HloInstruction* init_value,
      HloInstruction* arg, absl::Span<const int64_t> dimensions,
      llvm::Align element_alignment);

  // Checks if the given concatenate instruction can use a fast (memcpy)
  // implementation.
  absl::Status CanDoFastConcatenate(const HloInstruction* instr) const;

  // Emits a fast concatenate operation using memcpy. Assumes all preconditions
  // are met prior to calling this function (see CanDoFastConcatenate).
  absl::Status EmitFastConcatenate(
      const HloInstruction* instr,
      absl::Span<const llvm_ir::IrArray> source_arrays,
      const llvm_ir::IrArray& target_array);

  // Emits LLVM IR to transfer "element_count" elements of type "primitive_type"
  // from the address "source" to the address "target".
  void EmitTransferElements(llvm::Value* target, llvm::Value* source,
                            int64_t element_count, PrimitiveType primitive_type,
                            const llvm_ir::IrArray& target_array,
                            const llvm_ir::IrArray& source_array);

  // Emit slice-to-dynamic.
  absl::Status EmitSliceToDynamic(
      const HloInstruction* hlo,
      absl::Span<const llvm_ir::IrArray> source_arrays,
      const llvm_ir::IrArray& target_array);

  // Assignment of the buffers needed by the computation and their shape
  // information.
  const BufferAssignment& assignment_;

  // The LLVM module into which IR will be emitted.
  llvm::Module* module_;

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
  // N.B. `main_builder_` must be ordered before `compute_function_` as
  // `IrFunction::~IrFunction` references `main_builder_`. This will ensure that
  // the destructor for `compute_function_` will run before the destructor for
  // `main_builder_`.
  llvm::IRBuilder<> main_builder_;
  // The current builder to use for IR emission. This is either `main_builder_`
  // or a temporary builder that replaces it.
  llvm::IRBuilderBase* current_builder_;
  std::stack<IrFunction> compute_function_;
  // The state of allow_reassociation_ is required so that that it is
  // transitive to all nested computations.
  bool allow_reassociation_ = false;

  // The buffer allocation slice for the root of the computation being compiled.
  // Only relevant for thread local computations.
  BufferAllocation::Slice computation_root_allocation_;

  // Maps the buffer allocation slices for the parameters to the computation
  // being compiled to their parameter numbers.  Only relevant for thread local
  // computations.
  absl::flat_hash_map<BufferAllocation::Index, int64_t>
      computation_parameter_allocations_;

  // Maps HLOs to Values emitted for them.
  absl::flat_hash_map<const HloInstruction*, llvm::Value*> emitted_value_;

  llvm_ir::AliasAnalysis alias_analysis_;

  // Given a load instruction and a shape or buffer size, annotate the load's
  // result with the alignment required by the shape or size.
  void AttachAlignmentMetadataForLoad(llvm::LoadInst* load, const Shape& shape);
  void AttachAlignmentMetadataForLoad(llvm::LoadInst* load,
                                      int64_t buffer_size);

  // Given a load instruction and a shape or buffer size, annotate the load's
  // result with the dereferenceable bytes required by the shape / buffer size.
  void AttachDereferenceableMetadataForLoad(llvm::LoadInst* load,
                                            const Shape& shape);
  static void AttachDereferenceableMetadataForLoad(llvm::LoadInst* load,
                                                   int64_t buffer_size);

  // Given a load instruction, annotate the load's result with the invariant
  // load metadata.
  void AttachInvariantLoadMetadataForLoad(llvm::LoadInst* load) const;
  static void AttachInvariantLoadMetadataForLoad(llvm::LoadInst* load,
                                                 const HloModuleConfig& config);

  // Calculate the alignment of a buffer allocated for a given shape.
  int MinimumAlignmentForShape(const Shape& shape);

  // Calculate the alignment of a buffer allocated for a given primitive type.
  int MinimumAlignmentForPrimitiveType(PrimitiveType primitive_type);

  // Returns the number of bytes within the shape.
  int64_t ByteSizeOf(const Shape& shape) const;

  // Returns a ConstExpr bitcast.
  llvm::Constant* EmitGlobalForLiteral(const Literal& literal);

  CpuElementalIrEmitter ElementalIrEmmiterFactory();

  const HloModule& hlo_module_;
  const HloModuleConfig& hlo_module_config_;

  bool is_top_level_computation_;

  const TargetMachineFeatures& target_machine_features_;

  struct LayoutSensitiveLiteralWrapper {
    const Literal& literal;

    template <typename H>
    friend H AbslHashValue(H h, const LayoutSensitiveLiteralWrapper& wrapper) {
      return Literal::Hash<H, /*layout_sensitive=*/true>(std::move(h),
                                                         wrapper.literal);
    }

    bool operator==(const LayoutSensitiveLiteralWrapper& other) const {
      return literal.Equal(other.literal, /*layout_sensitive=*/true);
    }

    // This is needed for InsertOrDie to work.
    friend std::ostream& operator<<(
        std::ostream& out, const LayoutSensitiveLiteralWrapper& wrapper) {
      return out << wrapper.literal;
    }
  };

  absl::flat_hash_map<LayoutSensitiveLiteralWrapper, llvm::Constant*>
      emitted_literals_;

  absl::flat_hash_map<BufferAllocation::Index, llvm::Constant*>
      constant_buffer_to_global_;

  std::vector<const HloComputation*> thread_local_computations_;
  std::vector<const HloComputation*> global_computations_;

  absl::flat_hash_map<BufferAllocation::Slice, int64_t>
      slice_to_buffer_table_index_;

  IrEmitter(const IrEmitter&) = delete;
  IrEmitter& operator=(const IrEmitter&) = delete;
};

// Decoupled implementation of IrEmitter::EmitTransferElements.
void EmitTransferElements(llvm::Value* target, llvm::Value* source,
                          int64_t element_count, PrimitiveType primitive_type,
                          const llvm_ir::IrArray& target_array,
                          const llvm_ir::IrArray& source_array,
                          llvm::Module* module, llvm::IRBuilderBase& b);

// Decoupled implementation of IrEmitter::EmitFastConcatenate.
// Returns true if the concatenate was parallelized.
absl::StatusOr<bool> EmitFastConcatenate(
    const HloInstruction* instr,
    absl::Span<const llvm_ir::IrArray> source_arrays,
    const llvm_ir::IrArray& target_array, llvm::Module* module,
    llvm::IRBuilderBase& b, llvm::Value* workgroup_id = nullptr,
    int64_t num_workgroups = -1);

}  // namespace cpu
}  // namespace xla

#endif  // XLA_SERVICE_CPU_IR_EMITTER_H_
