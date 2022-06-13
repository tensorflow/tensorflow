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

#ifndef TENSORFLOW_COMPILER_XLA_SERVICE_HLO_COMPUTATION_H_
#define TENSORFLOW_COMPILER_XLA_SERVICE_HLO_COMPUTATION_H_

#include <functional>
#include <list>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "absl/strings/cord.h"
#include "absl/types/span.h"
#include "tensorflow/compiler/xla/iterator_util.h"
#include "tensorflow/compiler/xla/map_util.h"
#include "tensorflow/compiler/xla/service/dfs_hlo_visitor.h"
#include "tensorflow/compiler/xla/service/hlo.pb.h"
#include "tensorflow/compiler/xla/service/hlo_clone_context.h"
#include "tensorflow/compiler/xla/service/hlo_instruction.h"
#include "tensorflow/compiler/xla/service/name_uniquer.h"
#include "tensorflow/compiler/xla/shape_tree.h"
#include "tensorflow/compiler/xla/statusor.h"
#include "tensorflow/compiler/xla/types.h"
#include "tensorflow/compiler/xla/xla_data.pb.h"
#include "tensorflow/core/lib/core/status.h"

namespace xla {

class HloModule;

// Describes a computation at the HLO level.
//
// You can think of an HloComputation like a function.  It has some inputs
// (parameters) and returns exactly one value (the value of its root node).  If
// you want to return multiple values, you can return a tuple.
//
// The instructions inside of a computation do not have an explicit total order.
// Instead, they have a partial order determined by their data and control
// dependencies.
//
// An HloModule contains one "entry computation" -- this is like main() in a C
// program.  Every other computation inside of a module is attached to one or
// more HloInstructions, as a "nested computation".  For example, the kMap
// instruction has a nested computation and "applies" it to every element of its
// input, elementwise.  (That is, the input [x, y, z] is transformed to [f(x),
// f(y), f(z)].)
class HloComputation {
 public:
  // Used by instructions_.
  using InstructionList = std::list<std::unique_ptr<HloInstruction>>;

  // Builder class for HloComputation.
  class Builder {
   public:
    explicit Builder(const std::string& name,
                     HloInstruction* fusion_instruction = nullptr)
        : name_(name),
          last_added_instruction_(nullptr),
          fusion_instruction_(fusion_instruction) {}
    Builder(Builder&& b) = default;
    virtual ~Builder() = default;

    // Build and return an HloComputation. The parameter root_instruction
    // specifies the already-added instruction to use as the root. If
    // root_instruction is nullptr then use the last added instruction as the
    // root.
    std::unique_ptr<HloComputation> Build(
        HloInstruction* root_instruction = nullptr);

    virtual HloInstruction* AddInstruction(
        std::unique_ptr<HloInstruction> instruction) {
      instructions_.push_back(std::move(instruction));
      last_added_instruction_ = instructions_.back().get();
      return last_added_instruction_;
    }

    Status ForEachInstruction(
        const std::function<Status(const HloInstruction*)>& func) const {
      for (const auto& instruction : instructions_) {
        TF_RETURN_IF_ERROR(func(instruction.get()));
      }
      return OkStatus();
    }

    HloInstruction* last_added_instruction() const {
      return last_added_instruction_;
    }

   private:
    const std::string name_;
    HloInstruction* last_added_instruction_;
    HloInstruction* fusion_instruction_;
    std::vector<std::unique_ptr<HloInstruction>> instructions_;

    Builder(const Builder&) = delete;
    Builder& operator=(const Builder&) = delete;
  };

  // Helper class to automatically set the OpMetadata for every instruction
  // added to a computation.
  class MetadataBuilder {
   public:
    MetadataBuilder(HloComputation* computation, const OpMetadata& metadata)
        : computation_(computation), metadata_(metadata) {}

    HloInstruction* AddInstruction(
        std::unique_ptr<HloInstruction> instruction) {
      instruction->set_metadata(metadata_);
      return computation_->AddInstruction(std::move(instruction));
    }

   private:
    HloComputation* computation_;
    OpMetadata metadata_;
  };

  ~HloComputation();

  // Add an instruction to the computation. The computation takes ownership of
  // the instruction.
  HloInstruction* AddInstruction(std::unique_ptr<HloInstruction> instruction,
                                 const std::string& new_name = "");

  // Replace the old parameter at index param_no with
  // `instruction`. Updates uses and root instruction. Removes old
  // instruction from computation. No check is done on the shape.
  HloInstruction* ReplaceParameter(int64_t param_no,
                                   std::unique_ptr<HloInstruction> instruction);

  // Remove the param_no'th parameter from the computation.
  // Note this is only applicatable to the computation for the fusion
  // instruction.
  Status RemoveParameter(int64_t param_no);

  // Remove unused parameters from the computation.
  // Note this is only applicatable to the computation for the fusion
  // instruction.
  Status RemoveUnusedParametersFromFusedComputation();

  // Remove unused parameters from the computation. Unlike
  // RemoveUnusedParametersFromFusedComputation, this function can be used
  // to remove parameters from non-fusion computations.
  Status RemoveUnusedParametersFromAnyComputation();

  // Adds a new parameter instruction to a fusion computation.
  //
  // This should be a new parameter. Instruction will be appended to parameters
  // and inserted to the instruction list.
  HloInstruction* AddParameter(std::unique_ptr<HloInstruction> instruction);

  // Adds a new parameter instruction to the entry computation and update
  // the parent module config to reflect the change.
  //
  // This should be a new parameter. Instruction will be appended to parameters
  // and inserted to the instruction list.
  HloInstruction* AddEntryComputationParameter(
      std::unique_ptr<HloInstruction> instruction);

  // Replaces an old parameter with a new parameter. Adds the new parameter
  // instruction to the entry computation.  Updates users instruction.
  Status ReplaceEntryComputationParameter(
      int64_t param_no, HloInstruction* old_instruction,
      std::unique_ptr<HloInstruction> instruction);

  // Remove an instruction from the computation. The instruction must have no
  // users. Instruction is deallocated with this call.
  Status RemoveInstruction(HloInstruction* instruction);

  // Removes an instruction from the computation. The instruction must have no
  // users. Instruction is deallocated with this call. The instruction will be
  // removed even if it is marked as not removable.
  Status ForceRemoveInstruction(HloInstruction* instruction);

  // Remove an instruction (including side effecting ones) from the computation
  // and also transitively any operand that has no side effect and no users post
  // removing an instruction. The instruction must have no users. Instruction is
  // deallocated with this call. If given, the cleanup routine is executed on a
  // removed instruction before its deallocation.
  Status RemoveInstructionAndUnusedOperands(
      HloInstruction* instruction,
      std::function<void(HloInstruction*)> cleanup = nullptr);

  // Set the root of the computation to the given instruction. The instruction
  // must have already been added to the computation. In addition it must have
  // the same shape as the result of the computation for non fusion
  // computations, except if accept_different_shape is set to true.
  void set_root_instruction(HloInstruction* new_root_instruction,
                            bool accept_different_shape = false);

  // Return the root instruction of the computation. The root instruction is the
  // instruction which produces the output of the computation.
  HloInstruction* root_instruction() const { return root_instruction_; }

  // Returns the number of parameters for this computation.
  int64_t num_parameters() const { return param_instructions_.size(); }

  // Returns the parameter instruction for the given parameter number.
  HloInstruction* parameter_instruction(int64_t param_no) const {
    CHECK_GE(param_no, 0);
    CHECK_LT(param_no, static_cast<int64_t>(param_instructions_.size()))
        << "Computation " << name() << " has no parameter number " << param_no;
    return param_instructions_[param_no];
  }

  const std::vector<HloInstruction*>& parameter_instructions() const {
    return param_instructions_;
  }

  const std::string& name() const { return name_; }

  // Use the given NameUniquer to select a unique name for the computation based
  // on the computation's existing name.
  void UniquifyName(NameUniquer* name_uniquer);

  // Return a string representation of the computation.
  //
  // (We express the default options using an overload rather than a default
  // param because gdb ignores default params, but does resolve overloads.)
  std::string ToString() const { return ToString(HloPrintOptions()); }
  std::string ToString(const HloPrintOptions& options) const;

  // Overload which accepts an order to emit the instructions in.
  std::string ToString(
      const HloPrintOptions& options,
      absl::Span<const HloInstruction* const> instruction_order) const;

  // Returns a Cord representation of the computation.
  //
  // (We express the default options using an overload rather than a default
  // param because gdb ignores default params, but does resolve overloads.)
  absl::Cord ToCord() const { return ToCord(HloPrintOptions()); }
  absl::Cord ToCord(const HloPrintOptions& options) const;

  // Overload which accepts an order to emit the instructions in.
  absl::Cord ToCord(
      const HloPrintOptions& options,
      absl::Span<const HloInstruction* const> instruction_order) const;

  // Returns a serialized representation of this computation.
  HloComputationProto ToProto() const;

  // Creates a computation from the given proto. Arguments:
  //
  //   proto: the proto to convert from.
  //   computation_map: a map from computation id to HloComputation*. This map
  //     must contain all computations which the newly constructed computation
  //     calls.
  static StatusOr<std::unique_ptr<HloComputation>> CreateFromProto(
      const HloComputationProto& proto,
      const absl::flat_hash_map<int64_t, HloComputation*>& computation_map,
      bool prohibit_empty_literal = true);

  // Generates a hash value of an HLO computation. Hash considers
  // information on opcode, shape, operands, and typically a root instruction.
  // This function returns the same hash value for equivalent HLO computations,
  // with respect to HloComputation::Equal() method.
  template <typename H>
  friend H AbslHashValue(H h, const HloComputation& computation) {
    auto instructions = computation.MakeInstructionPostOrder();
    for (auto* instruction : instructions) {
      h = H::combine(std::move(h), *instruction);
    }
    return H::combine(std::move(h), instructions.size());
  }

  using InstructionSequence = tensorflow::gtl::iterator_range<
      UnwrappingIterator<std::list<std::unique_ptr<HloInstruction>>::iterator>>;

  using ConstInstructionSequence =
      tensorflow::gtl::iterator_range<UnwrappingIterator<
          std::list<std::unique_ptr<HloInstruction>>::const_iterator>>;

  // Gets the instructions in this computation.
  //
  // The returned type is a range of HloInstruction*s, so you can iterate over
  // it using a range-based for loop in the natural way:
  //
  //   for (HloInstruction* instr : computation->instructions()) { ... }
  //
  ConstInstructionSequence instructions() const {
    return {MakeUnwrappingIterator(instructions_.begin()),
            MakeUnwrappingIterator(instructions_.end())};
  }
  InstructionSequence instructions() {
    return {MakeUnwrappingIterator(instructions_.begin()),
            MakeUnwrappingIterator(instructions_.end())};
  }

  // Compute and return a post-order of the instructions in the computation. In
  // this order, definitions of values always appear before their uses.
  std::vector<HloInstruction*> MakeInstructionPostOrder() const;

  int64_t instruction_count() const { return instruction_iterators_.size(); }

  // Creates and returns a list of the embedded computations called by this
  // computation. This includes all embedded computations called directly or
  // transitively. The embedded computations are sorted such that if computation
  // A calls computation B (eg, via a map instruction) then A will appear after
  // B in the list.
  std::vector<HloComputation*> MakeEmbeddedComputationsList() const;

  // Creates a fusion instruction containing the given instructions.
  // `fusion_kind` indicates the type of the fusion, e.g., loop fusion or fusion
  // into a library call. Instructions must be in reverse topological order
  // (root of the fused expression first). Replaces all uses of the original
  // root instruction with the fusion instruction. The original instructions are
  // removed if they have no uses after fusion (this is necessarily true for at
  // least the root).
  HloInstruction* CreateFusionInstruction(
      absl::Span<HloInstruction* const> instructions_to_fuse,
      HloInstruction::FusionKind fusion_kind);

  // Creates an async start/done instruction pair where instruction is wrapped
  // inside an asynchronous computation. The context shapes are appended to the
  // output tuple of the asynchronous start which is backend specific. Returns
  // the async done instruction. The new async start instruction is the operand
  // of the async done instruction so that can be accessed using that.
  StatusOr<HloInstruction*> CreateAsyncInstructions(
      HloInstruction* instruction, absl::Span<const Shape> context_shapes);

  // Create a deep copy of the given instruction and return the instruction
  // producing the copied result. All instructions performing the copy are added
  // to the computation. For array-shaped values, this method trivially returns
  // a kCopy instruction. For tuple-shaped instructions, the copy is performed
  // with a series of kGetTupleElement and kTuple instructions. If
  // indices_to_copy is non-null then this ShapeTree indicates which elements
  // (arrays) of the shape to copy. Non-copied elements are passed through
  // transparently. If copies_added is non-null, then the added kCopy
  // instructions will be inserted in the respective index in the given
  // ShapeTree.
  StatusOr<HloInstruction*> DeepCopyInstruction(
      HloInstruction* instruction,
      const ShapeTree<bool>* indices_to_copy = nullptr,
      ShapeTree<HloInstruction*>* copies_added = nullptr);

  // As above, but uses a custom function to copy the leaf nodes, which could
  // create alternative HLOs other than kCopy, or even pass-throughs.
  StatusOr<HloInstruction*> DeepCopyInstructionWithCustomCopier(
      HloInstruction* instruction,
      const std::function<
          HloInstruction*(HloInstruction* leaf, const ShapeIndex& leaf_index,
                          HloComputation* computation)>& copy_leaf);

  // Computes and returns the ProgramShape of this computation (shape of
  // parameters and result with layout).
  ProgramShape ComputeProgramShape(bool include_ids = true) const;

  // Return whether `*this` and `other` are functionally equivalent.
  bool Equal(const HloComputation& other, bool is_layout_sensitive) const {
    return EqualInternal(other, is_layout_sensitive,
                         /*ignore_channel_id_values=*/false);
  }

  // Same as Equal() but ignores channel ID value mismatches on instructions, as
  // long as the two instructions both have channel IDs or neither has a channel
  // ID.
  bool EqualIgnoringChannelIdValues(const HloComputation& other,
                                    bool is_layout_sensitive) const {
    return EqualInternal(other, is_layout_sensitive,
                         /*ignore_channel_id_values=*/true);
  }

  // Return whether `*this` and `other` are functionally equivalent.
  bool operator==(const HloComputation& other) const {
    return Equal(other, true);
  }
  bool operator!=(const HloComputation& other) const {
    return !(*this == other);
  }

  // Replaces old instruction with newly created instruction. Removes old
  // instruction from computation. Updates uses and root instruction.
  Status ReplaceWithNewInstruction(
      HloInstruction* old_instruction,
      std::unique_ptr<HloInstruction> new_instruction);

  // Replaces an old instruction with a newly created instruction, and adds the
  // new instruction as an entry computation's parameter. Removes old
  // instruction from computation. Updates uses and root instruction.
  Status ReplaceWithNewEntryComputationParameter(
      HloInstruction* old_instruction,
      std::unique_ptr<HloInstruction> new_instruction);

  // Replace old instruction with new instruction.  Updates uses and root
  // instruction. Removes old instruction from computation. Precondition:
  // old_instruction and new_instruction must have the compatible shapes.
  // If preserve_sharding is true, the replacement will fail if both new and old
  // instruction have sharding that is not compatible, and the function will
  // return false. Otherwise, when the replacement happens, if |new_instruction|
  // doesn't have any sharding information it will receive the sharding
  // information of |old_instruction|, and function will return true.
  StatusOr<bool> ReplaceInstruction(HloInstruction* old_instruction,
                                    HloInstruction* new_instruction,
                                    bool preserve_sharding);

  // Same as above, with preserve_sharding=false. Since this replacement always
  // happens, it returns just a Status as opposed to StatusOr<bool>
  Status ReplaceInstruction(HloInstruction* old_instruction,
                            HloInstruction* new_instruction);

  // Same as ReplaceInstruction, but the new instruction can have a different
  // shape.
  StatusOr<bool> ReplaceInstructionWithDifferentShape(
      HloInstruction* old_instruction, HloInstruction* new_instruction,
      bool preserve_sharding);
  Status ReplaceInstructionWithDifferentShape(HloInstruction* old_instruction,
                                              HloInstruction* new_instruction);

  // Set/get the module containing this computation.
  void set_parent(HloModule* module) { parent_ = module; }
  const HloModule* parent() const { return parent_; }
  HloModule* parent() { return parent_; }

  // Visit every node in the computation in DFS post-order with the given
  // visitor. This is similar to calling HloInstruction::Accept on the root of
  // the computation except this method also visits instructions not reachable
  // via the root. The root instruction of the computation is visited last, and
  // the visitor's FinishVisit method is called once upon completion (with the
  // root instruction as the argument).
  template <typename HloInstructionPtr>
  Status Accept(DfsHloVisitorBase<HloInstructionPtr>* visitor) const;

  // Same as Accept() above, but the order of operand and control predecessor
  // visitation is determined by the given operand order; if compare(A, B) ==
  // true, A is visited before B.
  Status AcceptWithOperandOrder(
      DfsHloVisitor* visitor,
      const HloInstruction::CompareFunction& operand_order) const;

  // Visit every node in the computation in the given order. 'order' must
  // be a topological sort of all instructions in the computation.
  template <typename HloInstructionPtr>
  Status AcceptOrdered(DfsHloVisitorBase<HloInstructionPtr>* visitor,
                       absl::Span<HloInstruction* const> order) const;

  // Returns a deep copy of this computation including all instructions.
  // If the clone context is specified, it will be populated with the cloned
  // object mappings, and its module() will be used to add new computations
  // into.
  std::unique_ptr<HloComputation> Clone(const std::string& suffix = "clone",
                                        HloCloneContext* context = nullptr);

  // Like Clone(), but if an instruction is present in replacement_map, we use
  // the map's value to replace that instruction in the cloned computation.
  //
  // If replacements is nullptr, don't perform replacement.
  // If replacements maps a key to nullptr, we remove that instruction from the
  // new computation.  If an element of `replacements` references an instruction
  // that's not already in the computation, it's cloned and added to the new
  // computation.
  //
  // 'extra_parameters' allows to specify additional parameters that should be
  // added to the computation.
  //
  // All relevant instructions are cloned, *including* unique_ptr in the
  // `replacements` map.
  std::unique_ptr<HloComputation> CloneWithReplacements(
      const absl::flat_hash_map<const HloInstruction*,
                                std::unique_ptr<HloInstruction>>* replacements,
      absl::Span<const HloInstruction* const> extra_parameters = {},
      HloCloneContext* context = nullptr, const std::string& suffix = "clone",
      const HloInstruction* new_root = nullptr);

  // Convenience overloads for CloneWithReplacements.  You want to do
  //
  //   CloneWithReplacements({{a, std::move(b)}, {c, std::move(d)}})  // ERROR
  //
  // but that doesn't work because std::initializer_list is not movable.  These
  // overloads let you do
  //
  //   CloneWithReplacementPairs({a, std::move(b)}, {c, std::move(d)});   // OK
  //
  std::unique_ptr<HloComputation> CloneWithReplacementPairs(
      std::pair<const HloInstruction*, std::unique_ptr<HloInstruction>> r1,
      HloCloneContext* context = nullptr, const std::string& suffix = "clone");
  std::unique_ptr<HloComputation> CloneWithReplacementPairs(
      std::pair<const HloInstruction*, std::unique_ptr<HloInstruction>> r1,
      std::pair<const HloInstruction*, std::unique_ptr<HloInstruction>> r2,
      HloCloneContext* context = nullptr, const std::string& suffix = "clone");
  std::unique_ptr<HloComputation> CloneWithReplacementPairs(
      std::pair<const HloInstruction*, std::unique_ptr<HloInstruction>> r1,
      std::pair<const HloInstruction*, std::unique_ptr<HloInstruction>> r2,
      std::pair<const HloInstruction*, std::unique_ptr<HloInstruction>> r3,
      HloCloneContext* context = nullptr, const std::string& suffix = "clone");

  // Returns true if the given instruction can be removed from the computation.
  // Parameter instructions cannot be removed without violating invariants of
  // the HLO computation with the exception of fusion computation. A parameter
  // instruction is removable for a fusion computation.
  //
  // Note that IsSafelyRemovable() is a necessary condition to remove an
  // instruction rather than a sufficient condition. For example, instructions
  // with side-effect (e.g., Send, Infeed) may be removed from a computation,
  // but the transformation must guarantee the invariants relevant to the
  // instructions still hold (e.g., Send and Recv must be removed together to
  // make each channel complete).
  bool IsSafelyRemovable(const HloInstruction* instruction);

  // Returns a map from channel-id to the group of instructions associated with
  // the channel. These instructions will be considered as a single node for
  // dependency purposes. Send and RecvDone are in the group, and AllReduces
  // with the same channel id are in the group.
  using ChannelDependencyGroup =
      absl::flat_hash_map<int64_t, absl::InlinedVector<HloInstruction*, 1>>;
  ChannelDependencyGroup ComputeChannelDependencies() const;

  // Returns true if this computation has a side effect. A computation has a
  // side effect if it contains one or more instructions with a side effect.
  bool HasSideEffect() const;

  // Returns if this computation is a fusion computation.
  // Do not use this method to determine if fusion_instruction_ != nullptr.
  // Instead, directly do: FusionInstruction() != nullptr
  bool IsFusionComputation() const { return is_fusion_computation_; }

  // Returns if this computation is the entry computation of the module.
  bool IsEntryComputation() const;

  // Returns the owning fusion instruction, or nullptr if this is not a fusion
  // computation.
  HloInstruction* FusionInstruction() const { return fusion_instruction_; }
  void SetFusionInstruction(HloInstruction* fusion_instruction) {
    fusion_instruction_ = fusion_instruction;
    is_fusion_computation_ |= (fusion_instruction != nullptr);
  }

  // Returns if this computation is a custom-call computation.
  bool IsCustomCallComputation() const { return is_custom_call_computation_; }

  // Returns the owning custom call instruction, or nullptr if this is not a
  // custom call computation.
  HloInstruction* CustomCallInstruction() const {
    return custom_call_instruction_;
  }
  void SetCustomCallInstruction(HloInstruction* custom_call_instruction) {
    custom_call_instruction_ = custom_call_instruction;
    is_custom_call_computation_ |= (custom_call_instruction != nullptr);
  }

  // Returns if this computation is an async computation.
  bool IsAsyncComputation() const { return !async_instructions_.empty(); }

  // Returns the owning async instruction. It's empty if this is not an async
  // computation.
  const std::vector<HloInstruction*>& AsyncInstructions() const {
    return async_instructions_;
  }

  std::vector<HloInstruction*>& AsyncInstructions() {
    return async_instructions_;
  }

  void AddAsyncInstruction(HloInstruction* async_instruction) {
    CHECK(async_instruction != nullptr)
        << "Nullptr shouldn't be added as commputation's async instruction. ";
    async_instructions_.push_back(async_instruction);
  }

  void RemoveAsyncInstruction(HloInstruction* async_instruction) {
    if (async_instruction == nullptr) {
      return;
    }
    async_instructions_.erase(
        std::remove(async_instructions_.begin(), async_instructions_.end(),
                    async_instruction),
        async_instructions_.end());
  }

  // Returns if this computation is invoked by an Hlo instruction.
  bool IsCalledComputation() const {
    return IsFusionComputation() || IsCustomCallComputation();
  }

  // Clear the unique ID of the computation so that it can be re-assigned, such
  // as for the purpose of compacting the unique IDs.
  void ClearUniqueIdInternal() { unique_id_ = -1; }

  // The id of this computation should be unique within the module.
  void SetUniqueId(int64_t id) {
    CHECK_EQ(unique_id_, -1);
    CHECK_GE(id, 0);
    unique_id_ = id;
  }

  // Returns the instruction in this computation that has name `name`.  Returns
  // null if there is no such computation.
  HloInstruction* GetInstructionWithName(absl::string_view name);

  int64_t unique_id() const { return unique_id_; }

  // Deallocate instructions that are marked by "RemoveInstruction". The two
  // stage clean up process is designed such that HloPass can have stable
  // internal pointers to HloInstructions while we create and remove
  // HloInstructions in a pass.
  void Cleanup() { to_be_deleted_.clear(); }

  // Returns true if a given instruction is marked dead in this computation.
  bool IsMarkedAsDead(const HloInstruction* inst);

 private:
  explicit HloComputation(
      const std::string& name, int parameter_count,
      std::vector<std::unique_ptr<HloInstruction>>* instructions,
      HloInstruction* root_instruction, HloInstruction* fusion_instruction);

  // Internal helper for adding instructions.
  HloInstruction* AddInstructionInternal(
      std::unique_ptr<HloInstruction> instruction);

  // Internal helper for comparison with different options.
  bool EqualInternal(const HloComputation& other, bool is_layout_sensitive,
                     bool ignore_channel_id_values) const;

  // Fuses HLOs in instructions_to_fuse into fusion_instruction.
  //
  // Pre-condition: fusion_instruction's opcode is kFusion.
  void FuseInstructionsInto(
      absl::Span<HloInstruction* const> instructions_to_fuse,
      HloInstruction* fusion_instruction);

  // Internal helper for recursive copying of an instruction. Creates and
  // returns a deep copy of the given instruction.
  StatusOr<HloInstruction*> DeepCopyHelper(
      HloInstruction* instruction, ShapeIndex* index,
      const std::function<
          HloInstruction*(HloInstruction* leaf, const ShapeIndex& leaf_index,
                          HloComputation* computation)>& copy_leaf);

  // Internal helper to collect unreachable roots.
  std::vector<HloInstruction*> CollectUnreachableRoots() const;

  enum VisitState { kVisiting, kVisited };
  void ComputeInstructionPostOrder(
      HloInstruction* root,
      HloComputation::ChannelDependencyGroup& channel_dependencies,
      absl::flat_hash_map<HloInstruction*, VisitState>& visited,
      std::vector<HloInstruction*>& post_order) const;

  Status RemoveUnusedParametersImpl(bool allow_non_fusion);

  Status RemoveInstructionImpl(HloInstruction* instruction,
                               bool ignore_safety_check);

  std::string name_;
  int64_t unique_id_;
  HloInstruction* root_instruction_;

  // If this computation is a fusion computation, this field points to the
  // corresponding fusion instruction (if it is live). Otherwise, this is null.
  HloInstruction* fusion_instruction_;

  // Determines whether this computation is a fusion computation. A fusion
  // computation ordinarily also has a non-null fusion_instruction_. However, if
  // a fusion instruction is removed during compilation, the fusion computation
  // becomes unreachable, and its fusion_instruction_ is set to null. We still
  // need to regard such computations as fusion computations for HLO scheduling
  // purposes.
  bool is_fusion_computation_;

  // If this computation is a custom-call computation, this field points to the
  // corresponding custom-call instruction (if it is live). Otherwise, this is
  // null.
  HloInstruction* custom_call_instruction_;

  // Determines whether this computation is a custom-call computation.
  bool is_custom_call_computation_;

  // If this computation is an async computation, this field points to the
  // corresponding async instructions (if live) that call this computation.
  // Otherwise, this is empty.
  std::vector<HloInstruction*> async_instructions_;

  // Module containing this computation.
  HloModule* parent_ = nullptr;

  // Store instructions in std::list as they can be added and removed
  // arbitrarily and we want a stable iteration order. Keep a map from
  // instruction pointer to location in the list for fast lookup.
  InstructionList instructions_;
  absl::flat_hash_map<const HloInstruction*, InstructionList::iterator>
      instruction_iterators_;

  // Removed instructions are moved into to_be_deleted_ first and then
  // deallocated when Cleanup is called.
  std::vector<std::unique_ptr<HloInstruction>> to_be_deleted_;

  std::vector<HloInstruction*> param_instructions_;

  HloComputation(const HloComputation&) = delete;
  HloComputation& operator=(const HloComputation&) = delete;
};

template <typename HloInstructionPtr>
Status HloComputation::Accept(
    DfsHloVisitorBase<HloInstructionPtr>* visitor) const {
  // Visit unreachable roots. Beware that the visitor might delete the currently
  // visited root, which would invalidate iterators if the unreachable roots
  // weren't computed ahead of time.
  for (HloInstruction* root : CollectUnreachableRoots()) {
    VLOG(3) << "Traversing unreachable root: " << root->ToString();
    // Call FinishVisit only at the end.
    TF_RETURN_IF_ERROR(root->Accept(visitor, /*call_finish_visit=*/false));
  }
  // Visit the computation root instruction last.
  return root_instruction()->Accept(visitor, /*call_finish_visit=*/true);
}

// Explicit instantiations.
template Status HloComputation::Accept(DfsHloVisitor* visitor) const;
template Status HloComputation::Accept(ConstDfsHloVisitor* visitor) const;

template <typename HloInstructionPtr>
Status HloComputation::AcceptOrdered(
    DfsHloVisitorBase<HloInstructionPtr>* visitor,
    absl::Span<HloInstruction* const> order) const {
  VLOG(3) << "Accepting visitor with order.";
  for (HloInstruction* root : CollectUnreachableRoots()) {
    TF_RET_CHECK(absl::c_linear_search(order, root)) << root->ToString();
  }
  TF_RET_CHECK(order.size() == instruction_count());
  absl::flat_hash_set<const HloInstruction*> visited;
  for (const HloInstruction* instruction : order) {
    VLOG(3) << "Visiting ordered: " << instruction->ToString();
    TF_RET_CHECK(instruction_iterators_.contains(instruction))
        << "Instruction " << instruction->name() << " is not in computation "
        << name();
    TF_RET_CHECK(!visited.contains(instruction))
        << "Instruction " << instruction->name()
        << " appears more than once in order";
    HloInstruction* mutable_instruction =
        const_cast<HloInstruction*>(instruction);
    TF_RETURN_IF_ERROR(visitor->Preprocess(mutable_instruction));
    TF_RETURN_IF_ERROR(mutable_instruction->Visit(visitor));
    visitor->SetVisited(*mutable_instruction);
    TF_RETURN_IF_ERROR(visitor->Postprocess(mutable_instruction));
    visited.insert(instruction);
  }
  TF_RETURN_IF_ERROR(visitor->FinishVisit(root_instruction()));
  return OkStatus();
}

// Explicit instantiations.
template Status HloComputation::AcceptOrdered(
    DfsHloVisitor*, absl::Span<HloInstruction* const>) const;
template Status HloComputation::AcceptOrdered(
    ConstDfsHloVisitor*, absl::Span<HloInstruction* const>) const;

}  // namespace xla

#endif  // TENSORFLOW_COMPILER_XLA_SERVICE_HLO_COMPUTATION_H_
