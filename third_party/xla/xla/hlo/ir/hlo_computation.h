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

#ifndef XLA_HLO_IR_HLO_COMPUTATION_H_
#define XLA_HLO_IR_HLO_COMPUTATION_H_

#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "absl/algorithm/container.h"
#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "absl/container/inlined_vector.h"
#include "absl/functional/function_ref.h"
#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/cord.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "xla/hlo/ir/dfs_hlo_visitor.h"
#include "xla/hlo/ir/hlo_clone_context.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/hlo/ir/ptrvec.h"
#include "xla/iterator_util.h"
#include "xla/printer.h"
#include "xla/service/hlo.pb.h"
#include "xla/service/name_uniquer.h"
#include "xla/shape.h"
#include "xla/shape_tree.h"
#include "xla/shape_util.h"
#include "xla/status_macros.h"
#include "xla/util.h"
#include "xla/xla_data.pb.h"
#include "tsl/lib/gtl/iterator_range.h"
#include "tsl/platform/errors.h"

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
  using InstructionList = std::vector<HloInstructionInfo>;

  // Builder class for HloComputation.
  class Builder {
   public:
    explicit Builder(absl::string_view name) : name_(name) {}
    Builder(Builder&& b) = default;
    virtual ~Builder() = default;

    // Build and return an HloComputation. The parameter root_instruction
    // specifies the already-added instruction to use as the root. If
    // root_instruction is nullptr then use the last added instruction as the
    // root.
    std::unique_ptr<HloComputation> Build(
        HloInstruction* root_instruction = nullptr);

    // Add the instruction to be part of this computation.
    // If the new instruction is derived from another one,
    // you probably want to do
    // `original_inst->AddInstruction(new_inst)` instead.
    virtual HloInstruction* AddInstruction(
        std::unique_ptr<HloInstruction> instruction) {
      auto* added_instruction = instruction.get();
      instructions_.push_back(std::move(instruction));
      return added_instruction;
    }

    HloInstruction* AddInstruction(std::unique_ptr<HloInstruction> instruction,
                                   std::optional<absl::string_view> new_name) {
      instruction->SetAndSanitizeName(new_name.value());
      return AddInstruction(std::move(instruction));
    }

    absl::StatusOr<HloInstruction*> AddParameter(
        std::unique_ptr<HloInstruction> parameter) {
      if (!parameter_numbers_.insert(parameter->parameter_number()).second) {
        return Internal("Duplicate parameter number %d",
                        parameter->parameter_number());
      }
      return AddInstruction(std::move(parameter));
    }

    absl::Status ForEachInstruction(
        absl::FunctionRef<absl::Status(const HloInstruction*)> func) const {
      for (const auto& instruction : instructions_) {
        TF_RETURN_IF_ERROR(func(instruction.get()));
      }
      return absl::OkStatus();
    }

    HloInstruction* last_added_instruction() const {
      return instructions_.empty() ? nullptr : instructions_.back().get();
    }

   private:
    const std::string name_;
    std::vector<std::unique_ptr<HloInstruction>> instructions_;
    absl::flat_hash_set<int> parameter_numbers_;

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

  // Helper class for returning the instruction post order for a computation,
  // but maintaining a cache to avoid repeated calls to
  // computation->MakeInstructionPostorder().  The cache is invalidated if
  // RecordChange(<something evaluating to true>)  is called.
  //
  // This class can be handy to avoid recomputing the instruction post order
  // when an optimization pass wants to make multiple passes over the
  // instructions.
  //
  // Example usage:
  //   for (auto* computation : module->computations(execution_threads)) {
  //     HloComputation::CachingPostOrder cpo(computation);
  //     for (auto instruction : cpo.PostOrder()) {  // Pass 1
  //       bool did_change = ... maybe do something to instruction ...;
  //       cpo.RecordChange(did_change);
  //     }
  //     for (auto instruction : cpo.PostOrder()) {  // Pass 2
  //       bool did_change = ... maybe do something else to instruction ...;
  //       cpo.RecordChange(did_change);
  //     }
  //   }
  class CachingPostOrder {
   public:
    explicit CachingPostOrder(const HloComputation* computation)
        : computation_(computation), recompute_(true) {}

    // Returns the instruction post-order for "computation"
    const std::vector<HloInstruction*>& PostOrder() {
      if (recompute_) {
        cached_post_order_ = computation_->MakeInstructionPostOrder();
        recompute_ = false;
      }
      return cached_post_order_;
    }

    void RecordChange(bool changed) { recompute_ |= changed; }

   private:
    const HloComputation* computation_;
    bool recompute_;
    std::vector<HloInstruction*> cached_post_order_;
  };

  ~HloComputation();

  // Add an instruction to the computation. The computation takes ownership of
  // the instruction.
  HloInstruction* AddInstruction(std::unique_ptr<HloInstruction> instruction,
                                 absl::string_view new_name = "");

  HloInstruction* AddInstruction(std::unique_ptr<HloInstruction> instruction,
                                 const OpMetadata* metadata);

  HloInstruction* AddInstruction(std::unique_ptr<HloInstruction> instruction,
                                 const OpMetadata* metadata,
                                 const FrontendAttributes* frontend_attributes);

  // Replace the old parameter at index param_no with
  // `instruction`. Updates uses and root instruction. Removes old
  // instruction from computation. No check is done on the shape.
  HloInstruction* ReplaceParameter(int64_t param_no,
                                   std::unique_ptr<HloInstruction> instruction);

  // Remove the param_no'th parameter from the computation.
  // Note this is only applicatable to the computation for the fusion
  // instruction.
  absl::Status RemoveParameter(int64_t param_no);

  // Remove unused parameters from the computation.
  // Note this is only applicatable to the computation for the fusion
  // instruction.
  absl::Status RemoveUnusedParametersFromFusedComputation();

  // Remove unused parameters from the computation. Unlike
  // RemoveUnusedParametersFromFusedComputation, this function can be used
  // to remove parameters from non-fusion computations.
  absl::Status RemoveUnusedParametersFromAnyComputation();

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
  absl::Status ReplaceEntryComputationParameter(
      int64_t param_no, HloInstruction* old_instruction,
      std::unique_ptr<HloInstruction> instruction);

  // Remove an instruction from the computation. The instruction must have no
  // users. Instruction is deallocated with this call.
  absl::Status RemoveInstruction(HloInstruction* instruction);

  // Removes an instruction from the computation. The instruction must have no
  // users. Instruction is deallocated with this call. The instruction will be
  // removed even if it is marked as not removable.
  absl::Status ForceRemoveInstruction(HloInstruction* instruction);

  // Remove an instruction (including side effecting ones) from the computation
  // and also transitively any operand that has no side effect and no users post
  // removing an instruction. The instruction must have no users. Instruction is
  // deallocated with this call. If given, the cleanup routine is executed on a
  // removed instruction before its deallocation. If ignore_control_dependencies
  // is set to true, if will remove the unused operands even when they have
  // control dependencies, and transitively pass the control dependencies from
  // the predecessors to the succesors of the removed instructions, so that the
  // logical exeuction order of the remaining unremoved instructions are
  // preserved.
  absl::Status RemoveInstructionAndUnusedOperands(
      HloInstruction* instruction,
      std::optional<absl::FunctionRef<void(HloInstruction*)>> cleanup =
          std::nullopt,
      bool ignore_control_dependencies = false);

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

  const HloInstruction::InstructionVector& parameter_instructions() const {
    return param_instructions_;
  }

  absl::string_view name() const { return name_; }

  // Sets the string identifier for this computation. Name will be sanitized to
  // match the regexp "[a-zA-Z_][a-zA-Z0-9_.-]*".
  //
  // See also HloModule::SetAndUniquifyComputationName(), which does this plus
  // UniqufyName().
  void SetAndSanitizeName(absl::string_view name) {
    name_ = NameUniquer::GetSanitizedName(name);
  }

  // Use the given NameUniquer to select a unique name for the computation based
  // on the computation's existing name.
  //
  // See also HloModule::SetAndUniquifyComputationName(), which does this plus
  // SetAndSanitizeName().
  void UniquifyName(NameUniquer* name_uniquer);

  // Use the given `module` to select a unique name for this computation based
  // on computation's existing name.
  void UniquifyName(HloModule* module);

  // Prints a string representation of the computation.
  //
  // (We express the default options using an overload rather than a default
  // param because gdb ignores default params, but does resolve overloads.)
  void Print(Printer* printer) const {
    return Print(printer, HloPrintOptions::Default());
  }
  void Print(Printer* printer, const HloPrintOptions& options) const;
  void Print(Printer* printer, const HloPrintOptions& options,
             absl::Span<const HloInstruction* const> instruction_order) const;

  // Return a string representation of the computation.
  //
  // (We express the default options using an overload rather than a default
  // param because gdb ignores default params, but does resolve overloads.)
  std::string ToString() const;
  std::string ToString(const HloPrintOptions& options) const;

  // Overload which accepts an order to emit the instructions in.
  std::string ToString(
      const HloPrintOptions& options,
      absl::Span<const HloInstruction* const> instruction_order) const;

  // Returns a Cord representation of the computation.
  //
  // (We express the default options using an overload rather than a default
  // param because gdb ignores default params, but does resolve overloads.)

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
  static absl::StatusOr<std::unique_ptr<HloComputation>> CreateFromProto(
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

  using InstructionSequence = tsl::gtl::iterator_range<
      UnwrappingIterator<HloInstructionList::iterator>>;

  using ConstInstructionSequence = tsl::gtl::iterator_range<
      UnwrappingIterator<HloInstructionList::const_iterator>>;

  // Gets the instructions in this computation.
  //
  // The returned type is a range of HloInstruction*s, so you can iterate over
  // it using a range-based for loop in the natural way:
  //
  //   for (HloInstruction* instr : computation->instructions()) { ... }
  //

  tsl::gtl::iterator_range<xla::HloInstructionUnwrappingConstIterator>
  instructions() const {
    const int end = instructions_.size();
    return {HloInstructionUnwrappingConstIterator(
                HloInstructionConstIterator(&instructions_, 0, end)),
            HloInstructionUnwrappingConstIterator(
                HloInstructionConstIterator(&instructions_, end, end))};
  }
  tsl::gtl::iterator_range<xla::HloInstructionUnwrappingIterator>
  instructions() {
    const int end = instructions_.size();
    return {HloInstructionUnwrappingIterator(
                HloInstructionIterator(&instructions_, 0, end)),
            HloInstructionUnwrappingIterator(
                HloInstructionIterator(&instructions_, end, end))};
  }
  tsl::gtl::iterator_range<HloInstructionIterator> instructions_with_info() {
    const int end = instructions_.size();
    return {HloInstructionIterator(&instructions_, 0, end),
            HloInstructionIterator(&instructions_, end, end)};
  }
  tsl::gtl::iterator_range<HloInstructionConstIterator> instructions_with_info()
      const {
    const int end = instructions_.size();
    return {HloInstructionConstIterator(&instructions_, 0, end),
            HloInstructionConstIterator(&instructions_, end, end)};
  }

  using ChannelDependencies =
      absl::flat_hash_map<const HloInstruction*,
                          absl::InlinedVector<HloInstruction*, 1>>;

  // Compute and return a post-order of the instructions in the computation. In
  // this order, definitions of values always appear before their uses.
  std::vector<HloInstruction*> MakeInstructionPostOrder() const;
  // Same as MakeInstructionPostOrder but starting at any instruction in the
  // computation, not just the root. Describes the corresponding subgraph.
  std::vector<HloInstruction*> MakeInstructionPostOrderFrom(
      HloInstruction&) const;
  std::vector<HloInstruction*> MakeInstructionPostOrder(
      const ChannelDependencies& channel_dependencies) const;
  // Same as MakeInstructionPostOrder but with special tie-breaking behavior.
  // Specifically, when ties (in ordering) between instructions occur, Reshapes
  // will be sorted before other operations.
  std::vector<HloInstruction*> MakeInstructionPostOrderWithReshapeFirst() const;

  // Calls `func` with each instruction in the computation in post-order.
  void ForEachInstructionPostOrder(
      absl::FunctionRef<void(HloInstruction*)> func) const;

  int64_t instruction_count() const { return instruction_count_; }

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

  // Creates a call instruction containing the given instructions. Instructions
  // must be in reverse topological order (root of the called computation
  // first). Replaces all uses of the original root instruction with the call
  // instruction. The original instructions are removed if they have no uses
  // after creating the call (this is necessarily true for at least the root).
  HloInstruction* CreateCallInstruction(
      absl::Span<HloInstruction* const> instructions_to_call);

  // Creates a composite call instruction containing the given instructions.
  // Instructions must be in reverse topological order (root of the called
  // computation first). Replaces all uses of the original root instruction with
  // the composite call instruction. The original instructions are removed if
  // they have no uses after creating the composite call (this is necessarily
  // true for at least the root).
  HloInstruction* CreateCompositeCallInstruction(
      absl::Span<HloInstruction* const> instructions_to_call,
      const std::string& name, const std::string& attributes, int64_t version);

  // Creates an async start/done instruction pair where instruction is wrapped
  // inside an asynchronous computation. The context shapes are appended to the
  // output tuple of the asynchronous start which is backend specific. Returns
  // the async done instruction. The new async start instruction is the operand
  // of the async done instruction so that can be accessed using that. If
  // present, `async_execution_thread` will be attached to the
  // async-start/update/done instructions as well as wrapped computations.
  // If `replace` is true, replace instruction with the async done instruction.
  // If `override_names` is true, the clone on `instruction` and the async op
  // created will get non-default names.
  absl::StatusOr<HloInstruction*> CreateAsyncInstructions(
      HloInstruction* instruction, absl::Span<const Shape> context_shapes,
      absl::string_view async_execution_thread =
          HloInstruction::kMainExecutionThread,
      bool replace = true, bool override_names = false);

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
  absl::StatusOr<HloInstruction*> DeepCopyInstruction(
      HloInstruction* instruction,
      const ShapeTree<bool>* indices_to_copy = nullptr,
      ShapeTree<HloInstruction*>* copies_added = nullptr);

  // As above, but uses a custom function to copy the leaf nodes, which could
  // create alternative HLOs other than kCopy, or even pass-throughs.
  absl::StatusOr<HloInstruction*> DeepCopyInstructionWithCustomCopier(
      HloInstruction* instruction,
      absl::FunctionRef<HloInstruction*(HloInstruction* leaf,
                                        const ShapeIndex& leaf_index,
                                        HloComputation* computation)>
          copy_leaf);

  // Computes and returns the ProgramShape of this computation (shape of
  // parameters and result with layout).
  ProgramShape ComputeProgramShape(bool include_ids = true) const;

  // Return whether `*this` and `other` are functionally equivalent.
  bool Equal(
      const HloComputation& other, bool is_layout_sensitive,
      std::optional<
          absl::FunctionRef<bool(const HloComputation*, const HloComputation*)>>
          computations_comparator = std::nullopt) const {
    return EqualInternal(other, is_layout_sensitive, computations_comparator,
                         /*ignore_channel_id_values=*/false,
                         /*ignore_execution_thread=*/false);
  }

  // Same as Equal() but ignores channel ID value mismatches on instructions, as
  // long as the two instructions both have channel IDs or neither has a channel
  // ID.
  bool EqualIgnoringChannelIdValues(
      const HloComputation& other, bool is_layout_sensitive,
      std::optional<
          absl::FunctionRef<bool(const HloComputation*, const HloComputation*)>>
          computations_comparator = std::nullopt) const {
    return EqualInternal(other, is_layout_sensitive, computations_comparator,
                         /*ignore_channel_id_values=*/true,
                         /*ignore_execution_thread=*/false);
  }

  bool EqualIgnoringExecutionThread(
      const HloComputation& other, bool is_layout_sensitive,
      bool ignore_channel_id_values,
      std::optional<
          absl::FunctionRef<bool(const HloComputation*, const HloComputation*)>>
          computations_comparator = std::nullopt) const {
    return EqualInternal(other, is_layout_sensitive, computations_comparator,
                         ignore_channel_id_values,
                         /*ignore_execution_thread=*/true);
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
  absl::Status ReplaceWithNewInstruction(
      HloInstruction* old_instruction,
      std::unique_ptr<HloInstruction> new_instruction);

  // Replaces an old instruction with a newly created instruction, and adds the
  // new instruction as an entry computation's parameter. Removes old
  // instruction from computation. Updates uses and root instruction.
  absl::Status ReplaceWithNewEntryComputationParameter(
      HloInstruction* old_instruction,
      std::unique_ptr<HloInstruction> new_instruction);

  // Replace old instruction with new instruction.  Updates uses and root
  // instruction. Removes old instruction from computation. Transitively removes
  // non-side effecting operands of old instruction that no longer have users,
  // similar to RemoveInstructionAndUnusedOperands(). Precondition:
  // old_instruction and new_instruction must have the compatible shapes.
  // If preserve_sharding is true, the replacement will fail if both new and old
  // instruction have sharding that is not compatible, and the function will
  // return false. Otherwise, when the replacement happens, if |new_instruction|
  // doesn't have any sharding information it will receive the sharding
  // information of |old_instruction|, and function will return true.
  absl::StatusOr<bool> ReplaceInstruction(HloInstruction* old_instruction,
                                          HloInstruction* new_instruction,
                                          bool preserve_sharding,
                                          bool relay_control_dependency = false,
                                          bool remove_unused_operands = true);

  // Same as above, with preserve_sharding=false. Since this replacement always
  // happens, it returns just a absl::Status as opposed to absl::StatusOr<bool>
  absl::Status ReplaceInstruction(HloInstruction* old_instruction,
                                  HloInstruction* new_instruction);

  // Same as ReplaceInstruction, but the new instruction can have a different
  // shape.
  absl::StatusOr<bool> ReplaceInstructionWithDifferentShape(
      HloInstruction* old_instruction, HloInstruction* new_instruction,
      bool preserve_sharding, bool relay_control_dependency = false,
      bool remove_unused_operands = true);
  absl::Status ReplaceInstructionWithDifferentShape(
      HloInstruction* old_instruction, HloInstruction* new_instruction);

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
  absl::Status Accept(DfsHloVisitorBase<HloInstructionPtr>* visitor) const;

  // Same as Accept() above, but the order of operand and control predecessor
  // visitation is determined by the given operand order; if compare(A, B) ==
  // true, A is visited before B.
  absl::Status AcceptWithOperandOrder(
      DfsHloVisitor* visitor,
      const HloInstruction::CompareFunction& operand_order) const;

  // Visit every node in the computation in the given order. 'order' must
  // be a topological sort of all instructions in the computation.
  template <typename HloInstructionPtr>
  absl::Status AcceptOrdered(DfsHloVisitorBase<HloInstructionPtr>* visitor,
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

  // Like CloneWithReplacements(), but this is a const method and `context` must
  // be specified.
  std::unique_ptr<HloComputation> CloneInContext(
      HloCloneContext& context,
      const absl::flat_hash_map<const HloInstruction*,
                                std::unique_ptr<HloInstruction>>* replacements =
          nullptr,
      absl::Span<const HloInstruction* const> extra_parameters = {},
      const std::string& suffix = "clone",
      const HloInstruction* new_root = nullptr) const;

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
  bool IsSafelyRemovable(const HloInstruction* instruction,
                         bool ignore_control_dependency = false);

  // Returns a map from an instruction to the group of instructions associated
  // with the same channel. These instructions will be considered as a single
  // node for dependency purposes.
  // RecvDone ops will map to the corresponding Send op.
  // Cross-partition collectives will map to every other instruction with the
  // same channel ID (it doesn't map to itself).
  ChannelDependencies ComputeChannelDependencies() const;

  // Returns true if this computation has a side effect. A computation has a
  // side effect if it contains one or more instructions with a side effect.
  bool HasSideEffect() const;

  // Returns if this computation is a fusion computation.
  // Do not use this method to determine if fusion_instruction_ != nullptr.
  // Instead, directly do: FusionInstruction() != nullptr
  bool IsFusionComputation() const {
    return instruction_type() == InstructionType::kFusion;
  }

  // Returns if this computation is the entry computation of the module.
  bool IsEntryComputation() const;

  // Returns the owning fusion instruction, or nullptr if this is not a fusion
  // computation.
  HloInstruction* FusionInstruction() const {
    return instruction_type() == InstructionType::kFusion ? instruction()
                                                          : nullptr;
  }
  void SetFusionInstruction(HloInstruction* fusion_instruction) {
    SetInstruction(fusion_instruction, InstructionType::kFusion);
  }

  // Returns if this computation is a custom-call computation.
  bool IsCustomCallComputation() const {
    return instruction_type() == InstructionType::kCustomCall;
  }

  // Returns the owning custom call instruction, or nullptr if this is not a
  // custom call computation.
  HloInstruction* CustomCallInstruction() const {
    return instruction_type() == InstructionType::kCustomCall ? instruction()
                                                              : nullptr;
  }
  void SetCustomCallInstruction(HloInstruction* custom_call_instruction) {
    SetInstruction(custom_call_instruction, InstructionType::kCustomCall);
  }

  // Returns if this computation is a to_apply region of a collective.
  bool IsCollectiveCalledComputation() const {
    return instruction_type() == InstructionType::kCollective;
  }

  // Returns the owning collective call instruction, or nullptr if this is not a
  // collective call computation.
  HloInstruction* CollectiveCallInstruction() const {
    return instruction_type() == InstructionType::kCollective ? instruction()
                                                              : nullptr;
  }

  void SetCollectiveCallInstruction(
      HloInstruction* collective_call_instruction) {
    SetInstruction(collective_call_instruction, InstructionType::kCollective);
  }

  // Returns if this computation is a body computation of a while.
  bool IsWhileBodyComputation() const {
    return instruction_type() == InstructionType::kWhile;
  }

  // Returns the owning while call instruction, or nullptr if this is not a
  // while call body computation.
  HloInstruction* WhileCallInstruction() const {
    return instruction_type() == InstructionType::kWhile ? instruction()
                                                         : nullptr;
  }

  void SetWhileCallInstruction(HloInstruction* while_call_instruction) {
    CHECK(while_call_instruction != nullptr);
    CHECK(while_call_instruction->opcode() == HloOpcode::kWhile);
    SetInstruction(while_call_instruction, InstructionType::kWhile);
  }

  // Returns if this computation is a branch computation of a conditional.
  bool IsConditionalBranchComputation() const {
    return instruction_type() == InstructionType::kConditional;
  }

  // Returns the owning conditional call instruction, or nullptr if this is not
  // a conditional branch computation.
  HloInstruction* ConditionalCallInstruction() const {
    return instruction_type() == InstructionType::kConditional ? instruction()
                                                               : nullptr;
  }

  void SetConditionalCallInstruction(
      HloInstruction* conditional_call_instruction) {
    CHECK(conditional_call_instruction != nullptr);
    CHECK(conditional_call_instruction->opcode() == HloOpcode::kConditional);
    SetInstruction(conditional_call_instruction, InstructionType::kConditional);
  }

  // Returns if this computation is an async computation.
  bool IsAsyncComputation() const { return async_start_ != nullptr; }

  // Returns the owning async instruction. It's nullptr if this is not an async
  // computation.
  HloInstruction* AsyncStart() const { return async_start_; }

  void AddAsyncStart(HloInstruction* async_instruction) {
    // TODO: Add instruction type for async instructions.
    CHECK(instruction_type() == InstructionType::kUnset);
    CHECK(async_instruction->opcode() == HloOpcode::kAsyncStart);
    async_start_ = async_instruction;
  }

  void RemoveAsyncStart() { async_start_ = nullptr; }

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

  void SetExecutionThread(absl::string_view execution_thread) {
    execution_thread_ = std::string(execution_thread);
  }

  absl::string_view execution_thread() const { return execution_thread_; }
  // Returns true if this computation is annotated on "main" execution thread.
  bool IsMainThread() const {
    return execution_thread_ == HloInstruction::kMainExecutionThread;
  }

  // Deallocates instructions that are marked by "RemoveInstruction" and
  // compacts the instructions_ vector by removing the deleted instructions'
  // entries (a.k.a. tombstones).
  // This two-stage clean up process is designed such that HloPass can have
  // stable internal pointers to HloInstructions while we create and remove
  // HloInstructions in a pass.
  // Note: the removal operation is stable because some users depend on it.
  void Cleanup();

  // Returns true if a given instruction is marked dead in this computation.
  bool IsMarkedAsDead(const HloInstruction* inst);

  // Returns true iff this computation can be inlined as a single instruction.
  bool CanExpandIntoSingleInstruction() const;

 private:
  explicit HloComputation(
      const std::string& name, int parameter_count,
      std::vector<std::unique_ptr<HloInstruction>>* instructions,
      HloInstruction* root_instruction);

  // Internal helper for adding instructions.
  HloInstruction* AddInstructionInternal(
      std::unique_ptr<HloInstruction> instruction);

  // Internal helper for comparison with different options.
  bool EqualInternal(
      const HloComputation& other, bool is_layout_sensitive,
      std::optional<
          absl::FunctionRef<bool(const HloComputation*, const HloComputation*)>>
          computations_comparator,
      bool ignore_channel_id_values, bool ignore_execution_thread) const;
  // Appends (fuses) HLOs in instructions_to_append into the called computation
  // of the caller.
  void AppendInstructionsIntoCalledComputation(
      absl::Span<HloInstruction* const> instructions_to_append,
      HloInstruction* caller);

  // Internal helper for recursive copying of an instruction. Creates and
  // returns a deep copy of the given instruction.
  absl::StatusOr<HloInstruction*> DeepCopyHelper(
      HloInstruction* instruction, ShapeIndex* index,
      absl::FunctionRef<HloInstruction*(HloInstruction* leaf,
                                        const ShapeIndex& leaf_index,
                                        HloComputation* computation)>
          copy_leaf);

  // Internal helper to collect unreachable roots.
  std::vector<HloInstruction*> CollectUnreachableRoots() const;

  class VisitMap;
  void ComputeInstructionPostOrder(
      HloInstruction* root, const ChannelDependencies& channel_dependencies,
      VisitMap& visited, std::vector<HloInstruction*>& post_order,
      std::vector<HloInstruction*>* dfs_stack_scratch) const;

  void ForEachInstructionPostOrderImpl(
      absl::FunctionRef<void(HloInstruction*)> func, HloInstruction* root,
      const ChannelDependencies& channel_dependencies, VisitMap& visited,
      std::vector<HloInstruction*>* dfs_stack_scratch) const;

  absl::Status RemoveUnusedParametersImpl(bool allow_non_fusion);

  absl::Status RemoveInstructionImpl(HloInstruction* instruction,
                                     bool ignore_safety_check);

  enum class InstructionType : uint8_t {
    kUnset,
    // This computation is a fusion computation. A fusion computation ordinarily
    // also has a non-null instruction. However, if a fusion instruction
    // is removed during compilation, the fusion computation becomes
    // unreachable, and its instruction is set to null. We still need to regard
    // such computations as fusion computations for HLO scheduling purposes.
    kFusion,
    // This computation is a custom-call computation.
    kCustomCall,
    // This computation is a collective computation.
    kCollective,
    // This computation is a while body computation.
    kWhile,
    // This computation is a conditional branch computation.
    kConditional,
  };
  static constexpr uintptr_t kInstructionTypeMask = 0b111;
  static_assert(static_cast<int>(InstructionType::kUnset) == 0,
                "kUnset must be 0.");

  InstructionType instruction_type() const {
    return static_cast<InstructionType>(instruction_and_type_ &
                                        kInstructionTypeMask);
  }

  HloInstruction* instruction() const {
    return reinterpret_cast<HloInstruction*>(instruction_and_type_ &
                                             ~kInstructionTypeMask);
  }

  void SetInstruction(HloInstruction* instruction, InstructionType type);

  int64_t unique_id_;
  HloInstruction* root_instruction_;

  // Module containing this computation.
  HloModule* parent_ = nullptr;

  // Contains HloInstruction* and its type.
  // The respective type in the least significant three bits.
  uintptr_t instruction_and_type_ = 0;

  // If this computation is an async computation, this field points to the
  // first async instruction (async-start) in the asynchronous op chain that
  // calls this computation.
  // Otherwise, this is empty.
  HloInstruction* async_start_ = nullptr;

  HloInstruction::InstructionVector param_instructions_;

  // Store instructions in std::vector as they can be added and removed
  // arbitrarily and we want a stable iteration order.
  // For the reverse mapping we use HloInstruction::index_in_parent_.
  //
  // Note: removals from this vector must be stable because some users depend on
  // it. See the Cleanup() method for details on the two-stage removal process.
  HloInstructionList instructions_;

  // Number of not-marked-for-deletion entries in instructions_.
  int64_t instruction_count_;

  // Removed instructions are moved into to_be_deleted_ first and then
  // deallocated when Cleanup is called.
  PtrVec<HloInstruction*> to_be_deleted_;

  // Execution thread of this computation. By default, it's main thread.
  std::string execution_thread_ = HloInstruction::kMainExecutionThread;

  std::string name_;

  HloComputation(const HloComputation&) = delete;
  HloComputation& operator=(const HloComputation&) = delete;
};

template <typename HloInstructionPtr>
absl::Status HloComputation::Accept(
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
template absl::Status HloComputation::Accept(DfsHloVisitor* visitor) const;
template absl::Status HloComputation::Accept(ConstDfsHloVisitor* visitor) const;

template <typename HloInstructionPtr>
absl::Status HloComputation::AcceptOrdered(
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
  return visitor->FinishVisit(root_instruction());
}

// Explicit instantiations.
template absl::Status HloComputation::AcceptOrdered(
    DfsHloVisitor*, absl::Span<HloInstruction* const>) const;
template absl::Status HloComputation::AcceptOrdered(
    ConstDfsHloVisitor*, absl::Span<HloInstruction* const>) const;

}  // namespace xla

#endif  // XLA_HLO_IR_HLO_COMPUTATION_H_
