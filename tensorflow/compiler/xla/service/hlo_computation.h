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
#include <unordered_set>
#include <utility>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "absl/types/span.h"
#include "tensorflow/compiler/xla/iterator_util.h"
#include "tensorflow/compiler/xla/map_util.h"
#include "tensorflow/compiler/xla/service/dfs_hlo_visitor.h"
#include "tensorflow/compiler/xla/service/dfs_hlo_visitor_with_default.h"
#include "tensorflow/compiler/xla/service/hlo.pb.h"
#include "tensorflow/compiler/xla/service/hlo_clone_context.h"
#include "tensorflow/compiler/xla/service/hlo_instruction.h"
#include "tensorflow/compiler/xla/service/name_uniquer.h"
#include "tensorflow/compiler/xla/shape_tree.h"
#include "tensorflow/compiler/xla/statusor.h"
#include "tensorflow/compiler/xla/types.h"
#include "tensorflow/compiler/xla/xla_data.pb.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/platform/macros.h"
#include "tensorflow/core/platform/types.h"

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
  // Builder class for HloComputation.
  class Builder {
   public:
    explicit Builder(const string& name,
                     HloInstruction* fusion_instruction = nullptr)
        : name_(name),
          last_added_instruction_(nullptr),
          fusion_instruction_(fusion_instruction) {}

    // Build and return an HloComputation. The parameter root_instruction
    // specifies the already-added instruction to use as the root. If
    // root_instruction is nullptr then use the last added instruction as the
    // root.
    std::unique_ptr<HloComputation> Build(
        HloInstruction* root_instruction = nullptr);

    HloInstruction* AddInstruction(
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
      return Status::OK();
    }

   private:
    const string name_;
    HloInstruction* last_added_instruction_;
    HloInstruction* fusion_instruction_;
    std::vector<std::unique_ptr<HloInstruction>> instructions_;
  };

  // Add an instruction to the computation. The computation takes ownership of
  // the instruction.
  HloInstruction* AddInstruction(std::unique_ptr<HloInstruction> instruction);

  // Remove the param_no'th parameter from the computation.
  // Note this is only applicatable to the computation for the fusion
  // instruction.
  Status RemoveParameter(int64 param_no);

  // Remove unused parameters from the computation.
  // Note this is only applicatable to the computation for the fusion
  // instruction.
  Status RemoveUnusedParameters();

  // Add new parameter instruction to the computation.
  // This should be a new parameter. Instruction will be appended to parameters
  // and inserted to the instruction list.
  HloInstruction* AddParameter(std::unique_ptr<HloInstruction> instruction);

  // Remove an instruction from the computation. The instruction must have no
  // users. Instruction is deallocated with this call.
  Status RemoveInstruction(HloInstruction* instruction);

  // Remove an instruction (including side effecting ones) from the computation
  // and also transitively any operand that has no side effect and no users post
  // removing an instruction. The instruction must have no users. Instruction is
  // deallocated with this call.
  Status RemoveInstructionAndUnusedOperands(HloInstruction* instruction);

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
  int64 num_parameters() const { return param_instructions_.size(); }

  // Returns the parameter instruction for the given parameter number.
  HloInstruction* parameter_instruction(int64 param_no) const {
    CHECK_GE(param_no, 0);
    CHECK_LT(param_no, static_cast<int64>(param_instructions_.size()))
        << "Computation " << name() << " has no parameter number " << param_no;
    return param_instructions_[param_no];
  }

  const std::vector<HloInstruction*>& parameter_instructions() const {
    return param_instructions_;
  }

  const string& name() const { return name_; }

  // Use the given NameUniquer to select a unique name for the computation based
  // on the computation's existing name.
  void UniquifyName(NameUniquer* name_uniquer);

  // Return a string representation of the computation.
  //
  // (We express the default options using an overload rather than a default
  // param because gdb ignores default params, but does resolve overloads.)
  string ToString() const { return ToString(HloPrintOptions()); }
  string ToString(const HloPrintOptions& options) const;

  // Overload which accepts an order to emit the instructions in.
  string ToString(
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
      const absl::flat_hash_map<int64, HloComputation*>& computation_map);

  // Gets the instructions in this computation.
  //
  // The returned type is a range of HloInstruction*s, so you can iterate over
  // it using a range-based for loop in the natural way:
  //
  //   for (HloInstruction* instr : computation->instructions()) { ... }
  //
  tensorflow::gtl::iterator_range<UnwrappingIterator<
      std::list<std::unique_ptr<HloInstruction>>::const_iterator>>
  instructions() const {
    return {MakeUnwrappingIterator(instructions_.begin()),
            MakeUnwrappingIterator(instructions_.end())};
  }
  tensorflow::gtl::iterator_range<
      UnwrappingIterator<std::list<std::unique_ptr<HloInstruction>>::iterator>>
  instructions() {
    return {MakeUnwrappingIterator(instructions_.begin()),
            MakeUnwrappingIterator(instructions_.end())};
  }

  // Compute and return a post-order of the instructions in the computation. In
  // this order, definitions of values always appear before their uses.
  std::vector<HloInstruction*> MakeInstructionPostOrder() const;

  int64 instruction_count() const { return instruction_iterators_.size(); }

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
  ProgramShape ComputeProgramShape() const;

  // Return whether `*this` and `other` are functionally equivalent.
  bool operator==(const HloComputation& other) const;

  // Replaces old instruction with newly created instruction. Removes old
  // instruction from computation. Updates uses and root instruction.
  Status ReplaceWithNewInstruction(
      HloInstruction* old_instruction,
      std::unique_ptr<HloInstruction> new_instruction);

  // Replace old instruction with new instruction.  Updates uses and root
  // instruction. Removes old instruction from computation. Precondition:
  // old_instruction and new_instruction must have the compatible shapes.
  Status ReplaceInstruction(HloInstruction* old_instruction,
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

  // Same as Accept() above, but the visitor is given as a function.
  Status Accept(const std::function<Status(HloInstruction*)>& visitor_func);
  Status Accept(
      const std::function<Status(const HloInstruction*)>& visitor_func) const;

  // Returns a deep copy of this computation including all instructions.
  // If the clone context is specified, it will be populated with the cloned
  // object mappings, and its module() will be used to add new computations
  // into.
  std::unique_ptr<HloComputation> Clone(const string& suffix = "clone",
                                        HloCloneContext* context = nullptr);

  // Like Clone(), but if an instruction is present in replacement_map, we use
  // the map's value to replace that instruction in the cloned computation.
  //
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
      absl::flat_hash_map<const HloInstruction*,
                          std::unique_ptr<HloInstruction>>
          replacements,
      absl::Span<const HloInstruction* const> extra_parameters = {},
      HloCloneContext* context = nullptr, const string& suffix = "clone");

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
      HloCloneContext* context = nullptr, const string& suffix = "clone");
  std::unique_ptr<HloComputation> CloneWithReplacementPairs(
      std::pair<const HloInstruction*, std::unique_ptr<HloInstruction>> r1,
      std::pair<const HloInstruction*, std::unique_ptr<HloInstruction>> r2,
      HloCloneContext* context = nullptr, const string& suffix = "clone");
  std::unique_ptr<HloComputation> CloneWithReplacementPairs(
      std::pair<const HloInstruction*, std::unique_ptr<HloInstruction>> r1,
      std::pair<const HloInstruction*, std::unique_ptr<HloInstruction>> r2,
      std::pair<const HloInstruction*, std::unique_ptr<HloInstruction>> r3,
      HloCloneContext* context = nullptr, const string& suffix = "clone");

  // Returns true if the given instruction can be removed from the computation.
  // Parameter instructions cannot be removed without violating invariants of
  // the HLO computation with the exception of fusion computation. A parameter
  // instruction is removable for a fusion computation.
  //
  // Note that IsRemovable() is a necessariy condition to remove an instruction
  // rather than a sufficient condition. For example, instructions with
  // side-effect (e.g., Send, Infeed) may be removed from a computation, but the
  // transformation must guarantee the invariants relevant to the instructions
  // still hold (e.g., Send and Recv must be removed together to make each
  // channel complete).
  bool IsRemovable(const HloInstruction* instruction);

  // Returns a map from channel-id to directed dependencies of the channel
  // instructions. For send&recv pairs it means the send instruction and for
  // all-reduce the union of the dependencies for all participating
  // instructions.
  using ChannelDependencyMap =
      absl::flat_hash_map<int64, absl::InlinedVector<HloInstruction*, 1>>;
  ChannelDependencyMap ComputeChannelDependencies() const;

  // Returns true if this computation has a side effect. A computation has a
  // side effect if it contains one or more instructions with a side effect.
  bool HasSideEffect() const;

  // Returns if this computation is a fusion computation.
  bool IsFusionComputation() const { return fusion_instruction_ != nullptr; }

  // Returns the owning fusion instruction, or nullptr if this is not a fusion
  // computation.
  HloInstruction* FusionInstruction() const { return fusion_instruction_; }
  void SetFusionInstruction(HloInstruction* fusion_instruction) {
    fusion_instruction_ = fusion_instruction;
  }

  // Clear the unique ID of the computation so that it can be re-assigned, such
  // as for the purpose of compacting the unique IDs.
  void ClearUniqueIdInternal() { unique_id_ = -1; }

  // The id of this computation should be unique within the module.
  void SetUniqueId(int64 id) {
    CHECK_EQ(unique_id_, -1);
    CHECK_GE(id, 0);
    unique_id_ = id;
  }

  // Returns the instruction in this computation that has name `name`.  Returns
  // null if there is no such computation.
  HloInstruction* GetInstructionWithName(absl::string_view name);

  int64 unique_id() const { return unique_id_; }

 private:
  explicit HloComputation(
      const string& name, int parameter_count,
      std::vector<std::unique_ptr<HloInstruction>>* instructions,
      HloInstruction* root_instruction, HloInstruction* fusion_instruction);

  // Internal helper for adding instructions.
  HloInstruction* AddInstructionInternal(
      std::unique_ptr<HloInstruction> instruction);

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
      const HloComputation::ChannelDependencyMap& channel_dependency_map,
      std::vector<HloInstruction*>* post_order, HloInstruction* root,
      absl::flat_hash_map<HloInstruction*, VisitState>* visited) const;

  string name_;
  int64 unique_id_;
  HloInstruction* root_instruction_;

  // If this computation is a fusion computation, this field points to the
  // corresponding fusion instruction.  Otherwise, this is null.
  HloInstruction* fusion_instruction_;

  // Module containing this computation.
  HloModule* parent_ = nullptr;

  // Store instructions in std::list as they can be added and removed
  // arbitrarily and we want a stable iteration order. Keep a map from
  // instruction pointer to location in the list for fast lookup.
  using InstructionList = std::list<std::unique_ptr<HloInstruction>>;
  InstructionList instructions_;
  absl::flat_hash_map<const HloInstruction*, InstructionList::iterator>
      instruction_iterators_;

  std::vector<HloInstruction*> param_instructions_;

  TF_DISALLOW_COPY_AND_ASSIGN(HloComputation);
};

}  // namespace xla

#endif  // TENSORFLOW_COMPILER_XLA_SERVICE_HLO_COMPUTATION_H_
