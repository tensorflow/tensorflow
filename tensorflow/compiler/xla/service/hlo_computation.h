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

#include <list>
#include <memory>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include "tensorflow/compiler/xla/iterator_util.h"
#include "tensorflow/compiler/xla/map_util.h"
#include "tensorflow/compiler/xla/service/dfs_hlo_visitor.h"
#include "tensorflow/compiler/xla/service/dfs_hlo_visitor_with_default.h"
#include "tensorflow/compiler/xla/service/hlo.pb.h"
#include "tensorflow/compiler/xla/service/hlo_instruction.h"
#include "tensorflow/compiler/xla/service/hlo_reachability.h"
#include "tensorflow/compiler/xla/service/name_uniquer.h"
#include "tensorflow/compiler/xla/shape_tree.h"
#include "tensorflow/compiler/xla/statusor.h"
#include "tensorflow/compiler/xla/types.h"
#include "tensorflow/compiler/xla/xla_data.pb.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/lib/gtl/array_slice.h"
#include "tensorflow/core/lib/gtl/flatmap.h"
#include "tensorflow/core/lib/gtl/flatset.h"
#include "tensorflow/core/platform/macros.h"
#include "tensorflow/core/platform/types.h"

namespace xla {

class HloModule;

// Describes a computation at the HLO level.
//
// An HloComputation contains a directed acyclic graph of HLO instructions. The
// computation has a single root instruction which produces the output of the
// computation.
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

  // Add new parameter instruction to the computation.
  // This should be a new parameter. Instruction will be appended to parameters
  // and inserted to the instruction list.
  HloInstruction* AddParameter(std::unique_ptr<HloInstruction> instruction);

  // Remove an instruction from the computation. The instruction must have no
  // users. Instruction is deallocated with this call.
  Status RemoveInstruction(HloInstruction* instruction);

  // Remove an instruction from the computation and also transitively any
  // operand that has no users post removing an instruction. The instruction
  // must have no users. Instruction is deallocated with this call.
  Status RemoveInstructionAndUnusedOperands(HloInstruction* instruction);

  // Set the root of the computation to the given instruction. The instruction
  // must have already been added to the computation and have the same shape as
  // the result of the computation for non fusion computations.
  void set_root_instruction(HloInstruction* new_root_instruction);

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
  string ToString(int nested_level = 0,
                  bool include_large_constants = false) const;

  // Returns a serialized representation of this computation.
  HloComputationProto ToProto() const;

  // Creates a computation from the given proto. Arguments:
  //
  //   module: the module which will contain the computation. The newly created
  //     computation is *not* added to the module, however.
  //   proto: the proto to convert from.
  //   computation_map: a map from computation name to HloComputation*. This map
  //     must contain all computations which the newly constructed computation
  //     calls.
  //  fusion_instruction: if non-null then the newly created computation will be
  //     constructed as a fused computation with this instruction as its fusion
  //     parent.
  static StatusOr<std::unique_ptr<HloComputation>> CreateFromProto(
      HloModule* module, const HloComputationProto& proto,
      tensorflow::gtl::FlatMap<string, HloComputation*>* computation_map,
      HloInstruction* fusion_instruction = nullptr);

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
  std::list<HloInstruction*> MakeInstructionPostOrder() const;

  // Computes and returns the reachability between HLO instructions in the
  // computation. The returned HloReachabilityMap is constructed such that
  // HloReachabilityMap::IsReachable(a, b) returns true iff there exists a
  // directed path (from producer to consumer) from 'a' to 'b'. Both data
  // dependencies (operands) and control dependencies are considered for
  // reachability. Trivially an instruction is reachable from itself.
  std::unique_ptr<HloReachabilityMap> ComputeReachability() const;

  // Updates the given reachability map after the immediate predecessor set
  // (operands and control predecessors) of 'instruction' has changed.
  void UpdateReachabilityThroughInstruction(
      const HloInstruction* instruction, HloReachabilityMap* reachability_map);

  int64 instruction_count() const { return instructions_.size(); }

  // Creates and returns a list of the embedded computations called by this
  // computation. This includes all embedded computations called directly or
  // transitively. The embedded computations are sorted such that if computation
  // A calls computation B (eg, via a map instruction) then A will appear after
  // B in the list.
  std::list<HloComputation*> MakeEmbeddedComputationsList() const;

  // Creates a fusion instruction containing the given instructions.
  // `fusion_kind` indicates the type of the fusion, e.g., loop fusion or fusion
  // into a library call. Instructions must be in reverse topological order
  // (root of the fused expression first). Replaces all uses of the original
  // root instruction with the fusion instruction. The original instructions are
  // removed if they have no uses after fusion (this is necessarily true for at
  // least the root).
  HloInstruction* CreateFusionInstruction(
      tensorflow::gtl::ArraySlice<HloInstruction*> instructions_to_fuse,
      HloInstruction::FusionKind fusion_kind);

  // Creates a fusion instruction that represents a backward convolution. This
  // is similar to CreateFusionInstruction but takes window and conv_dnums which
  // indicate the window and convolution dimension numbers of the backward
  // convolution.
  HloInstruction* CreateFusionInstructionForBackwardConvolution(
      tensorflow::gtl::ArraySlice<HloInstruction*> instructions_to_fuse,
      HloInstruction::FusionKind fusion_kind, const Window& window,
      const ConvolutionDimensionNumbers& conv_dnums);

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

  // Computes and returns the ProgramShape of this computation (shape of
  // parameters and result without layout).
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
                       const std::vector<const HloInstruction*>& order) const;

  // Same as Accept() above, but the visitor is given as a function.
  Status Accept(const std::function<Status(HloInstruction*)>& visitor_func);
  Status Accept(
      const std::function<Status(const HloInstruction*)>& visitor_func) const;

  // Returns a deep copy of this computation including all instructions.
  // If the module pointer is not nullptr, it will be the module where
  // the cloned computations will be added to (in order to support deep
  // cloning).
  std::unique_ptr<HloComputation> Clone(const string& suffix = "clone",
                                        HloModule* module = nullptr);

  // Like Clone(), but if an instruction is present in replacement_map, we use
  // the map's value to replace that instruction in the cloned computation.
  //
  // If replacements maps a key to nullptr, we remove that instruction from the
  // new computation.
  std::unique_ptr<HloComputation> CloneWithReplacements(
      std::unordered_map<const HloInstruction*, std::unique_ptr<HloInstruction>>
          replacements,
      HloModule* module = nullptr, const string& suffix = "clone");

  // Returns true if the given instruction can be removed from the
  // computation. Instructions such as parameters and send/receive instructions
  // cannot be removed without violating invariants of the HLO computation or
  // module with the exception of fusion computation.  A parameter instruction
  // is removable for a fusion computation.
  bool IsRemovable(const HloInstruction* instruction);

  // Returns true if this computation has a side effect. A computation has a
  // side effect if it contains one or more instructions with a side effect.
  bool HasSideEffect() const;

  // Returns if this computation is a fusion computation.
  bool IsFusionComputation() const { return fusion_instruction_ != nullptr; }

  // Returns the owning fusion instruction, or nullptr if this is not a fusion
  // computation.
  HloInstruction* FusionInstruction() const { return fusion_instruction_; }

 private:
  explicit HloComputation(
      const string& name, int parameter_count,
      std::vector<std::unique_ptr<HloInstruction>>* instructions,
      HloInstruction* root_instruction, HloInstruction* fusion_instruction);

  // Internal helper for adding instructions.
  HloInstruction* AddInstructionInternal(
      std::unique_ptr<HloInstruction> instruction);

  // Helper for setting the parent of instructions that are added to this
  // computation.
  void Reparent(HloInstruction* instruction);

  // Fuses HLOs in instructions_to_fuse into fusion_instruction.
  //
  // Pre-condition: fusion_instruction's opcode is kFusion.
  void FuseInstructionsInto(
      tensorflow::gtl::ArraySlice<HloInstruction*> instructions_to_fuse,
      HloInstruction* fusion_instruction);

  // Internal helper for recursive copying of an instruction. Creates and
  // returns a deep copy of the given instruction.
  StatusOr<HloInstruction*> DeepCopyHelper(
      HloInstruction* instruction, const ShapeTree<bool>* indices_to_copy,
      ShapeTree<HloInstruction*>* copies_added, ShapeIndex* index);

  // Internal helper to collect unreachable roots.
  std::vector<HloInstruction*> CollectUnreachableRoots() const;

  string name_;
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
  std::unordered_map<const HloInstruction*, InstructionList::iterator>
      instruction_iterators_;

  std::vector<HloInstruction*> param_instructions_;

  TF_DISALLOW_COPY_AND_ASSIGN(HloComputation);
};

}  // namespace xla

#endif  // TENSORFLOW_COMPILER_XLA_SERVICE_HLO_COMPUTATION_H_
