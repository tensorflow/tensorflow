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

#include "tensorflow/compiler/xla/map_util.h"
#include "tensorflow/compiler/xla/service/dfs_hlo_visitor.h"
#include "tensorflow/compiler/xla/service/dfs_hlo_visitor_with_default.h"
#include "tensorflow/compiler/xla/service/hlo_instruction.h"
#include "tensorflow/compiler/xla/service/name_uniquer.h"
#include "tensorflow/compiler/xla/statusor.h"
#include "tensorflow/compiler/xla/types.h"
#include "tensorflow/compiler/xla/xla_data.pb.h"
#include "tensorflow/core/lib/core/bitmap.h"
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
    explicit Builder(const string& name, bool is_fusion_computation = false)
        : name_(name),
          last_added_instruction_(nullptr),
          is_fusion_computation_(is_fusion_computation) {}

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
    bool is_fusion_computation_;
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

  // Replace all uses of "instruction_to_replace" with "instruction". Also, if
  // instruction_to_replace is the root of this computation then the root is set
  // to "instruction". Does not remove "instruction_to_replace".
  Status ReplaceUsesOfInstruction(HloInstruction* instruction_to_replace,
                                  HloInstruction* instruction);

  // Set the root of the computation to the given instruction. The instruction
  // must have already been added to the computation and have the same shape as
  // the result of the computation.
  void set_root_instruction(HloInstruction* instruction);

  // Return the root instruction of the computation. The root instruction is the
  // instruction which produces the output of the computation.
  HloInstruction* root_instruction() const { return root_instruction_; }

  // Returns the number of parameters for this computation.
  int64 num_parameters() const { return param_instructions_.size(); }

  // Returns the parameter instruction for the given parameter number.
  HloInstruction* parameter_instruction(int64 param_no) const {
    CHECK_GE(param_no, 0);
    CHECK_LT(param_no, static_cast<int64>(param_instructions_.size()));
    return param_instructions_[param_no];
  }

  const std::vector<HloInstruction*>& parameter_instructions() const {
    return param_instructions_;
  }

  const string& name() const { return name_; }
  void set_name(const string& name) { name_ = name; }

  // Return a string representation of the computation.
  string ToString(int nested_level = 0) const;

  const std::list<std::unique_ptr<HloInstruction>>& instructions() const {
    return instructions_;
  }

  // Compute and return a post-order of the instructions in the computation. In
  // this order, definitions of values always appear before their uses.
  std::list<HloInstruction*> MakeInstructionPostOrder() const;

  // Computes and returns the mapping from HLO to its transitive operands.
  class ReachabilityMap;
  std::unique_ptr<ReachabilityMap> ComputeTransitiveOperands() const;

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
  // with a series of kGetTupleElement and kTuple instructions.
  StatusOr<HloInstruction*> DeepCopyInstruction(HloInstruction* instruction);

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
  Status Accept(DfsHloVisitor* visitor) const;

  // Same as Accept() above, but the order of operand and control predecessor
  // visitation is determined by the given operand order; if compare(A, B) ==
  // true, A is visited before B.
  Status AcceptWithOperandOrder(
      DfsHloVisitor* visitor,
      const HloInstruction::CompareFunction& operand_order) const;

  // Visit every node in the computation in the given order. 'order' must
  // be a topological sort of all instructions in the computation.
  Status AcceptOrdered(DfsHloVisitor* visitor,
                       const std::vector<const HloInstruction*>& order) const;

  // Same as Accept() above, but the visitor is given as a function.
  Status Accept(const FunctionVisitor::VisitorFunction& visitor_func) const;

  // Returns a deep copy of this computation including all instructions.
  std::unique_ptr<HloComputation> Clone(const string& suffix = "clone");

  // Returns true if the given instruction can be removed from the
  // computation. Instructions such as parameters and send/receive instructions
  // cannot be removed without violating invariants of the HLO computation or
  // module with the exception of fusion computation.  A parameter instruction
  // is removable for a fusion computation.
  bool IsRemovable(const HloInstruction* instruction);

  // Returns if this computation is a fusion computation.
  bool IsFusionComputation() const { return is_fusion_computation_; }

 private:
  explicit HloComputation(
      const string& name, int parameter_count,
      std::vector<std::unique_ptr<HloInstruction>>* instructions,
      HloInstruction* root_instruction, bool is_fusion_computation = false);

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

  // Internal helper for copying a tuple value. Creates and returns a deep copy
  // of the given instruction. The given instruction must be tuple-shaped.
  StatusOr<HloInstruction*> DeepCopyTuple(HloInstruction* instruction);

  // Internal helper to collect unreachable roots.
  std::vector<HloInstruction*> CollectUnreachableRoots() const;

  string name_;
  HloInstruction* root_instruction_;

  // A tag shows if this is a fusion computation.
  bool is_fusion_computation_;

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

  // Unique name generator for instruction identifiers. Instruction names should
  // be unique per computation and this is enforced when instructions are added
  // to the computation.
  NameUniquer instruction_name_uniquer_;

  TF_DISALLOW_COPY_AND_ASSIGN(HloComputation);
};

class HloComputation::ReachabilityMap {
 public:
  // Sets up an empty reachable matrix for the full set of
  // instructions specified in "all_instructions"
  explicit ReachabilityMap(const std::list<HloInstruction*>& all_instructions);
  // Sets entry so that IsReachable(a, b) will return true
  void SetReachable(const HloInstruction* a, const HloInstruction* b);

  // Sets IsReachable(a_inst, b_inst) as well as IsReachable(a_inst, trans)
  // for all "trans" s.t. "IsReachable(b_inst, trans)" is true
  void SetReachableAndTransitiveClosure(const HloInstruction* a_inst,
                                        const HloInstruction* b_inst);

  // Returns true if "b" is reachable from "a"
  bool IsReachable(const HloInstruction* a, const HloInstruction* b) const;

  // Returns true if "b" is reachable from "a" or "a" is reachable from "b"
  bool IsConnected(const HloInstruction* a, const HloInstruction* b) const;

 private:
  friend class HloComputation;

  // dense id assignment from HloInstruction* to number
  tensorflow::gtl::FlatMap<const HloInstruction*, int> ids_;
  // matrix_(a,b) is true iff b is reachable from a
  tensorflow::core::Bitmap matrix_;
};

}  // namespace xla

#endif  // TENSORFLOW_COMPILER_XLA_SERVICE_HLO_COMPUTATION_H_
