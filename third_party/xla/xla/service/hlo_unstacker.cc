/* Copyright 2024 The OpenXLA Authors.

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

#include "xla/service/hlo_unstacker.h"

#include <algorithm>
#include <cstdint>
#include <deque>
#include <functional>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "absl/algorithm/container.h"
#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/strings/match.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/map_util.h"
#include "xla/service/hlo_creation_utils.h"
#include "xla/service/pattern_matcher.h"
#include "xla/service/tuple_util.h"
#include "xla/service/while_loop_unroller.h"
#include "xla/shape.h"
#include "xla/shape_util.h"
#include "tsl/platform/errors.h"
#include "tsl/platform/statusor.h"

namespace xla {
namespace {

// TODO: b/352400145 - Unify the patterns, handlers and their type into a class
// or struct.
enum class PatternType {
  DSFusionNoBitcastPattern,
  DSFusionPattern,
  NestedDSFusionPattern,
  Other,
};

static std::string PatternTypeToString(PatternType pattern_type) {
  switch (pattern_type) {
    case PatternType::DSFusionNoBitcastPattern:
      return "DSFusionNoBitcastPattern";
    case PatternType::DSFusionPattern:
      return "DSFusionPattern";
    case PatternType::NestedDSFusionPattern:
      return "NestedDSFusionPattern";
    case PatternType::Other:
      return "Other";
  }
}

// Holds the information about custom unstacking patterns.
struct PatternInfo {
  PatternType type;
  std::vector<const HloInstruction*> unstacked_instrs;
  const HloInstruction* instr;
  Shape unstacked_shape;
  HloComputation* unstacking_computation;

  std::string ToString() const {
    if (unstacking_computation == nullptr) {
      return absl::StrCat("type: \n\t", PatternTypeToString(type), "\n",
                          "instr: \n\t", instr->name(), "\n", "shape: \n\t",
                          unstacked_shape.ToString(true));
    } else {
      return absl::StrCat("type: \n\t", PatternTypeToString(type), "\n",
                          "instr: \n\t", instr->name(), "\n", "shape: \n\t",
                          unstacked_shape.ToString(true), "\n", "comp: \n",
                          unstacking_computation->name());
    }
  }
};

// TODO: b/342457472 - Remove this struct and move its field to the
// UnstackerTransformer as static members. A struct that holds the required
// information for unstacking that is fixed across different unstacker
// instastances.
struct UnstackerMetadata {
  static absl::StatusOr<UnstackerMetadata> Create(
      HloModule* module, std::function<bool(HloInstruction*)> unfuse_slice) {
    UnstackerMetadata metadata;
    TF_ASSIGN_OR_RETURN(
        bool prepared,
        WhileLoopUnroller::PrepareModuleForUnrolling(module, {}));
    if (prepared) {
      VLOG(3) << "Prepared module: " << module->name() << " for unstacking.";
    }
    std::vector<std::pair<HloInstruction*, WhileLoopConfig>> loops =
        WhileLoopUnroller::GetUnrollableLoops(module, {},
                                              /*unroll_config=*/std::nullopt);
    for (const auto& [instr, while_loop_config] : loops) {
      metadata.unrollable_loop_bodies[instr->while_body()] = while_loop_config;
      metadata.bodies[instr->while_body()] = instr;
    }
    metadata.unfuse_slice = unfuse_slice;
    return metadata;
  }
  absl::flat_hash_map<HloComputation*, WhileLoopConfig> unrollable_loop_bodies;
  absl::flat_hash_map<const HloComputation*, HloInstruction*> bodies;
  // Vector containing pairs of custom patterns and their corresponding handler
  // lambdas. The patterns are checked in the order in which they are inserted
  // into this vector.
  std::vector<
      std::pair<std::function<std::optional<PatternInfo>(
                    const UnstackerMetadata&, const HloInstruction*, int64_t)>,
                std::function<absl::Status(HloInstruction*, const Shape&)>>>
      custom_handlers;
  std::function<bool(HloInstruction*)> unfuse_slice;
};

// Performs the two-step unstacking. Each instance of this class is responsible
// for a single operand of a while loop.
class UnstackerTransformer {
 public:
  // Default unroll_factor of -1 indicates full unrolling
  explicit UnstackerTransformer(const UnstackerMetadata& metadata)
      : metadata_(metadata) {}

  // Given an instruction and the index of the its changed operand, it applies
  // the custom handler and populates body_changes lambdas that unstacks the hlo
  // graph accordingly.
  std::vector<const HloInstruction*> HandleInstruction(
      const HloInstruction* instr, int64_t changed_idx) {
    // Currently, we only unstack operands that are used within fusion
    // computations.
    if (instr->opcode() != HloOpcode::kFusion) {
      return {};
    }
    VLOG(3) << "HandleInstruction(" << instr->shape().ToString()
            << instr->name() << ", " << changed_idx << ")";

    for (const auto& [custom_pattern, custom_handler] :
         metadata_.custom_handlers) {
      std::optional<PatternInfo> stacked_user =
          custom_pattern(metadata_, instr, changed_idx);
      // Try the next pattern if current pattern is not found.
      if (!stacked_user.has_value()) {
        continue;
      }
      PatternInfo& pattern_info = stacked_user.value();
      pattern_type_ = pattern_info.type;
      VLOG(3) << "PatternInfo:" << "\n" << pattern_info.ToString();

      if (pattern_info.unstacking_computation != nullptr &&
          unstacking_computation_ != nullptr) {
        if (!absl::EqualsIgnoreCase(
                pattern_info.unstacking_computation->ToString(
                    HloPrintOptions::Fingerprint()),
                unstacking_computation_->ToString(
                    HloPrintOptions::Fingerprint()))) {
          VLOG(3) << "Seen multiple unstacking computations, cannot handle: "
                  << "\n previous computations: \n"
                  << unstacking_computation_->ToString(
                         HloPrintOptions::Fingerprint())
                  << "\n current computations: \n"
                  << pattern_info.unstacking_computation->ToString(
                         HloPrintOptions::Fingerprint());
          return {};
        }
      }

      if (pattern_info.unstacking_computation != nullptr) {
        unstacking_computation_ = pattern_info.unstacking_computation;
      }

      unstacked_shape_ = std::make_unique<Shape>(pattern_info.unstacked_shape);
      unstacked_instrs_.push_back(instr);

      // Wrapper function around the unstacker lambda which calls the unstacker.
      std::function<absl::Status()> unstack_wrapper =
          [&custom_handler = custom_handler,
           pattern_info]() mutable -> absl::Status {
        HloInstruction* mutable_dynamic_slicing_fusion =
            const_cast<HloInstruction*>(pattern_info.instr);
        return custom_handler(mutable_dynamic_slicing_fusion,
                              pattern_info.unstacked_shape.tuple_shapes(0));
      };
      body_changes_.push_back(unstack_wrapper);
      return pattern_info.unstacked_instrs;
    }
    return {};
  }

  const UnstackerMetadata& GetMetadata() const { return metadata_; }

  std::vector<const HloInstruction*>& GetUnstackedInstructions() {
    return unstacked_instrs_;
  }

  const Shape* GetUnstackedShape() const { return unstacked_shape_.get(); }

  // The function returns a mutable pointer to the unstacking computation since
  // the pointer is later used to clone the computation.
  HloComputation* GetUnstackingComputation() const {
    return unstacking_computation_;
  }

  std::vector<std::function<void(const UnstackerTransformer&)>>&
  GetLoopChanges() {
    return loop_changes_;
  }

  std::vector<std::function<absl::Status()>>& GetBodyChanges() {
    return body_changes_;
  }

  absl::flat_hash_map<HloInstruction*, std::vector<int64_t>>&
  GetOperandChanges() {
    return operand_changes_;
  }

  void AddOperandChange(HloInstruction* instr, int64_t index) {
    operand_changes_[instr].push_back(index);
  }

  void AddLoopChange(
      std::function<void(const UnstackerTransformer&)> loop_change) {
    loop_changes_.push_back(loop_change);
  }

  PatternType GetPatternType() const { return pattern_type_; }

 private:
  PatternType pattern_type_;
  const UnstackerMetadata& metadata_;
  // This pointer is populated if the unstacker finds unstackable loop input.
  std::unique_ptr<Shape> unstacked_shape_ = nullptr;
  // This is a pointer to the computation that is responsible for unstacking. It
  // is used to hoist the unstacking computations outside the loop bodies.
  // std::unique_ptr<HloComputation>
  HloComputation* unstacking_computation_ = nullptr;
  // A vector of lambdas that describe necessary changes to the shape of the
  // loops to unstack. The lambdas accept the pointer to the new unstacked
  // shape.
  std::vector<std::function<void(const UnstackerTransformer&)>> loop_changes_;
  // a list of lambdas that captures all the changes to the hlo graph needed for
  // unstacking.
  std::vector<std::function<absl::Status()>> body_changes_;
  // A map that tracks the index of the changed operand for instructions of type
  // get-tuple-element, tuple, and while during unstacking.
  absl::flat_hash_map<HloInstruction*, std::vector<int64_t>> operand_changes_;
  // Holds the list of unstacked instructions that will be used to identify
  // loops that need to be unrolled.
  std::vector<const HloInstruction*> unstacked_instrs_;
};

bool CanUnstackWhileOperand(const HloInstruction* while_instr,
                            UnstackerTransformer& unstacker, int64_t index);

bool UnstackWhileOperandAtIndex(
    const UnstackerMetadata& metadata, HloInstruction* while_instr,
    int64_t index, std::vector<const HloInstruction*>& unstacked_instructions);

// Given a gte and an unstacker instance, this function walks down the graph of
// the users in BFS manner and propagates the index of the changed input operand
// for kGetTupleElement, kTuple, and kWhile instructions. Moreover, if checks if
// the a user should be handled with the provided custom handler(s) inside the
// unstacker instance. Note that this function does NOT change the shape of any
// instruction, it merely keeps track of the instructions and where in the input
// operands the change need to be applied later.
bool PropagateGteShapeChange(HloInstruction* gte,
                             UnstackerTransformer& unstacker) {
  VLOG(5) << "PropagateGteShapeChange(" << gte->name() << ")";

  HloInstruction* parent_while = nullptr;
  if (unstacker.GetMetadata().bodies.contains(gte->parent())) {
    parent_while = unstacker.GetMetadata().bodies.at(gte->parent());
    if (parent_while->while_body() != gte->parent()) {
      parent_while = nullptr;
    }
  }

  std::vector<const HloInstruction*> handled_instrs;
  // TODO: b/343457903 - Use HloDataflowAnalysis to track the usage of a value
  // instead of manually applying bfs
  //
  // Apply BFS to propagate the index of the changed operand. We put all the
  // changed instructions along with the index of the changed operand in the
  // visited map and then propagate the change to the users of the instruction.
  absl::flat_hash_map<HloInstruction*, int64_t> visited;
  std::deque<HloInstruction*> worklist;
  worklist.push_back(gte);
  visited.insert({gte, gte->tuple_index()});
  unstacker.AddOperandChange(gte, gte->tuple_index());
  while (!worklist.empty()) {
    HloInstruction* changed_instr_to_propagate = worklist.front();
    // The index of the changed operand that needs to be propagated.
    int64_t changed_operand_index =
        FindOrDie(visited, changed_instr_to_propagate);
    worklist.pop_front();
    for (HloInstruction* user : changed_instr_to_propagate->users()) {
      if (ContainsKey(visited, user)) {
        continue;
      }
      // We explicitly propagate the changed index for three types of users,
      // namely, get-tuple-element, tuple and while users. The rationale is that
      // the output shape of these three instruction types are inferred only by
      // their input operand(s). Finally, we check if the user can be handled by
      // the provided custom handler in HandleInstruction method.
      if (user->opcode() == HloOpcode::kGetTupleElement) {
        if (user->tuple_index() != changed_operand_index) {
          continue;
        }
        // Since we insert the gte user only if the index of the gte is equal to
        // the changed operand of its tuple input, we are sure that this gte
        // instruction will get the new shape eventually and the
        // change_operand_index does not matter.
        visited.insert({user, changed_operand_index});
        unstacker.AddOperandChange(user, changed_operand_index);
        worklist.push_back(user);
      } else if (user->opcode() == HloOpcode::kTuple) {
        for (int64_t i = 0; i < user->operand_count(); ++i) {
          if (user->operand(i) == changed_instr_to_propagate) {
            visited.insert({user, i});
            unstacker.AddOperandChange(user, i);
            worklist.push_back(user);
            if (parent_while != nullptr && user->IsRoot() &&
                i != gte->tuple_index()) {
              bool changed_nested_while =
                  CanUnstackWhileOperand(parent_while, unstacker, i);
              if (!changed_nested_while) {
                return false;
              }
            }
          }
        }
      } else if (user->opcode() == HloOpcode::kWhile) {
        // Recursively check the inner while for unstacking and populate
        // unstacker instance.
        bool changed_nested_while =
            CanUnstackWhileOperand(user, unstacker, changed_operand_index);
        if (!changed_nested_while) {
          return false;
        }
        visited.insert({user, changed_operand_index});
        unstacker.AddOperandChange(user, changed_operand_index);
        worklist.push_back(user);
      } else {
        if (absl::c_find(handled_instrs, user) != handled_instrs.end()) {
          continue;
        }
        // If already unstacked, we do not need to handle again.
        if (user->IsCustomCall("DynamicGte") ||
            user->IsCustomCall("DynamicTuple")) {
          continue;
        }
        int64_t use_index = user->operand_index(changed_instr_to_propagate);
        std::vector<const HloInstruction*> curr_handled_instrs =
            unstacker.HandleInstruction(user, use_index);
        if (curr_handled_instrs.empty()) {
          VLOG(3) << "Custom unstacker not found for " << user->name();
          return false;
        }
        for (const HloInstruction* instr : curr_handled_instrs) {
          // TODO: b/352400145 - Here we check if the user has the same shape as
          // the stacked tensor (how to capture this more robustly?). if so, we
          // need to add the user to the worklist to get updated.
          for (HloInstruction* handled_instr_user : instr->users()) {
            if (user->shape() == gte->shape()) {
              visited.insert({handled_instr_user, changed_operand_index});
              unstacker.AddOperandChange(handled_instr_user,
                                         changed_operand_index);
              worklist.push_back(handled_instr_user);
            }
          }
          handled_instrs.push_back(instr);
        }
      }
    }
  }
  return true;
}

// Within the given computation, finds all the gte instruction with the
// following form: get-tuple-elements(operand), index=idx and collects all the
// new shapes. new_shape is the new shape at idx of the operand of the gte.
bool CanPropagateGteShapeChangesInComputation(
    const HloComputation* comp, const HloInstruction* operand,
    UnstackerTransformer& shape_transformer, int64_t idx) {
  VLOG(3) << "Propagating shape change of index " << idx
          << " in : " << comp->name();
  for (HloInstruction* instr : comp->MakeInstructionPostOrder()) {
    // We only need to propagate changes through the gte instructions with index
    // = idx.
    if (instr->opcode() == HloOpcode::kGetTupleElement &&
        instr->tuple_index() == idx) {
      if (instr->operand(0) != operand) {
        continue;
      }
      // If propagation is not possible (no custom handler provided for the
      // users of the candidate), we bail early.
      bool can_propagate = PropagateGteShapeChange(instr, shape_transformer);
      if (!can_propagate) {
        VLOG(3) << "Failed to propagate shape change for " << instr->name();
        return false;
      }
    }
  }
  VLOG(3) << "Finish propagating shape change of index " << idx
          << " in: " << comp->name();
  return true;
}

std::unique_ptr<HloInstruction> DynamicSliceToSlice(
    HloInstruction* dynamic_slice, HloInstruction* input, int64_t i) {
  std::vector<int64_t> new_start_indices;
  new_start_indices.reserve(dynamic_slice->shape().dimensions().size());
  std::vector<int64_t> new_limit_indices;
  new_limit_indices.reserve(dynamic_slice->shape().dimensions().size());
  std::vector<int64_t> new_strides;
  new_strides.reserve(dynamic_slice->shape().dimensions().size());
  new_start_indices.push_back(i);
  new_limit_indices.push_back(i + 1);
  new_strides.push_back(1);
  for (int64_t j = 1; j < dynamic_slice->shape().dimensions().size(); ++j) {
    new_start_indices.push_back(0);
    new_limit_indices.push_back(
        dynamic_slice->mutable_operand(0)->shape().dimensions(j));
    new_strides.push_back(1);
  }
  return HloInstruction::CreateSlice(dynamic_slice->shape(), input,
                                     new_start_indices, new_limit_indices,
                                     new_strides);
}

bool ShouldUnfuseSlices(const UnstackerMetadata& metadata, HloInstruction* ds) {
  HloInstruction* input = ds->mutable_operand(0);
  for (int64_t i = 0; i < input->shape().dimensions(0); ++i) {
    HloInstruction* slice =
        ds->AddInstruction(DynamicSliceToSlice(ds, input, i));
    if (!metadata.unfuse_slice(slice)) {
      CHECK_OK(slice->parent()->RemoveInstruction(slice));
      return false;
    }
    CHECK_OK(slice->parent()->RemoveInstruction(slice));
  }
  return true;
}

// This function is responsible for:
// 1. Hoisting the unstacking computation outside the while_instr.
// 2. Replacing the input of the while_instr with the new unstacked version.
void UnstackWhileInput(const UnstackerTransformer& unstacker,
                       HloInstruction* while_instr, int64_t index) {
  VLOG(3) << "Unstacking while input: " << while_instr->name() << " at "
          << index;
  const Shape* new_shape = unstacker.GetUnstackedShape();
  HloComputation* unstacking_computation = unstacker.GetUnstackingComputation();
  const Shape& slice_shape = new_shape->tuple_shapes(0);
  HloInstruction* old_while_input =
      while_instr->while_init()->mutable_operand(index);
  // If the input is a tuple, i.e., while_instr has already been unstacked
  // during unstacking of its parent, we do not need to unstack it again.
  if (old_while_input->shape().IsTuple()) {
    VLOG(3) << "Input is already unstacked: " << old_while_input->name();
    return;
  }

  std::vector<HloInstruction*> slices;
  // If the input is an AllocateBuffer, we simply break it down into a tuple of
  // AllocateBuffer instructions, one per slice.
  if (old_while_input->IsCustomCall("AllocateBuffer")) {
    for (int64_t i = 0; i < new_shape->tuple_shapes_size(); ++i) {
      slices.push_back(while_instr->AddInstruction(
          HloInstruction::CreateCustomCall(slice_shape, {}, "AllocateBuffer")));
    }
  } else {
    // TODO: b/341815540 - Instead of creating the unstacked tuple for every
    // input index, we should reuse if the input and unstacking computations are
    // the same.
    //
    // Hoist the unstacking computation outside the while_instr and create a
    // tuple of slices.
    for (int64_t i = 0; i < new_shape->tuple_shapes_size(); ++i) {
      HloInstruction* root_instr = unstacking_computation->root_instruction();
      // TODO: b/352400145 - After unifying patterns and handlers, instead of
      // using the pattern type to determine the unstacked input, we should use
      // the pattern object to call the appropriate method.
      //
      // For DSFusionPattern and NestedDSFusionPattern, we rewrite the
      // dynamic-slice as a slice instruction in the hope that these slices are
      // later prefetched using async-slice by MSA. For other patterns, we
      // resort to the original unstacking computation until we find benefit in
      // doing otherwise.
      HloInstruction* slice = nullptr;
      if (unstacker.GetPatternType() == PatternType::DSFusionPattern ||
          unstacker.GetPatternType() == PatternType::NestedDSFusionPattern ||
          unstacker.GetPatternType() == PatternType::DSFusionNoBitcastPattern) {
        if (unstacker.GetPatternType() == PatternType::DSFusionPattern ||
            unstacker.GetPatternType() == PatternType::NestedDSFusionPattern) {
          slice = while_instr->AddInstruction(DynamicSliceToSlice(
              root_instr->mutable_operand(0), old_while_input, i));
        } else if (unstacker.GetPatternType() ==
                   PatternType::DSFusionNoBitcastPattern) {
          slice = while_instr->AddInstruction(
              DynamicSliceToSlice(root_instr, old_while_input, i));
        }
      }
      if (slice == nullptr || !unstacker.GetMetadata().unfuse_slice(slice)) {
        std::vector<HloInstruction*> operands = {
            old_while_input,
            while_instr->AddInstruction(MakeScalarConstantWithShape(
                unstacking_computation->parameter_instruction(1)->shape(), i))};
        slice = while_instr->AddInstruction(HloInstruction::CreateFusion(
            slice_shape, HloInstruction::FusionKind::kLoop, operands,
            while_instr->GetModule()->AddEmbeddedComputation(
                unstacking_computation->Clone()),
            "hoisted"));
      }
      slices.push_back(slice);
    }
  }
  HloInstruction* new_operand_element =
      while_instr->AddInstruction(HloInstruction::CreateTuple(slices));
  HloInstruction* new_while_init =
      TupleUtil::ReplaceTupleWith(new_operand_element,
                                  while_instr->while_init(), {index}, false)
          .value();
  CHECK_OK(while_instr->ReplaceOperandWithDifferentShape(0, new_while_init));
}

bool CanUnstackWhileOperand(const HloInstruction* while_instr,
                            UnstackerTransformer& unstacker, int64_t index) {
  VLOG(5) << "ReplaceWhileOperandShape: " << while_instr->name() << " at "
          << index;

  bool body_changes_collected = CanPropagateGteShapeChangesInComputation(
      while_instr->while_body(),
      while_instr->while_body()->parameter_instruction(0), unstacker, index);
  if (!body_changes_collected) {
    return false;
  }

  bool condition_changes_collected = CanPropagateGteShapeChangesInComputation(
      while_instr->while_condition(),
      while_instr->while_condition()->parameter_instruction(0), unstacker,
      index);
  if (!condition_changes_collected) {
    return false;
  }

  // Check if we can propagate the changes through the output of the while
  // at index.
  bool parent_changes_collected = CanPropagateGteShapeChangesInComputation(
      while_instr->parent(), while_instr, unstacker, index);
  if (!parent_changes_collected) {
    VLOG(3) << "Failed: parent_changes_collected";
    return false;
  }

  HloInstruction* root_operand =
      while_instr->while_body()->root_instruction()->mutable_operand(index);
  if (root_operand == nullptr) {
    return false;
  }

  HloInstruction* gte_operand = nullptr;
  // Currently, we only support unstacking of while operands that either:
  // 1. Are parameters of the while_body.
  // 2. Are get-tuple-elements of another while instruction.
  if (Match(root_operand, match::GetTupleElement(match::Op(&gte_operand)))) {
    if (Match(gte_operand, match::While())) {
      VLOG(3) << "Faced a gte originating from loop: "
              << root_operand->ToString();
      bool loop_feeding_root_changes_collected = CanUnstackWhileOperand(
          root_operand->operand(0), unstacker, root_operand->tuple_index());
      if (!loop_feeding_root_changes_collected) {
        VLOG(3) << "Failed: loop " << root_operand->operand(0)->name()
                << " output at " << index << " is not unstackable";
        return false;
      }
    } else if (!Match(gte_operand, match::Parameter().WithParameterNum(0))) {
      VLOG(3) << "Failed: root operand of while_body at " << index
              << " is not a parameter";
      return false;
    }
  }

  auto loop_change = [=](const UnstackerTransformer& unstacker,
                         HloInstruction* loop, int64_t idx) mutable {
    Shape old_shape = ShapeUtil::MakeStaticShape(
        loop->while_body()->parameter_instruction(0)->shape());
    ShapeUtil::UpdateTupleShape(*unstacker.GetUnstackedShape(), idx,
                                &old_shape);

    loop->while_body()->ReplaceParameter(
        0, HloInstruction::CreateParameter(0, old_shape, "unstacked"));
    loop->while_condition()->ReplaceParameter(
        0, HloInstruction::CreateParameter(0, old_shape, "unstacked"));

    CHECK_NE(unstacker.GetUnstackingComputation(), nullptr);
    UnstackWhileInput(unstacker, loop, idx);
    // Update the input and output shape of the loop.
    *loop->mutable_shape() = old_shape;
  };
  auto loop_change_wrapper = [&loop_change, while_instr,
                              index](const UnstackerTransformer& unstacker) {
    HloInstruction* mutable_loop = const_cast<HloInstruction*>(while_instr);
    loop_change(unstacker, mutable_loop, index);
  };
  unstacker.AddLoopChange(loop_change_wrapper);
  return true;
}

// Apply the two-step unstacking algorithm to the given while_instr at the given
// index.
bool UnstackWhileOperandAtIndex(
    const UnstackerMetadata& metadata, HloInstruction* while_instr,
    int64_t index, std::vector<const HloInstruction*>& unstacked_instructions) {
  UnstackerTransformer unstacker = UnstackerTransformer(metadata);

  // First step of unstacking to determine whether while_instr at index is
  // unstackable.
  bool can_unstack = CanUnstackWhileOperand(while_instr, unstacker, index);
  if (!can_unstack) {
    VLOG(3) << "Unstacking failed for " << while_instr->name() << " at "
            << index;
    return false;
  }

  // If unstacker has not found an unstackable shape, there is no point in
  // applying the unstacker changes.
  if (unstacker.GetUnstackedShape() == nullptr) {
    VLOG(3) << "Failed: unstacked shape is null";
    return false;
  }

  // If unstacker has not found an unstackable shape, there is no point in
  // applying the unstacker changes.
  if (unstacker.GetUnstackingComputation() == nullptr) {
    VLOG(3) << "Failed: unstacking computation is null";
    return false;
  }

  // At this point, we have the unstacked_shape at hand. We go ahead and apply
  // all the changes that required the unstacked shape.
  //
  // Update the shape of get-tuple-element, tuple, and, while instructions
  // based on the unstacked_shape and the index of the changed operand.
  for (auto& [instr, indices] : unstacker.GetOperandChanges()) {
    switch (instr->opcode()) {
      case HloOpcode::kGetTupleElement:
        VLOG(3) << "Changing shape of: " << instr->name();
        *instr->mutable_shape() = *unstacker.GetUnstackedShape();
        break;
      case HloOpcode::kTuple: {
        for (int64_t index : indices) {
          VLOG(3) << "Changing shape of: " << instr->name() << " at " << index;
          *instr->mutable_shape()->mutable_tuple_shapes(index) =
              *unstacker.GetUnstackedShape();
        }
        break;
      }
      case HloOpcode::kWhile:
        for (int64_t index : indices) {
          VLOG(3) << "Changing shape of: " << instr->name() << " at " << index;
          ShapeUtil::UpdateTupleShape(*unstacker.GetUnstackedShape(), index,
                                      instr->mutable_shape());
        }
        break;
      default:
        LOG(FATAL) << "Unsupported opcode: " << instr->name();
    }
  }
  // Apply the changes to the body according to the provided custom handler.
  for (const auto& body_change : unstacker.GetBodyChanges()) {
    CHECK_OK(body_change());
  }
  // Apply the changes to the shape of the loop body and condition computations.
  for (auto& loop_change : unstacker.GetLoopChanges()) {
    loop_change(unstacker);
  }
  for (const HloInstruction* instr : unstacker.GetUnstackedInstructions()) {
    unstacked_instructions.push_back(instr);
  }
  return true;
}

Shape MakeUnstackedShapeFromSlice(const Shape& slice_shape, int64_t layers) {
  std::vector<Shape> shapes;
  shapes.reserve(layers);
  for (int64_t i = 0; i < layers; ++i) {
    shapes.push_back(slice_shape);
  }
  return ShapeUtil::MakeTupleShape(shapes);
}

// Checks if the given instruction is a fusion with num_fusion_params
// parameters inside an unrollable loop. If so, it returns the loop config.
std::optional<WhileLoopConfig> IsFusionInsideUnrollableLoopWithNumParameter(
    const UnstackerMetadata& metadata, const HloInstruction* instr,
    std::optional<int64_t> num_fusion_params) {
  if (instr->opcode() != HloOpcode::kFusion) {
    return std::nullopt;
  }
  if (num_fusion_params.has_value()) {
    if (instr->fused_parameters().size() != num_fusion_params) {
      VLOG(3) << "Fusion has different number of parameters";
      return std::nullopt;
    }
  }
  if (!metadata.unrollable_loop_bodies.contains(instr->parent())) {
    VLOG(5) << "Fusion not inside unrollable while body, " << instr->name()
            << " inside " << instr->parent()->name();
    return std::nullopt;
  }
  return metadata.unrollable_loop_bodies.at(instr->parent());
}

// Checks if the instruction is a fusion with num_fusion_params parameters
// inside an unrollable loop and within its fusion computation there is an
// effectively static dynamic-slice instruction on the most major dimension of
// the operand at the given stacked_operand_idx. If so, it returns the
// dynamic-slice instruction.
HloInstruction* GetMostMajorEffectivelyStaticDynamicSliceInFusion(
    const UnstackerMetadata& metadata, const HloInstruction* instr,
    std::optional<int64_t> num_fusion_params, int64_t stacked_operand_idx) {
  std::optional<WhileLoopConfig> while_instr_config =
      IsFusionInsideUnrollableLoopWithNumParameter(metadata, instr,
                                                   num_fusion_params);
  if (!while_instr_config.has_value()) {
    return nullptr;
  }
  for (HloInstruction* fused_instr :
       instr->fused_instructions_computation()->MakeInstructionPostOrder()) {
    std::optional<int64_t> dynamic_index =
        MatchEffectivelyStaticDynamicSliceInsideLoop(
            fused_instr,
            instr->fused_instructions_computation()->parameter_instruction(
                stacked_operand_idx),
            while_instr_config.value());
    if (dynamic_index.has_value() && dynamic_index.value() == 0) {
      return fused_instr;
    }
  }
  return nullptr;
}

// Checks if the instruction is a fusion with num_fusion_params parameters
// inside an unrollable loop and within its fusion computation looks for the
// dynamic-index instruction that covers the shape of the operand at the given
// index.
HloInstruction* GetMostMajorShapeCoveringDynamicIndexInFusion(
    const UnstackerMetadata& metadata, const HloInstruction* instr,
    HloOpcode opcode, int64_t num_fusion_params, int64_t stacked_operand_idx) {
  std::optional<WhileLoopConfig> while_instr_config =
      IsFusionInsideUnrollableLoopWithNumParameter(metadata, instr,
                                                   num_fusion_params);
  if (!while_instr_config.has_value()) {
    return nullptr;
  }
  for (HloInstruction* fused_instr :
       instr->fused_instructions_computation()->MakeInstructionPostOrder()) {
    if (fused_instr->opcode() != opcode) {
      continue;
    }
    std::optional<int64_t> dynamic_index =
        MatchShapeCoveringDynamicIndexInstruction(
            fused_instr,
            instr->fused_instructions_computation()->parameter_instruction(
                stacked_operand_idx),
            opcode, while_instr_config.value());
    if (dynamic_index.has_value() && dynamic_index.value() == 0) {
      return fused_instr;
    }
  }
  return nullptr;
}

// This function recognizes fusions with the following pattern:
// fusion(stacked, f(loop_iteration_var))
// computation {
//   p0 = parameter(0)
//   p1 = parameter(1)
//   slice = dynamic_slice(p0, p1, zero, ...)
//   ROOT bitcast = bitcast(slice)
// }
// where f is a function of loop_iteration_var. It indicates that the slicing
// offset is effectively static after unrolling.
std::optional<PatternInfo> GetDSFusionPattern(const UnstackerMetadata& metadata,
                                              const HloInstruction* instr,
                                              int64_t stacked_operand_idx) {
  VLOG(3) << "Checking DSFusion";
  HloInstruction* shape_covering_instr =
      GetMostMajorEffectivelyStaticDynamicSliceInFusion(metadata, instr, 2,
                                                        stacked_operand_idx);
  if (shape_covering_instr == nullptr) {
    return std::nullopt;
  }
  if (!ShouldUnfuseSlices(metadata, shape_covering_instr)) {
    return std::nullopt;
  }
  HloInstruction* bitcast_operand = nullptr;
  if (Match(instr->fused_instructions_computation()->root_instruction(),
            match::Bitcast(match::Op(&bitcast_operand)))) {
    if (bitcast_operand == shape_covering_instr) {
      PatternInfo pattern_info;
      pattern_info.type = PatternType::DSFusionPattern;
      pattern_info.instr = instr;
      const Shape& slice_shape = shape_covering_instr->shape();
      const int64_t num_layers = instr->operand(0)->shape().dimensions(0);
      pattern_info.unstacked_shape =
          MakeUnstackedShapeFromSlice(slice_shape, num_layers);
      pattern_info.unstacking_computation =
          instr->fused_instructions_computation();
      pattern_info.unstacked_instrs.push_back(instr);
      return pattern_info;
    }
  }
  return std::nullopt;
}

absl::Status UnstackDSFusionPattern(
    HloInstruction* mutable_dynamic_slicing_fusion, const Shape& slice_shape) {
  HloComputation* parent_loop = mutable_dynamic_slicing_fusion->parent();

  HloInstruction* stacked = mutable_dynamic_slicing_fusion->mutable_operand(0);
  HloInstruction* offset = mutable_dynamic_slicing_fusion->mutable_operand(1);

  HloInstruction* new_operand =
      parent_loop->AddInstruction(HloInstruction::CreateCustomCall(
          slice_shape, {stacked, offset}, "DynamicGte"));

  HloInstruction* bitcast = mutable_dynamic_slicing_fusion->AddInstruction(
      HloInstruction::CreateBitcast(mutable_dynamic_slicing_fusion->shape(),
                                    new_operand));
  return mutable_dynamic_slicing_fusion->ReplaceAllUsesWithDifferentShape(
      bitcast);
}

// This function recognizes fusions with the following pattern:
// fusion(stacked, f(loop_iteration_var))
// computation {
//   p0 = parameter(0)
//   p1 = parameter(1)
//   ROOT slice = dynamic_slice(p0, p1, zero, ...)
// }
// where f is a function of loop_iteration_var. It indicates that the slicing
// offset is effectively static after unrolling.
std::optional<PatternInfo> GetDSFusionNoBitcastPattern(
    const UnstackerMetadata& metadata, const HloInstruction* instr,
    int64_t stacked_operand_idx) {
  VLOG(3) << "Checking DSFusionNoBitcast";
  HloInstruction* shape_covering_instr =
      GetMostMajorEffectivelyStaticDynamicSliceInFusion(metadata, instr, 2,
                                                        stacked_operand_idx);
  if (shape_covering_instr == nullptr) {
    return std::nullopt;
  }
  if (instr->fused_instructions_computation()->root_instruction() !=
      shape_covering_instr) {
    return std::nullopt;
  }
  PatternInfo pattern_info;
  pattern_info.type = PatternType::DSFusionNoBitcastPattern;
  pattern_info.instr = instr;
  const Shape& slice_shape = shape_covering_instr->shape();
  const int64_t num_layers = instr->operand(0)->shape().dimensions(0);
  pattern_info.unstacked_shape =
      MakeUnstackedShapeFromSlice(slice_shape, num_layers);
  pattern_info.unstacking_computation = instr->fused_instructions_computation();
  pattern_info.unstacked_instrs.push_back(instr);
  return pattern_info;
}

absl::Status UnstackDSFusionNoBitcastPattern(
    HloInstruction* mutable_dynamic_slicing_fusion, const Shape& slice_shape) {
  HloComputation* parent_loop = mutable_dynamic_slicing_fusion->parent();

  HloInstruction* stacked = mutable_dynamic_slicing_fusion->mutable_operand(0);
  HloInstruction* offset = mutable_dynamic_slicing_fusion->mutable_operand(1);

  HloInstruction* new_operand =
      parent_loop->AddInstruction(HloInstruction::CreateCustomCall(
          slice_shape, {stacked, offset}, "DynamicGte"));

  return mutable_dynamic_slicing_fusion->ReplaceAllUsesWithDifferentShape(
      new_operand);
}

// This function recognizes fusions with the following pattern:
// fusion(stacked, update, loop_iteration_var)
// computation {
//   p0 = parameter(0)
//   p1 = parameter(1)
//   p2 = parameter(2)
//   update = bitcast(p1)
//   ROOT dus = dynamic_update_slice(p0, update, p2, zero, ...)
// }
std::optional<PatternInfo> GetDUSFusionPattern(
    const UnstackerMetadata& metadata, const HloInstruction* instr,
    int64_t stacked_operand_idx) {
  VLOG(3) << "Checking DUSFusion";
  HloInstruction* shape_covering_instr =
      GetMostMajorShapeCoveringDynamicIndexInFusion(
          metadata, instr, HloOpcode::kDynamicUpdateSlice, 3,
          stacked_operand_idx);
  if (shape_covering_instr == nullptr) {
    return std::nullopt;
  }
  if (Match(shape_covering_instr->operand(1),
            match::Bitcast(match::Parameter()))) {
    if (shape_covering_instr->parent()->root_instruction() ==
        shape_covering_instr) {
      PatternInfo pattern_info;
      pattern_info.type = PatternType::Other;
      pattern_info.instr = instr;
      pattern_info.unstacked_shape = MakeUnstackedShapeFromSlice(
          instr->operand(2)->shape(), instr->operand(0)->shape().dimensions(0));
      pattern_info.unstacking_computation = nullptr;
      pattern_info.unstacked_instrs.push_back(instr);
      return pattern_info;
    }
  }
  return std::nullopt;
}

absl::Status UnstackDUSFusionPattern(
    HloInstruction* mutable_dynamic_update_slicing_fusion,
    const Shape& slice_shape) {
  HloComputation* parent_loop = mutable_dynamic_update_slicing_fusion->parent();
  // TODO: (b/350043079) - automatically find the input, offset and update
  // indices.
  HloInstruction* stacked =
      mutable_dynamic_update_slicing_fusion->mutable_operand(0);
  HloInstruction* offset =
      mutable_dynamic_update_slicing_fusion->mutable_operand(1);
  HloInstruction* update =
      mutable_dynamic_update_slicing_fusion->mutable_operand(2);
  HloInstruction* new_operand =
      parent_loop->AddInstruction(HloInstruction::CreateCustomCall(
          stacked->shape(), {stacked, update, offset}, "DynamicTuple"));
  for (HloInstruction* user : mutable_dynamic_update_slicing_fusion->users()) {
    TF_RETURN_IF_ERROR(
        mutable_dynamic_update_slicing_fusion->ReplaceUseWithDifferentShape(
            user, new_operand));
  }
  return absl::OkStatus();
}

// This function recognizes fusions with the following pattern:
// fusion(stackd, update, loop_iteration_var)
// computation {
//   p0 = parameter(0)
//   p1 = parameter(1)
//   p2 = parameter(2)
//   pad = pad(p1, ...)
//   update = bitcast(pad)
//   ROOT dus = dynamic_update_slice(p0, update, p2, zero, ...)
// }
std::optional<PatternInfo> GetDUSFusionWithPadPattern(
    const UnstackerMetadata& metadata, const HloInstruction* instr,
    int64_t stacked_operand_idx) {
  VLOG(3) << "Checking DUSFusionWithPad";
  HloInstruction* shape_covering_instr =
      GetMostMajorShapeCoveringDynamicIndexInFusion(
          metadata, instr, HloOpcode::kDynamicUpdateSlice, 3,
          stacked_operand_idx);
  if (shape_covering_instr == nullptr) {
    return std::nullopt;
  }
  if (Match(
          shape_covering_instr->operand(1),
          match::Bitcast(match::Pad(match::Parameter(), match::Constant())))) {
    if (shape_covering_instr->parent()->root_instruction() ==
        shape_covering_instr) {
      const HloInstruction* pad_instr =
          shape_covering_instr->operand(1)->operand(0);
      PatternInfo pattern_info;
      pattern_info.type = PatternType::Other;
      pattern_info.instr = instr;
      pattern_info.unstacked_shape = MakeUnstackedShapeFromSlice(
          pad_instr->shape(),
          shape_covering_instr->operand(0)->shape().dimensions(0));
      pattern_info.unstacking_computation = nullptr;
      pattern_info.unstacked_instrs.push_back(instr);
      return pattern_info;
    }
  }
  return std::nullopt;
}

// Unstacks the DUSFusionWithPad pattern by removing the dynamic-update-slice
// from the fusion and feeding the padding fusion to the dynamic-tuple
// custom-call.
absl::Status UnstackDUSFusionWithPadPattern(
    HloInstruction* mutable_dynamic_update_slicing_fusion,
    const Shape& slice_shape) {
  HloComputation* parent_loop = mutable_dynamic_update_slicing_fusion->parent();
  HloComputation* fused_computation =
      mutable_dynamic_update_slicing_fusion->fused_instructions_computation();
  HloInstruction* stacked =
      mutable_dynamic_update_slicing_fusion->mutable_operand(
          fused_computation->root_instruction()
              ->mutable_operand(0)
              ->parameter_number());
  HloInstruction* offset =
      mutable_dynamic_update_slicing_fusion->mutable_operand(
          fused_computation->root_instruction()
              ->mutable_operand(2)
              ->parameter_number());

  HloInstruction* pad_instr = fused_computation->root_instruction()
                                  ->mutable_operand(1)
                                  ->mutable_operand(0);
  fused_computation->set_root_instruction(pad_instr, true);
  *mutable_dynamic_update_slicing_fusion->mutable_shape() = pad_instr->shape();

  HloInstruction* new_operand =
      parent_loop->AddInstruction(HloInstruction::CreateCustomCall(
          stacked->shape(),
          {stacked, mutable_dynamic_update_slicing_fusion, offset},
          "DynamicTuple"));
  for (HloInstruction* user : mutable_dynamic_update_slicing_fusion->users()) {
    if (user != new_operand) {
      TF_RETURN_IF_ERROR(
          mutable_dynamic_update_slicing_fusion->ReplaceUseWithDifferentShape(
              user, new_operand));
    }
  }
  return absl::OkStatus();
}

// This function recognizes fusions with the following pattern:
// fusion(stackd, update, loop_iteration_var)
// computation {
//   p0 = parameter(0)
//   p1 = parameter(1)
//   slice = dynamic-slice(p0, p1, zero)
//   broadcast = broadcast(constant)
//   add = add(slice, broadcast)
//   ROOT reduce = reduce(add, zero), apply=+
// }
std::optional<PatternInfo> GetDSFusionWithAddPattern(
    const UnstackerMetadata& metadata, const HloInstruction* instr,
    int64_t stacked_operand_idx) {
  VLOG(3) << "Checking DSFusionWithAdd";
  HloInstruction* shape_covering_instr =
      GetMostMajorShapeCoveringDynamicIndexInFusion(
          metadata, instr, HloOpcode::kDynamicSlice, 2, stacked_operand_idx);
  if (shape_covering_instr == nullptr) {
    return std::nullopt;
  }
  HloComputation* fused_computation = instr->fused_instructions_computation();
  HloInstruction* fusion_root = fused_computation->root_instruction();
  HloInstruction* add_operand;
  if (Match(fusion_root,
            match::Reduce(match::Add(match::Op(&add_operand),
                                     match::Broadcast(match::Constant())),
                          match::Constant()))) {
    if (add_operand == shape_covering_instr) {
      const int64_t num_layers = instr->operand(0)->shape().dimensions(0);
      PatternInfo pattern_info;
      pattern_info.type = PatternType::Other;
      pattern_info.instr = instr;
      pattern_info.unstacked_shape =
          MakeUnstackedShapeFromSlice(instr->shape(), num_layers);
      HloComputation::Builder builder("unstack_add");
      HloInstruction* p0 =
          builder.AddInstruction(HloInstruction::CreateParameter(
              0, fused_computation->parameter_instruction(0)->shape(), "p0"));
      HloInstruction* p1 =
          builder.AddInstruction(HloInstruction::CreateParameter(
              1, fused_computation->parameter_instruction(1)->shape(), "p1"));
      HloInstruction* zero =
          builder.AddInstruction(MakeScalarConstantWithShape(p1->shape(), 0));
      std::vector<HloInstruction*> slice_starts;
      slice_starts.reserve(shape_covering_instr->shape().dimensions().size());
      slice_starts.push_back(p1);
      for (int64_t i = 0;
           i < static_cast<int64_t>(
                   shape_covering_instr->shape().dimensions().size()) -
                   1;
           i++) {
        slice_starts.push_back(zero);
      }
      HloInstruction* slice =
          builder.AddInstruction(HloInstruction::CreateDynamicSlice(
              shape_covering_instr->shape(), p0, slice_starts,
              shape_covering_instr->dynamic_slice_sizes()));
      HloInstruction* zero_reduce =
          builder.AddInstruction(MakeScalarConstantWithShape(
              ShapeUtil::MakeScalarShape(slice->shape().element_type()), 0));
      HloInstruction* reduce =
          builder.AddInstruction(HloInstruction::CreateReduce(
              instr->shape(), slice, zero_reduce, fusion_root->dimensions(),
              fused_computation->root_instruction()->to_apply()));
      HloComputation* unstack_add =
          instr->GetModule()->AddEmbeddedComputation(builder.Build());
      unstack_add->set_root_instruction(reduce);
      pattern_info.unstacking_computation = unstack_add;
      pattern_info.unstacked_instrs.push_back(instr);
      return pattern_info;
    }
  }
  return std::nullopt;
}

absl::Status UnstackDSFusionWithAddPattern(
    HloInstruction* mutable_dynamic_slice_with_add_fusion,
    const Shape& slice_shape) {
  HloComputation* parent_loop = mutable_dynamic_slice_with_add_fusion->parent();
  HloInstruction* stacked =
      mutable_dynamic_slice_with_add_fusion->mutable_operand(0);
  HloInstruction* offset =
      mutable_dynamic_slice_with_add_fusion->mutable_operand(1);
  HloInstruction* new_operand =
      parent_loop->AddInstruction(HloInstruction::CreateCustomCall(
          slice_shape, {stacked, offset}, "DynamicGte"));
  HloInstruction* one = parent_loop->AddInstruction(MakeScalarConstantWithShape(
      ShapeUtil::MakeScalarShape(slice_shape.element_type()), 1));
  HloInstruction* broadcast = parent_loop->AddInstruction(
      HloInstruction::CreateBroadcast(slice_shape, one, {}));
  HloInstruction* add = mutable_dynamic_slice_with_add_fusion->AddInstruction(
      HloInstruction::CreateBinary(new_operand->shape(), HloOpcode::kAdd,
                                   new_operand, broadcast));
  TF_RETURN_IF_ERROR(
      mutable_dynamic_slice_with_add_fusion->ReplaceAllUsesWith(add));
  return absl::OkStatus();
}

// This method checks if the given instruction is a fusion with the following
// properties:
// 1. It is inside the body of an unrollable loop
// 2. The parameter at stacked_operand_index has a single user inside the
//    fused computation.
// 3. The single user is a fusion with two operands with the following form:
//    fusion(stacked_param, slicing_offset)
//    (We assume that the stacked parameter is always the first operand and
//    the slicing offset is the second operand.)
// 4. The fusion user contains a shape-covering dynamic-slice instruction.
std::optional<PatternInfo> GetNestedDSFusionPattern(
    const UnstackerMetadata& metadata, const HloInstruction* instr,
    int64_t stacked_operand_idx) {
  if (instr->opcode() != HloOpcode::kFusion) {
    return std::nullopt;
  }
  if (!metadata.unrollable_loop_bodies.contains(instr->parent())) {
    VLOG(5) << "Instruction not inside unrollable while body, " << instr->name()
            << " inside " << instr->parent()->name();
    return std::nullopt;
  }

  WhileLoopConfig while_instr_config =
      metadata.unrollable_loop_bodies.at(instr->parent());

  VLOG(3) << "Checking NestedDSFusionPattern";

  HloInstruction* inner_fusion_user = nullptr;
  for (HloInstruction* fused_instr :
       instr->fused_instructions_computation()->MakeInstructionPostOrder()) {
    // Find the changed parameter in the fused computation
    if (Match(fused_instr, match::Parameter(stacked_operand_idx))) {
      // There must be a single fusion user
      if (fused_instr->user_count() != 1) {
        return std::nullopt;
      }
      if (Match(fused_instr->users()[0],
                match::Fusion(match::Op(), match::Op()))) {
        inner_fusion_user = fused_instr->users()[0];
        break;
      }
    }
  }
  if (inner_fusion_user == nullptr) {
    return std::nullopt;
  }
  for (HloInstruction* inner_fusion_instr :
       inner_fusion_user->fused_instructions_computation()
           ->MakeInstructionPostOrder()) {
    if (!Match(inner_fusion_instr, match::DynamicSlice())) {
      continue;
    }
    std::optional<int64_t> dynamic_index =
        MatchEffectivelyStaticDynamicSliceInsideLoop(
            inner_fusion_instr,
            inner_fusion_user->fused_instructions_computation()
                ->parameter_instruction(0),
            while_instr_config);
    if (dynamic_index.has_value() && dynamic_index.value() == 0) {
      const int64_t num_layers =
          inner_fusion_user->operand(0)->shape().dimensions(0);
      PatternInfo pattern_info;
      pattern_info.type = PatternType::NestedDSFusionPattern;
      pattern_info.instr = inner_fusion_user;
      pattern_info.unstacked_shape =
          MakeUnstackedShapeFromSlice(inner_fusion_instr->shape(), num_layers);
      pattern_info.unstacking_computation =
          inner_fusion_user->fused_instructions_computation();
      pattern_info.unstacked_instrs.push_back(inner_fusion_user);
      return pattern_info;
    }
  }
  return std::nullopt;
}

// The function below captures all the changes necessary to hlo graph for it's
// corresponding (GetNestedDSFusionPattern) pattern to unstack.
absl::Status UnstackNestedDSFusionPattern(
    HloInstruction* mutable_dynamic_slicing_fusion, const Shape& slice_shape) {
  // We are sure that this lambda is called with a nested fusion.
  HloInstruction* parent_fusion =
      mutable_dynamic_slicing_fusion->parent()->FusionInstruction();

  // Under the assumption that the stacked parameter is always the first
  // operand of the inner fusion.
  HloInstruction* stacked_in_ds_fusion =
      mutable_dynamic_slicing_fusion->mutable_operand(0);
  CHECK_EQ(stacked_in_ds_fusion->opcode(), HloOpcode::kParameter);
  int64_t stacked_param_number = stacked_in_ds_fusion->parameter_number();
  HloInstruction* stacked =
      parent_fusion->mutable_operand(stacked_param_number);

  // Under the assumption that the slicing offset is always the second
  // operand of the inner fusion.
  HloInstruction* offset_in_ds_fusion =
      mutable_dynamic_slicing_fusion->mutable_operand(1);
  CHECK_EQ(offset_in_ds_fusion->opcode(), HloOpcode::kParameter);
  HloInstruction* offset =
      parent_fusion->mutable_operand(offset_in_ds_fusion->parameter_number());

  HloInstruction* sliced_param =
      parent_fusion->fused_instructions_computation()->ReplaceParameter(
          stacked_param_number,
          HloInstruction::CreateParameter(stacked_param_number, slice_shape,
                                          "sliced"));
  HloInstruction* bitcast = mutable_dynamic_slicing_fusion->AddInstruction(
      HloInstruction::CreateBitcast(mutable_dynamic_slicing_fusion->shape(),
                                    sliced_param));
  HloInstruction* bitcast_fusion =
      mutable_dynamic_slicing_fusion->AddInstruction(
          HloInstruction::CreateFusion(mutable_dynamic_slicing_fusion->shape(),
                                       HloInstruction::FusionKind::kLoop,
                                       bitcast));
  TF_RETURN_IF_ERROR(
      mutable_dynamic_slicing_fusion->ReplaceAllUsesWith(bitcast_fusion));
  // Create the custom-call to dynamically get the tuple element given the
  // loop iteration number. We rely on WhileLoopUnroller to rewrite this as
  // a get-tuple-element hlo once the iteration number is known and loop
  // bodies are unrolled.
  HloInstruction* new_operand =
      parent_fusion->AddInstruction(HloInstruction::CreateCustomCall(
          slice_shape, {stacked, offset}, "DynamicGte"));
  return parent_fusion->ReplaceOperandWithDifferentShape(
      sliced_param->parameter_number(), new_operand);
}

// Identifies the following pattern:
//  computation {
//     ...
//     fusion.1 = fusion(...stacked...) // this is GetDSFusionPattern
//     fusion.2 = fusion(...stacked...) // this is GetDUSFusionPattern
//     ...
//   }
std::optional<PatternInfo> GetDSAndDUSPattern(const UnstackerMetadata& metadata,
                                              const HloInstruction* instr,
                                              int64_t stacked_operand_idx) {
  VLOG(3) << "Checking DSAndDUSPattern";
  if (instr->opcode() != HloOpcode::kFusion) {
    return std::nullopt;
  }
  const HloInstruction* stacked = instr->operand(stacked_operand_idx);
  if (stacked->user_count() != 2) {
    return std::nullopt;
  }

  HloInstruction* shape_covering_ds_instr =
      GetMostMajorShapeCoveringDynamicIndexInFusion(
          metadata, instr, HloOpcode::kDynamicSlice, 2, stacked_operand_idx);
  if (shape_covering_ds_instr == nullptr) {
    return std::nullopt;
  }
  HloInstruction* bitcast_operand = nullptr;
  if (!Match(instr->fused_instructions_computation()->root_instruction(),
             match::Bitcast(match::Op(&bitcast_operand)))) {
    return std::nullopt;
  }
  if (bitcast_operand != shape_covering_ds_instr) {
    return std::nullopt;
  }
  if (!GetDUSFusionPattern(metadata, stacked->users()[1],
                           stacked->users()[1]->operand_index(stacked))) {
    return std::nullopt;
  }
  PatternInfo pattern_info;
  pattern_info.type = PatternType::Other;
  pattern_info.instr = instr;
  const Shape& slice_shape = instr->shape();
  const int64_t num_layers = instr->operand(0)->shape().dimensions(0);
  pattern_info.unstacked_shape =
      MakeUnstackedShapeFromSlice(slice_shape, num_layers);
  pattern_info.unstacking_computation = instr->fused_instructions_computation();
  pattern_info.unstacked_instrs.push_back(instr);
  pattern_info.unstacked_instrs.push_back(stacked->users()[1]);
  return pattern_info;
}

absl::Status UnstackDSAndDUSPattern(HloInstruction* mutable_dynamic_slice,
                                    const Shape& slice_shape) {
  HloInstruction* stacked_gte = mutable_dynamic_slice->mutable_operand(0);
  int64_t stacked_gte_index = stacked_gte->tuple_index();
  HloComputation* parent = stacked_gte->parent();
  ShapeUtil::UpdateTupleShape(stacked_gte->shape(), stacked_gte_index,
                              parent->root_instruction()->mutable_shape());

  HloComputation* parent_loop = mutable_dynamic_slice->parent();
  HloInstruction* stacked = mutable_dynamic_slice->mutable_operand(0);
  HloInstruction* offset = mutable_dynamic_slice->mutable_operand(1);
  HloInstruction* new_operand =
      parent_loop->AddInstruction(HloInstruction::CreateCustomCall(
          slice_shape, {stacked, offset}, "DynamicGte"));
  TF_RETURN_IF_ERROR(
      mutable_dynamic_slice->ReplaceAllUsesWithDifferentShape(new_operand));

  HloInstruction* mutable_dynamic_update_slice = stacked_gte->users()[1];
  TF_RETURN_IF_ERROR(
      UnstackDUSFusionPattern(mutable_dynamic_update_slice, slice_shape));
  return absl::OkStatus();
}

// This function recognizes fusions with the following pattern:
// fusion(stacked, loop_iteration_var)
// computation {
//   p0 = parameter(0)
//   p1 = parameter(1)
//   slice = dynamic_slice(p0, p1, zero, ...)
//   ROOT reduce = reduce(slice, constant)
// }
std::optional<PatternInfo> GetReduceFusionPattern(
    const UnstackerMetadata& metadata, const HloInstruction* instr,
    int64_t stacked_operand_idx) {
  VLOG(3) << "Checking ReduceFusion";
  HloInstruction* shape_covering_instr =
      GetMostMajorShapeCoveringDynamicIndexInFusion(
          metadata, instr, HloOpcode::kDynamicSlice, 2, stacked_operand_idx);
  if (shape_covering_instr == nullptr) {
    return std::nullopt;
  }
  if (!ShouldUnfuseSlices(metadata, shape_covering_instr)) {
    return std::nullopt;
  }
  HloInstruction* reduce_operand = nullptr;
  HloInstruction* fusion_root =
      instr->fused_instructions_computation()->root_instruction();
  if (Match(fusion_root, match::Reduce(match::Op(&reduce_operand),
                                       match::ConstantScalar())) &&
      Match(fusion_root->to_apply()->root_instruction(),
            match::Add(match::Parameter(), match::Parameter()))) {
    if (reduce_operand == shape_covering_instr) {
      PatternInfo pattern_info;
      pattern_info.type = PatternType::Other;
      pattern_info.instr = instr;
      const Shape& slice_shape = instr->shape();
      const int64_t num_layers = instr->operand(0)->shape().dimensions(0);
      pattern_info.unstacked_shape =
          MakeUnstackedShapeFromSlice(slice_shape, num_layers);
      pattern_info.unstacking_computation =
          instr->fused_instructions_computation();
      pattern_info.unstacked_instrs.push_back(instr);
      return pattern_info;
    }
  }

  return std::nullopt;
}

absl::Status UnstackReduceFusionPattern(HloInstruction* mutable_reduce_fusion,
                                        const Shape& slice_shape) {
  HloComputation* parent_loop = mutable_reduce_fusion->parent();

  HloInstruction* stacked = mutable_reduce_fusion->mutable_operand(0);
  HloInstruction* offset = mutable_reduce_fusion->mutable_operand(1);

  HloInstruction* new_operand =
      parent_loop->AddInstruction(HloInstruction::CreateCustomCall(
          slice_shape, {stacked, offset}, "DynamicGte"));
  return mutable_reduce_fusion->ReplaceAllUsesWithDifferentShape(new_operand);
}

};  // namespace

// The entry point of the unstacking algorithm. Given a module, it creates the
// unstacking metadata and populates the unstacking custom handler(s). Moreover,
// it attempts unstacking each index of the loops in the entry computation of
// the module. Finally, it removes the unused computations and unrolls the
// module.
absl::StatusOr<bool> HloUnstacker::Run(
    HloModule* module,
    const absl::flat_hash_set<absl::string_view>& execution_threads) {
  TF_ASSIGN_OR_RETURN(auto metadata,
                      UnstackerMetadata::Create(module, unfuse_slice_));
  // The order of the patterns below is important, as it determines the order
  // in which the unstacking custom handlers are called. For example, applying
  // GetDSAndDUSPattern after GetDSFusionPattern would result in patterns of
  // GetDSAndDUSPattern not being recognized since GetDSFusionPattern is a
  // sub-pattern of GetDSAndDUSPattern.
  metadata.custom_handlers.push_back(
      std::make_pair(GetDSAndDUSPattern, UnstackDSAndDUSPattern));
  metadata.custom_handlers.push_back(
      std::make_pair(GetDSFusionPattern, UnstackDSFusionPattern));
  metadata.custom_handlers.push_back(
      std::make_pair(GetDUSFusionPattern, UnstackDUSFusionPattern));
  metadata.custom_handlers.push_back(std::make_pair(
      GetDUSFusionWithPadPattern, UnstackDUSFusionWithPadPattern));
  metadata.custom_handlers.push_back(
      std::make_pair(GetDSFusionWithAddPattern, UnstackDSFusionWithAddPattern));
  metadata.custom_handlers.push_back(
      std::make_pair(GetReduceFusionPattern, UnstackReduceFusionPattern));
  metadata.custom_handlers.push_back(
      std::make_pair(GetNestedDSFusionPattern, UnstackNestedDSFusionPattern));
  metadata.custom_handlers.push_back(std::make_pair(
      GetDSFusionNoBitcastPattern, UnstackDSFusionNoBitcastPattern));

  std::vector<HloInstruction*> entry_loops;
  for (HloInstruction* instr :
       module->entry_computation()->MakeInstructionPostOrder()) {
    // Only unstack standard loops with tuple input and output.
    if (Match(instr, match::While(match::Tuple())) &&
        Match(instr->while_body()->root_instruction(), match::Tuple())) {
      entry_loops.push_back(instr);
    }
  }

  int64_t num_unstacked = 0;
  bool unstacked = false;
  std::vector<const HloInstruction*> unstacked_instructions;
  for (HloInstruction* loop : entry_loops) {
    for (int64_t i = 0; i < loop->shape().tuple_shapes_size(); ++i) {
      // We don't handle tuples and if we see then we assume they come from a
      // previous unstacking attempt.
      if (loop->while_init()->operand(i)->shape().IsTuple()) {
        continue;
      }
      VLOG(3) << "Attempting to unstack " << loop->name() << " at " << i
              << " = " << loop->while_init()->operand(i)->shape().ToString(true)
              << loop->while_init()->operand(i)->ToShortString();
      bool current_unstacked =
          UnstackWhileOperandAtIndex(metadata, loop, i, unstacked_instructions);
      if (current_unstacked) {
        num_unstacked++;
        unstacked = true;
      }
      VLOG(3) << "###################";
    }
  }
  if (!unstacked) {
    return false;
  }
  // Unstacking computations are cloned, leaving the original unstacking
  // computation unused.
  TF_RETURN_IF_ERROR(module->RemoveUnusedComputations());
  // We rely on the WhileLoopUnroller pass to unroll unstacked loop bodies
  // and rewrite custom-calls created by unstacker, i.e., DynamicGte and
  // DynamicTuple.
  std::vector<HloInstruction*> loops_to_unroll;
  for (const HloInstruction* instr : unstacked_instructions) {
    HloInstruction* loop = metadata.bodies[instr->parent()];
    if (std::find(loops_to_unroll.begin(), loops_to_unroll.end(), loop) ==
        loops_to_unroll.end()) {
      loops_to_unroll.push_back(loop);
    }
  }
  // Go over the loops in reverse order to unroll the inner loops first.
  for (int64_t i = loops_to_unroll.size() - 1; i >= 0; --i) {
    HloInstruction* loop = loops_to_unroll[i];
    TF_ASSIGN_OR_RETURN(UnrollResult unroll_result,
                        WhileLoopUnroller::UnrollAndReturnReplacement(
                            loop, /*unroll_factor=*/-1,
                            /*wrap_in_trivial_loop=*/false,
                            /*force_unroll=*/true, /*prepare=*/false));
    bool unrolled = unroll_result.unrolled;
    CHECK(unrolled);
  }
  VLOG(3) << "after unstacking \n" << module->ToString();
  VLOG(3) << "Num unstacked: " << num_unstacked;
  return true;
}

}  // namespace xla
