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

#include <cstdint>
#include <deque>
#include <functional>
#include <memory>
#include <optional>
#include <utility>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
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

// TODO(b/342457472): Remove this struct and move its field to the
// UnstackerTransformer as static members. A struct that holds the required
// information for unstacking that is fixed across different unstacker
// instastances.
struct UnstackerMetadata {
  static absl::StatusOr<UnstackerMetadata> Create(HloModule* module) {
    UnstackerMetadata metadata;
    TF_ASSIGN_OR_RETURN(
        bool prepared,
        WhileLoopUnroller::PrepareModuleForUnrolling(module, {}));
    if (prepared) {
      VLOG(3) << "Prepared module: " << module->name() << " for unstacking.";
    }
    std::vector<std::pair<HloInstruction*, WhileLoopConfig>> loops =
        WhileLoopUnroller::GetUnrollableLoops(module, {});
    for (const auto& [instr, while_loop_config] : loops) {
      metadata.unrollable_loop_bodies[instr->while_body()] = while_loop_config;
    }
    return metadata;
  }
  absl::flat_hash_map<HloComputation*, WhileLoopConfig> unrollable_loop_bodies;
  // A pair of custom pattern and its handler lambda that describes the
  // transformation needed to unstack the hlo graph for the pattern.
  std::pair<std::function<const HloInstruction*(
                const UnstackerMetadata&, const HloInstruction*, int64_t)>,
            std::function<absl::Status(HloInstruction*, const Shape&)>>
      custom_handler;
};

// A struct that holds the required information for two-step unstacking. The
// content of each instance differs for each operand of a while loop.
struct UnstackerTransformer {
  UnstackerMetadata metadata;
  static absl::StatusOr<UnstackerTransformer> Create(
      const UnstackerMetadata& c) {
    UnstackerTransformer transformer;
    transformer.metadata = std::move(c);
    return transformer;
  }

  // Given an instruction and the index of the its changed operand, it applies
  // the custom handler and populates body_changes lambdas that unstacks the hlo
  // graph accordingly.
  bool HandleInstruction(const HloInstruction* instr, int64_t changed_idx) {
    VLOG(3) << "HandleInstruction(" << instr->shape().ToString()
            << instr->name() << ", " << changed_idx << ")";

    auto custom_pattern = metadata.custom_handler.first;
    auto custom_handler = metadata.custom_handler.second;

    const HloInstruction* stacked_user =
        custom_pattern(metadata, instr, changed_idx);
    if (stacked_user == nullptr) {
      return false;
    }
    if (unstacking_computation != nullptr) {
      LOG(ERROR) << "Seen multiple users, cannot handle. \n instr: "
                 << instr->ToString() << "\n hoisted_computation: "
                 << unstacking_computation->ToString(
                        HloPrintOptions::Fingerprint());
      return false;
    }

    unstacking_computation =
        stacked_user->fused_instructions_computation()->Clone(
            "hoisted_unstacking");
    VLOG(3) << "Unstacking computation: "
            << unstacking_computation->ToString(HloPrintOptions::Fingerprint());

    // TODO(b/342440749): Currently, we assume the stacked dimension is always
    // the most major dimension. This condition can be checked and terminate
    // unstacking if not met.
    Shape slice_shape = stacked_user->shape();
    int64_t num_layers = stacked_user->operand(0)->shape().dimensions(0);
    std::vector<Shape> shapes;
    for (int64_t i = 0; i < num_layers; ++i) {
      shapes.push_back(slice_shape);
    }
    unstacked_shape =
        std::make_unique<Shape>(ShapeUtil::MakeTupleShape(shapes));

    // Wrapper function around the unstacker lambda which calls the unstacker.
    std::function<absl::Status()> unstack_wrapper =
        [=]() mutable -> absl::Status {
      HloInstruction* mutable_dynamic_slicing_fusion =
          const_cast<HloInstruction*>(stacked_user);
      return custom_handler(mutable_dynamic_slicing_fusion, slice_shape);
    };
    body_changes.push_back(unstack_wrapper);
    return true;
  }

  // This pointer is populated if the unstacker finds unstackable loop input.
  std::unique_ptr<Shape> unstacked_shape = nullptr;
  // This is a pointer to the computation that is responsible for unstacking. It
  // is used to hoist the unstacking computations outside the loop bodies.
  std::unique_ptr<HloComputation> unstacking_computation = nullptr;
  // A vector of lambdas that describe necessary changes to the shape of the
  // loops to unstack. The lambdas accept the pointer to the new unstacked
  // shape.
  std::vector<std::function<void(const Shape*)>> loop_changes;
  // a list of lambdas that captures all the changes to the hlo graph needed for
  // unstacking.
  std::vector<std::function<absl::Status()>> body_changes;
  // A map that tracks the index of the changed operand for instructions of type
  // get-tuple-element, tuple, and while during unstacking.
  absl::flat_hash_map<HloInstruction*, int64_t> operand_changes;
};

bool CanUnstackWhileOperand(const HloInstruction* while_instr,
                            UnstackerTransformer& unstacker, int64_t index);

// Given a gte and an unstacker instance, this function walks down the graph of
// the users in BFS manner and propagates the index of the changed input operand
// for kGetTupleElement, kTuple, and kWhile instructions. Moreover, if checks if
// the a user should be handled with the provided custom handler(s) inside the
// unstacker instance. Note that this function does NOT change the shape of any
// instruction, it merely keeps track of the instructions and where in the input
// operands the change need to be applied later.
bool PropagateGteShapeChange(HloInstruction* gte,
                             UnstackerTransformer& unstacker) {
  VLOG(5) << "PropagateGteShapeChange(" << gte->ToString() << ")";

  // TODO(b/343457903): Use HloDataflowAnalysis to track the usage of a value
  // instead of manually applying bfs
  //
  // Apply BFS to propagate the index of the changed operand.
  absl::flat_hash_map<HloInstruction*, int64_t>& visited =
      unstacker.operand_changes;
  std::deque<HloInstruction*> worklist;
  worklist.push_back(gte);
  visited.insert({gte, gte->tuple_index()});
  while (!worklist.empty()) {
    HloInstruction* changed_instr_to_propagate = worklist.front();
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
        worklist.push_back(user);
      } else if (user->opcode() == HloOpcode::kTuple) {
        int64_t use_index = user->operand_index(changed_instr_to_propagate);
        visited.insert({user, {use_index}});
        worklist.push_back(user);
      } else if (user->opcode() == HloOpcode::kWhile) {
        // Recursively check the inner while for unstacking and populate
        // unstacker instance.
        bool changed_nested_while =
            CanUnstackWhileOperand(user, unstacker, changed_operand_index);
        if (!changed_nested_while) {
          return false;
        }
        visited.insert({user, changed_operand_index});
        worklist.push_back(user);
      } else {
        int64_t use_index = user->operand_index(changed_instr_to_propagate);
        if (!unstacker.HandleInstruction(user, use_index)) {
          VLOG(3) << "Custom unstacker not found for " << user->ToString();
          return false;
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
        VLOG(3) << "Failed to propagate shape change for " << instr->ToString();
        return false;
      }
    }
  }
  VLOG(3) << "Finish propagating shape change of index " << idx
          << " in: " << comp->name();
  return true;
}

bool CanUnstackWhileOperand(const HloInstruction* while_instr,
                            UnstackerTransformer& unstacker, int64_t index) {
  VLOG(5) << "ReplaceWhileOperandShape: " << while_instr->name() << " at "
          << index;

  bool body_changes_collected = CanPropagateGteShapeChangesInComputation(
      while_instr->while_body(),
      while_instr->while_body()->parameter_instruction(0), unstacker, index);

  bool condition_changes_collected = CanPropagateGteShapeChangesInComputation(
      while_instr->while_condition(),
      while_instr->while_condition()->parameter_instruction(0), unstacker,
      index);
  if (body_changes_collected && condition_changes_collected) {
    auto loop_change = [](HloInstruction* loop, const Shape* new_shape,
                          int64_t idx) mutable {
      Shape old_shape = ShapeUtil::MakeStaticShape(
          loop->while_body()->parameter_instruction(0)->shape());
      ShapeUtil::UpdateTupleShape(*new_shape, idx, &old_shape);

      loop->while_body()->ReplaceParameter(
          0, HloInstruction::CreateParameter(0, old_shape, "unstacked"));
      loop->while_condition()->ReplaceParameter(
          0, HloInstruction::CreateParameter(0, old_shape, "unstacked"));
    };
    auto loop_change_wrapper = [=](const Shape* new_shape) {
      HloInstruction* mutable_loop = const_cast<HloInstruction*>(while_instr);
      loop_change(mutable_loop, new_shape, index);
    };
    unstacker.loop_changes.push_back(loop_change_wrapper);
    return true;
  }
  return false;
}

// This function is responsible for:
// 1. Hoisting the unstacking computation outside the while_instr.
// 2. Replacing the input of the while_instr with the new unstacked version.
void UnstackWhileInput(const UnstackerTransformer& unstacker,
                       HloInstruction* while_instr, const Shape* new_shape,
                       int64_t index) {
  const Shape& slice_shape = new_shape->tuple_shapes(0);
  HloInstruction* old_while_input =
      while_instr->while_init()->mutable_operand(index);

  // TODO(b/341815540): Instead of creating the unstacked tuple for every input
  // index, we should reuse if the input and unstacking computations are the
  // same.
  //
  // Hoist the unstacking computation outside the while_instr and create a tuple
  // of slices.
  std::vector<HloInstruction*> slices;
  for (int64_t i = 0; i < new_shape->tuple_shapes_size(); ++i) {
    std::vector<HloInstruction*> operands = {
        old_while_input,
        while_instr->AddInstruction(MakeConstantWithShape(
            unstacker.unstacking_computation->parameter_instruction(1)->shape(),
            i))};
    HloInstruction* slice =
        while_instr->AddInstruction(HloInstruction::CreateFusion(
            slice_shape, HloInstruction::FusionKind::kLoop, operands,
            while_instr->GetModule()->AddEmbeddedComputation(
                unstacker.unstacking_computation->Clone()),
            "hoisted"));
    slices.push_back(slice);
  }
  HloInstruction* new_operand_element =
      while_instr->AddInstruction(HloInstruction::CreateTuple(slices));
  HloInstruction* new_while_init =
      TupleUtil::ReplaceTupleWith(new_operand_element,
                                  while_instr->while_init(), {index}, false)
          .value();
  CHECK_OK(while_instr->ReplaceOperandWithDifferentShape(0, new_while_init));
}

// Apply the two-step unstacking algorithm to the given while_instr at the given
// index.
bool UnstackWhileOperandAtIndex(const UnstackerMetadata& metadata,
                                HloInstruction* while_instr, int64_t index) {
  UnstackerTransformer unstacker =
      UnstackerTransformer::Create(metadata).value();

  // First step of unstacking to determine whether while_instr at index is
  // unstackable.
  bool can_unstack = CanUnstackWhileOperand(while_instr, unstacker, index);
  if (!can_unstack) {
    return false;
  }

  // Check if we can propagate the changes through the output of the while
  // at index.
  bool parent_changes_collected = CanPropagateGteShapeChangesInComputation(
      while_instr->parent(), while_instr, unstacker, index);
  if (!parent_changes_collected) {
    return false;
  }

  // If unstacker has not found an unstackable shape, there is no point in
  // applying the unstacker changes.
  if (unstacker.unstacked_shape == nullptr) {
    return false;
  }

  // At this point, we have the unstacked_shape at hand. We go ahead and apply
  // all the changes that required the unstacked shape.
  //
  // Update the shape of get-tuple-element, tuple, and, while instructions
  // based on the unstacked_shape and the index of the changed operand.
  for (const auto& [instr, index] : unstacker.operand_changes) {
    switch (instr->opcode()) {
      case HloOpcode::kGetTupleElement:
        *instr->mutable_shape() = *unstacker.unstacked_shape;
        break;
      case HloOpcode::kTuple:
        *instr->mutable_shape()->mutable_tuple_shapes(index) =
            *unstacker.unstacked_shape;
        break;
      case HloOpcode::kWhile:
        ShapeUtil::UpdateTupleShape(*unstacker.unstacked_shape, index,
                                    instr->mutable_shape());
        break;
      default:
        LOG(FATAL) << "Unsupported opcode: " << instr->ToString();
    }
  }
  // Apply the changes to the body according to the provided custom handler.
  for (const auto& body_change : unstacker.body_changes) {
    CHECK_OK(body_change());
  }
  // Update the input and output shape of the loop.
  UnstackWhileInput(unstacker, while_instr, unstacker.unstacked_shape.get(),
                    index);
  const Shape& new_while_shape = while_instr->while_init()->shape();
  *while_instr->mutable_shape() = new_while_shape;
  // Apply the changes to the shape of the loop body and condition
  // computations.
  for (auto& loop_change : unstacker.loop_changes) {
    loop_change(unstacker.unstacked_shape.get());
  }
  return true;
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
const HloInstruction* GetNestedDynamicSlicingFusion(
    const UnstackerMetadata& metadata, const HloInstruction* instr,
    int64_t stacked_operand_idx) {
  if (!Match(instr, match::Fusion())) {
    return nullptr;
  }

  if (!metadata.unrollable_loop_bodies.contains(instr->parent())) {
    VLOG(5) << "Instruction not inside unrollable while body, "
            << instr->ToString() << instr->parent()->ToString();
    return nullptr;
  }

  WhileLoopConfig while_instr_config =
      metadata.unrollable_loop_bodies.at(instr->parent());

  HloInstruction* inner_fusion_user = nullptr;
  for (HloInstruction* fused_instr :
       instr->fused_instructions_computation()->MakeInstructionPostOrder()) {
    // Find the changed parameter in the fused computation
    if (Match(fused_instr, match::Parameter(stacked_operand_idx))) {
      // There must be a single fusion user
      if (fused_instr->user_count() != 1) {
        return nullptr;
      }
      if (Match(fused_instr->users()[0],
                match::Fusion(match::Op(), match::Op()))) {
        inner_fusion_user = fused_instr->users()[0];
        break;
      }
    }
  }
  if (inner_fusion_user == nullptr) {
    return nullptr;
  }
  for (HloInstruction* inner_fusion_instr :
       inner_fusion_user->fused_instructions_computation()
           ->MakeInstructionPostOrder()) {
    if (!Match(inner_fusion_instr, match::DynamicSlice())) {
      continue;
    }
    std::optional<int64_t> dynamic_index =
        MatchShapeCoveringDynamicIndexInstruction(
            inner_fusion_instr,
            inner_fusion_user->fused_instructions_computation()
                ->parameter_instruction(0),
            HloOpcode::kDynamicSlice, while_instr_config);
    if (dynamic_index.has_value() && dynamic_index.value() == 0) {
      return inner_fusion_user;
    }
  }
  return nullptr;
}

// The function below captures all the changes necessary to hlo graph for it's
// corresponding (IsNestedDynamicSlicingFusion) pattern to unstack.
absl::Status UnstackNestedDynamicSlicingFusion(
    HloInstruction* mutable_dynamic_slicing_fusion, const Shape& slice_shape) {
  // We are sure that this lambda is called with a nested fusion.
  HloInstruction* parent_fusion =
      mutable_dynamic_slicing_fusion->parent()->FusionInstruction();
  VLOG(3) << "Found shape-covering dynamic slicing fusion inside a fusion: "
          << mutable_dynamic_slicing_fusion->name() << " inside "
          << parent_fusion->name();

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

  TF_RETURN_IF_ERROR(
      mutable_dynamic_slicing_fusion->ReplaceAllUsesWith(sliced_param));
  TF_RETURN_IF_ERROR(
      parent_fusion->fused_instructions_computation()
          ->RemoveInstructionAndUnusedOperands(mutable_dynamic_slicing_fusion));

  std::vector<Shape> parameters =
      parent_fusion->fused_instructions_computation()
          ->ComputeProgramShape()
          .parameters();
  parameters.at(stacked_param_number) = slice_shape;
  *parent_fusion->fused_instructions_computation()
       ->ComputeProgramShape()
       .mutable_parameters() = parameters;

  // Create the custom-call to dynamically get the tuple element given the
  // loop iteration number. We rely on WhileLoopUnroller to rewrite this as
  // a get-tuple-element hlo once the iteration number is known and loop
  // bodies are unrolled.
  HloInstruction* new_operand =
      parent_fusion->AddInstruction(HloInstruction::CreateCustomCall(
          slice_shape, {stacked, offset}, "DynamicGte"));
  return parent_fusion->ReplaceOperandWithDifferentShape(stacked_param_number,
                                                         new_operand);
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
  TF_ASSIGN_OR_RETURN(auto metadata, UnstackerMetadata::Create(module));

  // Custom handler is a pair of pattern and transformation function that
  // captures different cases of unstacking. It is decoupled from the unstacking
  // algorithm for modularity.
  metadata.custom_handler = std::make_pair(GetNestedDynamicSlicingFusion,
                                           UnstackNestedDynamicSlicingFusion);

  bool unstacked = false;
  for (HloInstruction* instr :
       module->entry_computation()->MakeInstructionPostOrder()) {
    if (instr->opcode() != HloOpcode::kWhile) {
      continue;
    }
    for (int64_t i = 0; i < instr->shape().tuple_shapes_size(); ++i) {
      VLOG(3) << "Attempting to unstack " << instr->name() << " at " << i
              << " with stacked shape "
              << instr->shape().tuple_shapes(i).ToString();
      if (UnstackWhileOperandAtIndex(metadata, instr, i)) {
        VLOG(3) << "Unstacked " << instr->name() << " at " << i
                << " with stacked shape "
                << instr->shape().tuple_shapes(i).ToString();
        unstacked |= true;
      }
    }
  }
  if (unstacked) {
    // Unstacking computations are cloned, leaving the original unstacking
    // computation unused.
    TF_RETURN_IF_ERROR(module->RemoveUnusedComputations());
    // We rely on the WhileLoopUnroller pass to unroll loop bodies and rewrite
    // custom-calls created by unstacker, i.e., DynamicGte and DynamicTuple.
    TF_RETURN_IF_ERROR(WhileLoopUnroller(-1, true).Run(module).status());
  }
  return unstacked;
}

}  // namespace xla
