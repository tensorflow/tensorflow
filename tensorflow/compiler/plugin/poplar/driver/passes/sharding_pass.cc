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

#include "tensorflow/compiler/plugin/poplar/driver/passes/sharding_pass.h"
#include "tensorflow/compiler/plugin/poplar/driver/tools/find_all_users.h"
#include "tensorflow/compiler/plugin/poplar/driver/tools/util.h"

#include "tensorflow/compiler/xla/service/call_graph.h"
#include "tensorflow/compiler/xla/service/hlo_module.h"

#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/lib/strings/str_util.h"

namespace xla {
namespace poplarplugin {

namespace {

bool IsAllowedTupleSharding(const HloInstruction* inst) {
  switch (inst->opcode()) {
    case HloOpcode::kCall:
    case HloOpcode::kWhile:
    case HloOpcode::kConditional:
    case HloOpcode::kTuple:
    case HloOpcode::kParameter:
    case HloOpcode::kInfeed:
    case HloOpcode::kOutfeed:
    case HloOpcode::kGetTupleElement:
      return true;
    default:
      return false;
  }
}

bool CompatibleShapes(const Shape& l, const Shape& r) {
  // Normal tensors are always acceptable for transferring sharding info
  if (l.IsArray() && r.IsArray()) return true;

  // Tuples must have the same number of components
  auto l_shapes = l.tuple_shapes();
  auto r_shapes = l.tuple_shapes();
  if (l_shapes.size() == r_shapes.size()) {
    return absl::c_equal(
        l_shapes, r_shapes,
        [](const Shape& l, const Shape& r) { return CompatibleShapes(l, r); });
  }

  return false;
}

HloSharding GetDefaultSharding(const Shape& shape) {
  return HloSharding::Single(shape, HloSharding::AssignDevice(0));
}

// Check whether a particular instruction/operand pair has sharding information
// available
bool HasShardingForOperand(const HloInstruction* inst, int operand) {
  const HloInstruction* sharding_inst;
  switch (inst->opcode()) {
    case HloOpcode::kCall: {
      // A call op takes its input sharding from the subcomputation parameter
      auto* comp = inst->to_apply();
      sharding_inst = comp->parameter_instruction(operand);
      break;
    }
    case HloOpcode::kWhile: {
      // A while op takes its input sharding from the subcomputation parameter
      auto* comp = inst->while_body();
      sharding_inst = comp->parameter_instruction(operand);
      break;
    }
    case HloOpcode::kConditional: {
      // A conditional op takes its input sharding from the subcomputation
      // parameter
      auto* comp = inst->true_computation();
      sharding_inst = comp->parameter_instruction(operand);
      break;
    }
    case HloOpcode::kGetTupleElement: {
      // A GTE must be processed in collection with other GTEs, so claim that
      // there is no sharding information available on its input. See the fn
      // CopyShardingFromUsers.
      return false;
    }
    default: {
      // All other ops hold their own input and output sharding information. In
      // most cases the input sharding and output sharding are identical (for
      // instance an elementwise op takes both inputs from the same IPU as the
      // output is on, and all are single tensors).
      sharding_inst = inst;
      break;
    }
  }
  return sharding_inst->has_sharding();
}

// Set the sharding based on opcode type.  Most operations have only single
// sharding.  Call, While, Conditional, Tuple and GetTupleElement ops can have
// tuple-type sharding.  If a single-type op is passed Tuple-like sharding then
// it will go to the device which is the most used in the tuple.
void SetSharding(HloInstruction* inst, const HloSharding& sharding) {
  if (IsAllowedTupleSharding(inst)) {
    inst->set_sharding(sharding);
  } else {
    if (sharding.IsTuple()) {
      inst->set_sharding(sharding.tuple_elements()[0]);
    } else {
      inst->set_sharding(sharding);
    }
  }
}

// Pass in a vector of shardings (tuple or otherwise) and this creates a tuple
// of those inputs, and applies to the instruction.
void SetTupleShardingFromVector(HloInstruction* inst,
                                const std::vector<HloSharding>& shardings) {
  std::vector<HloSharding> all_leaves;
  for (auto& s : shardings) {
    std::vector<HloSharding> leaves;
    if (s.IsTuple()) {
      leaves = s.tuple_elements();
    } else {
      leaves.push_back(s);
    }
    absl::c_copy(leaves, std::back_inserter(all_leaves));
  }

  SetSharding(inst, HloSharding::Tuple(inst->shape(), all_leaves));
}

bool CopyShardingFromUsers(HloInstruction* inst) {
  if (inst->user_count() == 0) {
    return false;
  }

  // If any user's operand input is available then copy the sharding
  for (auto* u : inst->users()) {
    for (int index = 0; index < u->operand_count(); index++) {
      if (u->operand(index) == inst) {
        if (HasShardingForOperand(u, index)) {
          auto sharding = GetShardingForOperand(u, index);
          SetSharding(inst, sharding);
          return true;
        }
      }
    }
  }

  if (!inst->shape().IsTuple()) {
    return false;
  }

  // Otherwise we need to find all of the GTEs that make up the tuple.  We don't
  // need to a GTE for every tuple output, just all of the ones that are present
  // in the graph.  A Tuple may have some of its outputs unused.
  int size = ShapeUtil::TupleElementCount(inst->shape());
  std::vector<HloInstruction*> gtes(size);
  for (auto* u : inst->users()) {
    if (u->opcode() == HloOpcode::kGetTupleElement) {
      if (u->tuple_index() < size) {
        gtes[u->tuple_index()] = u;
      }
    }
  }

  std::vector<HloSharding> gte_sharding;
  for (int gte = 0; gte < size; gte++) {
    auto* user = gtes[gte];
    if (user == nullptr) {
      // Unused tuple outputs are just assigned a default sharding
      auto s = GetDefaultSharding(
          ShapeUtil::GetTupleElementShape(inst->shape(), gte));
      gte_sharding.push_back(s);
    } else {
      if (user->has_sharding()) {
        gte_sharding.push_back(GetShardingForOperand(user, 0));
      } else {
        return false;
      }
    }
  }

  SetTupleShardingFromVector(inst, gte_sharding);
  return true;
}

bool CopyGteShardingFromOperand(HloInstruction* inst) {
  auto* operand = inst->operand(0);
  if (operand->has_sharding()) {
    int64 tuple_index = inst->tuple_index();
    auto s = GetShardingOfOutputTensor(operand);
    if (!s.IsTuple()) {
      s = HloSharding::SingleTuple(operand->shape(), s);
    }
    auto subsharding = s.GetSubSharding(operand->shape(), {tuple_index});
    SetSharding(inst, subsharding);
    return true;
  }

  return false;
}

bool CopyTupleShardingFromOperands(HloInstruction* inst) {
  if (absl::c_all_of(inst->operands(),
                     [](HloInstruction* u) { return u->has_sharding(); })) {
    std::vector<HloSharding> shardings;
    absl::c_transform(
        inst->operands(), std::back_inserter(shardings),
        [](HloInstruction* o) { return GetShardingOfOutputTensor(o); });
    SetTupleShardingFromVector(inst, shardings);
    return true;
  }

  return false;
}

bool CopyShardingFromOperands(HloInstruction* inst) {
  for (int o = 0; o < inst->operand_count(); o++) {
    auto* operand = inst->operand(o);
    if (operand->has_sharding()) {
      if (CompatibleShapes(inst->shape(), operand->shape())) {
        auto s = GetShardingOfOutputTensor(operand);
        SetSharding(inst, s);
        return true;
      }
    }
  }
  return false;
}

}  // namespace

StatusOr<bool> ShardingPass::Run(HloModule* module) {
  std::unique_ptr<CallGraph> call_graph = CallGraph::Build(module);

  // Remove unsupported sharding, and sharding on Tuple shaped ops.  We remove
  // sharding from ops which are allowed tuple-type sharding because their
  // sharding should follow the ops which they are sources/sinks for
  for (auto* comp : module->computations()) {
    for (auto* inst : comp->instructions()) {
      if (inst->has_sharding()) {
        bool remove_sharding = false;

        auto sharding = inst->sharding();
        if (!IsSupportedSharding(sharding)) {
          remove_sharding = true;
        }

        if (IsAllowedTupleSharding(inst)) {
          remove_sharding = true;
        }

        if (inst->opcode() == HloOpcode::kAfterAll) {
          remove_sharding = true;
        }

        if (remove_sharding) {
          inst->clear_sharding();
        }
      }
    }
  }

  if (!HaveSharding(module)) {
    return false;
  }

  auto comps = module->MakeComputationPostOrder();
  for (auto* comp : comps) {
    auto call_graph_node = call_graph->GetNode(comp);

    if (IsPopOpsFusion(comp)) {
      continue;
    }

    // Only call/while/if type computations can be sharded. map/sort/reduce ones
    // take the sharding of the caller
    if (call_graph_node.context() != CallContext::kSequential) {
      continue;
    }

    if (!HaveSharding(comp)) {
      // Do something here - we must apply sharding to a computation - maybe set
      // everything to shard 0? or defer until we have the callsite sharding and
      // copy to the called computation
      //
      // T7656
      continue;
    }

    bool done = false;
    int attempt = 0;
    while (!done) {
      done = true;
      bool made_progress = false;
      for (auto* inst : comp->MakeInstructionPostOrder()) {
        // If an instruction has no operands, and no users but the root Tuple,
        // then assign default sharding
        if (!inst->has_sharding() && inst->operand_count() == 0 &&
            absl::c_all_of(inst->users(), [](const HloInstruction* inst) {
              return inst == inst->parent()->root_instruction() &&
                     inst->opcode() == HloOpcode::kTuple;
            })) {
          SetSharding(inst, GetDefaultSharding(inst->shape()));
        }

        // Try to take sharding from users
        if (!inst->has_sharding()) {
          made_progress = CopyShardingFromUsers(inst);
        }

        // Try to take sharding from operands
        if (!inst->has_sharding()) {
          switch (inst->opcode()) {
            case HloOpcode::kGetTupleElement:
              made_progress = CopyGteShardingFromOperand(inst);
              break;
            case HloOpcode::kTuple:
              made_progress = CopyTupleShardingFromOperands(inst);
              break;
            default:
              made_progress = CopyShardingFromOperands(inst);
              break;
          }
        }

        if (!inst->has_sharding()) {
          done = false;
        }
      }
      if (!done && !made_progress) {
        switch (attempt) {
          case 0:
            // If we pass through the whole computation and cannot assign some
            // of the nodes, then we pick off some non-Tuple nodes and assign
            // them default sharding.  Tuple nodes are not included because
            // they might be mostly ok, but with only one part preventing them
            // from sharding properly.
            for (auto* inst : comp->instructions()) {
              if (!inst->has_sharding() && inst->shape().IsArray()) {
                SetSharding(inst, GetDefaultSharding(inst->shape()));
              }
            }
            break;
          case 1:
            // Should we now apply sharding to Tuples too?
          default:
            return xla::FailedPrecondition(
                "Could not apply sharding information to the %s computation.",
                comp->name().c_str());
        }
        attempt++;
      }
    }

    // Patch up GTE sharding.  GTEs should always have the sharding taken from
    // their operand, not their users.  During the initial copying of sharding
    // info, they are allowed to take the sharding of their users in order to
    // propagate sharding upwards through the graph.
    for (auto* inst : comp->MakeInstructionPostOrder()) {
      if (inst->opcode() == HloOpcode::kGetTupleElement) {
        CopyGteShardingFromOperand(inst);
      }
    }

    // Apply sharding to callers of this computation.  Caller nodes reflect the
    // sharding of the called subcomputation.
    for (auto cs : call_graph_node.caller_callsites()) {
      SetSharding(cs.instruction(), comp->root_instruction()->sharding());
    }
  }

  return true;
}

ShardingPass::ShardingPass() {}

}  // namespace poplarplugin
}  // namespace xla
