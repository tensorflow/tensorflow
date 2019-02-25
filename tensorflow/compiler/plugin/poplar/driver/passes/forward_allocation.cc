/* Copyright 2018 Graphcore Ltd
 */

/* Copyright 2016 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/compiler/plugin/poplar/driver/passes/forward_allocation.h"
#include "tensorflow/compiler/plugin/poplar/driver/compiler_annotations.h"
#include "tensorflow/compiler/plugin/poplar/driver/passes/allocation_finder.h"
#include "tensorflow/compiler/plugin/poplar/driver/tools/matcher_predicates.h"
#include "tensorflow/compiler/plugin/poplar/driver/tools/meta_graph.h"
#include "tensorflow/compiler/plugin/poplar/driver/tools/util.h"
#include "tensorflow/compiler/plugin/poplar/kernels/custom_kernels_util.h"
#include "tensorflow/compiler/xla/service/hlo_casting_utils.h"
#include "tensorflow/compiler/xla/service/hlo_instructions.h"

#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "absl/types/optional.h"

#include <limits>
#include <vector>

namespace xla {
namespace poplarplugin {

template <typename Predicate>
static absl::flat_hash_set<HloInstruction*> reduce(
    const absl::flat_hash_set<HloInstruction*>& values, Predicate pred) {
  // For some reason absl const iterator doesn't define begin and end - we use a
  // copy instead.
  absl::flat_hash_set<HloInstruction*> result;
  absl::c_copy_if(values, std::inserter(result, std::begin(result)), pred);
  return result;
}

template <typename Predicate>
static absl::optional<HloInstruction*> reduce_to_one(
    const absl::flat_hash_set<HloInstruction*>& values, Predicate pred) {
  auto result = reduce(values, pred);
  return result.size() == 1
             ? absl::optional<HloInstruction*>(*std::begin(result))
             : absl::nullopt;
}

template <typename T>
static bool is_independent(const HloInstruction* inst,
                           const T& possible_dependencies,
                           const HloReachabilityMap* reachability_map) {
  for (auto dep : possible_dependencies) {
    if (dep != inst && reachability_map->IsReachable(dep, inst)) {
      return false;
    }
  }
  return true;
}

// Returns a vector of independent instructions which we want to use as a
// target. Note that the order of the targets is in decreasing priority order,
// where we want to target bias adds first, then layer norms and then
// elementwise ops.
template <typename Predicate>
static absl::optional<std::vector<HloInstruction*>> find_all_targets(
    const absl::flat_hash_set<HloInstruction*>& values,
    const HloReachabilityMap* reachability_map, Predicate pred) {
  auto insts = reduce(values, pred);
  absl::flat_hash_set<HloInstruction*> has_dependency;
  // Check whether this_inst depends on any other instruction from reduction.
  for (auto this_inst : insts) {
    if (!is_independent(this_inst, insts, reachability_map)) {
      has_dependency.insert(this_inst);
    }
  }
  // Get the insts instructions which have no dependencies.
  for (auto dep : has_dependency) {
    insts.erase(dep);
  }
  // There are no valid targets.
  if (insts.size() == 0) {
    return absl::nullopt;
  }

  auto biases =
      reduce(insts, [](HloInstruction* inst) { return IsPopOpsBiasAdd(inst); });
  auto norms = reduce(insts, [](HloInstruction* inst) {
    return IsNormInferenceOrTraining(inst);
  });

  // Add the instructions in order.
  std::vector<HloInstruction*> result;
  result.insert(std::end(result), std::begin(biases), std::end(biases));
  result.insert(std::end(result), std::begin(norms), std::end(norms));
  for (auto inst : insts) {
    if (biases.count(inst) == 0 && norms.count(inst) == 0) {
      result.push_back(inst);
    }
  }

  return result;
}

static bool output_and_all_operands_same_type(const HloInstruction* inst) {
  const PrimitiveType& type = inst->shape().element_type();
  for (auto* operand : inst->operands()) {
    if (type != operand->shape().element_type()) {
      return false;
    }
  }
  return true;
}

// TODO - fix this.  it needs to take into account the indices of the path
// from one op to the next. and probably do something to do with in-place ops
static bool IsPrefixPathOk(const std::vector<HloInstruction*>& path) {
  const auto is_node_ok_on_path = [](HloInstruction* inst, const unsigned,
                                     const unsigned) {
    // Element-wise ops are ok.
    if (IsPopOpsElementwise(inst)) {
      return output_and_all_operands_same_type(inst);
    }
    switch (inst->opcode()) {
      case HloOpcode::kReshape:
      case HloOpcode::kTranspose:
        return output_and_all_operands_same_type(inst);
      default:
        break;
    }
    return false;
  };
  return MetaGraph<HloInstruction*>::IsPathOk(path, is_node_ok_on_path);
}

// TODO - fix this.  it needs to take into account the indices of the path
// from one op to the next. and probably do something to do with in-place ops.
// We allow the suffix path to have a GTE at the end of the path.
// For valid paths, either returns the GTE index for the last node or 0.
static absl::optional<int64> IsSuffixPathOk(
    const std::vector<HloInstruction*>& path) {
  const auto is_node_ok_on_path = [](HloInstruction* inst,
                                     const unsigned path_idx,
                                     const unsigned path_size) {
    // Element-wise ops are ok.
    if (IsPopOpsElementwise(inst)) {
      return output_and_all_operands_same_type(inst);
    }
    switch (inst->opcode()) {
      case HloOpcode::kGetTupleElement:
        // We only allow GTEs at the end of the path
        return path_idx == (path_size - 1);
      case HloOpcode::kReshape:
      case HloOpcode::kTranspose:
        return output_and_all_operands_same_type(inst);
      default:
        break;
    }
    return false;
  };
  bool path_ok = MetaGraph<HloInstruction*>::IsPathOk(path, is_node_ok_on_path);
  if (!path_ok) {
    return absl::nullopt;
  }
  // Get the GTE index at the end of the path if there was one.
  return (path.size() >= 1 &&
          path.back()->opcode() == HloOpcode::kGetTupleElement)
             ? path.back()->tuple_index()
             : 0LL;
}

// TODO - this should probably be in a more central location
static bool IsLayoutSensitiveTarget(const HloInstruction* target) {
  switch (target->opcode()) {
    case HloOpcode::kBatchNormInference:
    case HloOpcode::kBatchNormTraining:
      return true;
    case HloOpcode::kCustomCall: {
      if (IsPoplibsCustomOp(target)) {
        auto attribute_map = IPUCustomKernelsUtil::AttributeMap(target);
        auto statusor =
            attribute_map.GetAttributeFlatHashMap("layout_dependencies");
        if (!statusor.ok()) {
          LOG(FATAL) << "Custom Poplibs op " << target->ToString()
                     << " is missing \'layout_dependencies\' field.";
        }

        absl::flat_hash_map<int64, int64> layout_dependencies =
            statusor.ValueOrDie();
        return layout_dependencies.size();
      }
      break;
    }
    default:
      break;
  }
  return IsPopOpsElementwiseBinary(target);
}

// TODO - this should probably be in a more central location
static absl::optional<int64> IsLayoutSensitiveOperand(
    const HloInstruction* target, const HloInstruction* operand,
    const HloInstruction* layout_producer) {
  const auto op_idx = target->operand_index(operand);
  // Some PopOps elementwise binary ops have more than two inputs (for example
  // scaled inplace with a scalar) - we make sure that we only target the first
  // two operands.
  if (IsPopOpsElementwiseBinary(target) && op_idx < 2) {
    return op_idx;
  }

  switch (target->opcode()) {
    case HloOpcode::kBatchNormInference:
    case HloOpcode::kBatchNormTraining:
      // Only a layout sensitive target on operands index 1 and 2 iff
      // layout_producer is operand 0.
      if (target->operand(0) == layout_producer &&
          (op_idx == 1 || op_idx == 2)) {
        return op_idx;
      }
      return absl::nullopt;
    case HloOpcode::kCustomCall: {
      if (IsPoplibsCustomOp(target)) {
        auto attribute_map = IPUCustomKernelsUtil::AttributeMap(target);
        auto statusor =
            attribute_map.GetAttributeFlatHashMap("layout_dependencies");
        if (!statusor.ok()) {
          LOG(FATAL) << "Custom Poplibs op " << target->ToString()
                     << " is missing \'layout_dependencies\' field.";
        }
        // A Poplibs Custom op is layout sensative if the layout_producer is at
        // the right index.
        absl::flat_hash_map<int64, int64> layout_dependencies =
            statusor.ValueOrDie();
        auto itr = layout_dependencies.find(op_idx);
        if (itr != layout_dependencies.end() &&
            target->operand(itr->second) == layout_producer) {
          return op_idx;
        }
        return absl::nullopt;
      }
      break;
    }
    default:
      break;
  }
  return absl::nullopt;
}

// Depth First Tree traversal from source to non-tuple outputs, traversing
// through GetTupleElement.
void ForwardAllocation::FlattenInputs(
    HloInstruction* inst, std::vector<const HloInstruction*> path,
    absl::flat_hash_map<HloInstruction*, DeferredAllocationsPath>&
        input_to_deferred_allocation_path) {
  const Shape& shape = inst->shape();
  if (shape.IsTuple()) {
    // We can only defer allocation of tuples iff all the users of the inst are
    // unique GTEs with compatible sharding.
    absl::flat_hash_set<int64> tuple_indexes;
    for (HloInstruction* user : inst->users()) {
      if (user->opcode() == HloOpcode::kGetTupleElement) {
        auto tuple_index = user->tuple_index();
        if (tuple_indexes.contains(tuple_index)) {
          // We can't defer allocation here - we require GTEs to be unique.
          return;
        } else if (!user->has_compatible_sharding(inst)) {
          // We can't defer allocation here - we require compatible sharding -
          // otherwise a copy would have to take place which requires the tensor
          // to be allocated.
          return;
        } else {
          tuple_indexes.insert(tuple_index);
        }
      } else {
        // We can't defer allocation here - we can only look through GTEs.
        return;
      }
    }
    for (HloInstruction* user : inst->users()) {
      // We have guaranteed that we are only looking through GTEs.
      CHECK_EQ(user->opcode(), HloOpcode::kGetTupleElement);
      // We can only look through if it's inplace.
      if (inplace_instructions.contains(user)) {
        std::vector<const HloInstruction*> new_path(path);
        new_path.push_back(user);
        FlattenInputs(user, new_path, input_to_deferred_allocation_path);
      }
    }
  } else {
    // The back of the path is the current op so remove it.
    path.pop_back();
    // We need to traverse back and complete the information about which
    // sub-tensors we are deferring the allocation of.
    DeferredAllocationsPath deferred_allocation_path;
    const HloInstruction* last_gte = inst;
    int64 flat_tuple_index = 0;
    for (auto it = path.rbegin(); it != path.rend(); ++it) {
      // We guarantee that all the deferred allocations depend on GTEs only.
      CHECK_EQ(last_gte->opcode(), HloOpcode::kGetTupleElement);
      const HloInstruction* inst = *it;
      flat_tuple_index = InsertIntoTuple(inst->shape(), last_gte->tuple_index(),
                                         flat_tuple_index);
      deferred_allocation_path.push_back(
          std::make_pair(inst, flat_tuple_index));
      last_gte = inst;
    }
    input_to_deferred_allocation_path[inst] = deferred_allocation_path;
  }
}

// Inputs to the graph are non-tuple tensors which originate from parameters or
// infeeds. To find such tensors we traverse through GetTupleElement
// instructions, keeping track of this path. For example, given following HLO
// computation:
// clang-format off
//%comp (arg0: (f32[1,4,4,2], f32[1,1,2,2], (f32[1,2], f32[1,2]), f32[2], f32[2])) -> f32[1,4,4,2] {
// %arg0 = (f32[1,4,4,2], f32[1,1,2,2], (f32[1,2], f32[1,2]), f32[2], f32[2]) parameter(0)
// %gte0 = f32[1,4,4,2] get-tuple-element((f32[1,4,4,2], f32[1,1,2,2], (f32[1,2], f32[1,2]), f32[2], f32[2]) %arg0), index=0
// %gte1 = f32[1,1,2,2] get-tuple-element((f32[1,4,4,2], f32[1,1,2,2], (f32[1,2], f32[1,2]), f32[2], f32[2]) %arg0), index=1
// %convolution.36.29 = f32[1,4,4,2] convolution(%gte0, %gte1), window={size=1x1}, dim_labels=b01f_01io->b01f
// %gte2 = (f32[1,2], f32[1,2]) get-tuple-element((f32[1,4,4,2], f32[1,1,2,2], (f32[1,2], f32[1,2]), f32[2], f32[2]) %arg0), index=2
// %gte2.0 = f32[1,2] get-tuple-element((f32[1,2], f32[1,2]) %gte2), index=0
// %gte2.0_r = f32[2] reshape(%gte2.0)
// %gte2.1 = f32[1,2] get-tuple-element((f32[1,2], f32[1,2]) %gte2), index=1
// %gte2.1_r = f32[2] reshape(%gte2.1)
// %gte3 = f32[2] get-tuple-element((f32[1,4,4,2], f32[1,1,2,2], (f32[1,2], f32[1,2]), f32[2], f32[2]) %arg0), index=3
// %gte4 = f32[2] get-tuple-element((f32[1,4,4,2], f32[1,1,2,2], (f32[1,2], f32[1,2]), f32[2], f32[2]) %arg0), index=4
// ROOT %batch-norm-inference.36.31 = f32[1,4,4,2] batch-norm-inference(%convolution.36.29, %gte2.0_r, %gte2.1_r, %gte3, %gte4), epsilon=0.001, feature_index=3
//}
// clang-format on
// In this graph %arg0 is the input, but we traverse the graph and find that
// %gte0, %gte1, %gte2.0, %gte2.1, %gte3, %gte4 are the non tuple inputs and we
// find the forward allocations for those.
absl::flat_hash_map<HloInstruction*, DeferredAllocationsPath>
ForwardAllocation::FindInputs(HloComputation* comp) {
  absl::flat_hash_map<HloInstruction*, DeferredAllocationsPath>
      input_to_deferred_allocation_path;
  for (HloInstruction* inst : comp->MakeInstructionPostOrder()) {
    if (inst->opcode() == HloOpcode::kParameter ||
        inst->opcode() == HloOpcode::kInfeed) {
      FlattenInputs(inst, {inst}, input_to_deferred_allocation_path);
    }
  }
  return input_to_deferred_allocation_path;
}

StatusOr<bool> ForwardAllocation::Run(
    HloComputation* comp, std::set<const HloInstruction*>& ops_with_layout) {
  bool found_target = false;

  auto input_to_deferred_allocations = FindInputs(comp);

  const auto is_input = [&input_to_deferred_allocations,
                         this](HloInstruction* inst) {
    auto itr = input_to_deferred_allocations.find(inst);
    if (itr != input_to_deferred_allocations.end()) {
      return tensor_allocation_map.find(std::make_pair(inst, 0)) ==
             tensor_allocation_map.end();
    }
    return false;
  };

  const auto is_layout_producer = [&ops_with_layout](HloInstruction* inst) {
    return ops_with_layout.count(inst);
  };

  const auto get_operands = [](HloInstruction* inst) {
    return inst->operands();
  };

  const auto g =
      MetaGraph<HloInstruction*>(comp->root_instruction(), get_operands);
  const auto layout_producing_ops = g.FindVertices(is_layout_producer);

  std::unique_ptr<HloReachabilityMap> reachability_map =
      HloReachabilityMap::Build(comp);

  // Get everything that depends upon an op with a special layout
  const auto get_consumers = [is_layout_producer, &g](HloInstruction* inst) {
    return g.FindConsumers(inst, [is_layout_producer](HloInstruction* inst) {
      return !is_layout_producer(inst);
    });
  };
  const MetaGraph<HloInstruction*> layout_op_consumers(layout_producing_ops,
                                                       get_consumers);

  const auto alloc_dependencies = layout_op_consumers.Transpose();
  const auto source_ops = g.FindVertices(is_input);

  // Get everything that depends on a source op
  const auto get_source_consumers = [is_layout_producer, layout_producing_ops,
                                     alloc_dependencies,
                                     g](HloInstruction* inst) {
    return g.FindConsumers(inst,
                           [is_layout_producer, layout_producing_ops,
                            alloc_dependencies](HloInstruction* inst) {
                             return !is_layout_producer(inst) &&
                                    !alloc_dependencies.contains(inst) &&
                                    !layout_producing_ops.contains(inst);
                           },
                           true);
  };
  const MetaGraph<HloInstruction*> source_consumers(source_ops,
                                                    get_source_consumers);

  for (const auto& edges : source_consumers) {
    const auto& source = edges.first;
    if (!edges.second.empty()) {
      // Target is the op consuming the allocated tensor which is layout
      // sensitive.
      const auto is_valid_target = [&](HloInstruction* a) {
        return alloc_dependencies.contains(a) && IsLayoutSensitiveTarget(a);
      };
      const auto optional_targets = find_all_targets(
          edges.second, reachability_map.get(), is_valid_target);
      if (!optional_targets) {
        continue;
      }
      std::vector<HloInstruction*> targets = *optional_targets;
      for (HloInstruction* target : targets) {
        // Find layout producers for the target.
        // layout_producer is the op which produces the tensor whose layout is
        // important - it cannot have any allocation dependencies.
        const auto& itr = alloc_dependencies.find(target);
        // Skip if the target has not allocation dependencies or if the target
        // has no layout producer.
        if (itr == alloc_dependencies.end() || itr->second.empty()) {
          continue;
        }
        const auto is_not_alloc_dependency = [&](HloInstruction* a) {
          return !alloc_dependencies.contains(a);
        };
        // TODO we only allow a single layout producer at the moment.
        const auto optional_layout_producer =
            reduce_to_one(itr->second, is_not_alloc_dependency);
        if (!optional_layout_producer) {
          continue;
        }
        auto* layout_producer = *optional_layout_producer;

        // Try and find the shortest paths from/to target.
        auto optional_prefix = g.ShortestPath(source, target);
        auto optional_suffix = g.ShortestPath(layout_producer, target);
        if (!(optional_prefix && optional_suffix)) {
          continue;
        }
        auto prefix = *optional_prefix;
        auto suffix = *optional_suffix;
        // Only some operands are layout sensitive.
        auto optional_op_idx = IsLayoutSensitiveOperand(
            target, prefix.rbegin()[1], layout_producer);
        if (optional_op_idx) {
          const auto op_idx = *optional_op_idx;
          // The paths don't contain the source or target instructions
          prefix.erase(prefix.begin());
          prefix.pop_back();
          suffix.erase(suffix.begin());
          suffix.pop_back();
          const auto prefix_path_ok = IsPrefixPathOk(prefix);
          const auto suffix_path_ok = IsSuffixPathOk(suffix);
          if (prefix_path_ok && suffix_path_ok) {
            if (!source_consumers[source].contains(layout_producer)) {
              auto layout_output_idx = *suffix_path_ok;
              auto src = std::make_pair(source, 0);
              auto deferred_allocations_path =
                  input_to_deferred_allocations[source];
              std::vector<const HloInstruction*> csuffix(suffix.begin(),
                                                         suffix.end());
              std::vector<const HloInstruction*> cprefix(prefix.begin(),
                                                         prefix.end());

              // Make sure that the layout producer can be executed before the
              // source - i.e. source is not reachable form the layout producer.
              if (reachability_map->IsReachable(source, layout_producer)) {
                continue;
              }
              layout_producer->AddControlDependencyTo(source);
              reachability_map->UpdateReachabilityThroughInstruction(source);

              // Make sure that the target can be executed before all the other
              // independent targets with the new control dependency.
              // Keep track of any dependencies we add in case we have to undo
              // them.
              std::vector<HloInstruction*> added_dependants;
              bool dependencies_ok = true;
              for (auto new_dependent : targets) {
                if (new_dependent == target) {
                  continue;
                }
                if (!reachability_map->IsReachable(target, new_dependent)) {
                  target->AddControlDependencyTo(new_dependent);
                  reachability_map->UpdateReachabilityThroughInstruction(
                      new_dependent);
                  added_dependants.push_back(target);
                } else {
                  dependencies_ok = false;
                  break;
                }
              }
              if (!dependencies_ok) {
                // Remove all the added dependencies
                layout_producer->RemoveControlDependencyTo(source);
                reachability_map->UpdateReachabilityThroughInstruction(source);
                for (auto inst : added_dependants) {
                  target->RemoveControlDependencyTo(inst);
                  reachability_map->UpdateReachabilityThroughInstruction(inst);
                }
                continue;
              }
              tensor_allocation_map[src] = TensorTarget(
                  target, op_idx, layout_producer, layout_output_idx, csuffix,
                  cprefix, deferred_allocations_path);
              deferred_allocations.insert(deferred_allocations_path.begin(),
                                          deferred_allocations_path.end());
              found_target = true;
              break;
            }
          }
        }
      }
    }
  }
  return found_target;
}

ForwardAllocation::ForwardAllocation(CompilerAnnotations& annotations)
    : tensor_allocation_map(annotations.tensor_allocation_map),
      deferred_allocations(annotations.deferred_allocations),
      inplace_instructions(annotations.inplace_instructions) {}

StatusOr<bool> ForwardAllocation::Run(HloModule* module) {
  bool found_target = false;

  // An op with a layout is a non-tuple op that has been identified by the
  // Allocation Finder to have a layout, a Tensor allocation target or any op
  // that is in the path between the two.
  std::set<const HloInstruction*> ops_with_layout;
  for (auto& ta : tensor_allocation_map) {
    auto source = ta.first.first;
    if (!source->shape().IsTuple()) {
      ops_with_layout.insert(source);
    }
    ops_with_layout.insert(ta.second.tgt);
    for (auto& inst : ta.second.forward_path) {
      if (!inst->shape().IsTuple()) {
        ops_with_layout.insert(inst);
      }
    }
  }

  for (const auto& computation : module->computations()) {
    if (IsPopOpsFusion(computation)) {
      continue;
    }
    TF_ASSIGN_OR_RETURN(bool found_target_in_computation,
                        Run(computation, ops_with_layout));
    found_target |= found_target_in_computation;
  }

  return found_target;
}

}  // namespace poplarplugin
}  // namespace xla
