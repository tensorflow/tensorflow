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

#include "tensorflow/compiler/plugin/poplar/driver/forward_allocation.h"
#include "tensorflow/compiler/plugin/poplar/driver/allocation_finder.h"
#include "tensorflow/compiler/plugin/poplar/driver/compiler_annotations.h"
#include "tensorflow/compiler/plugin/poplar/driver/matcher_predicates.h"
#include "tensorflow/compiler/plugin/poplar/driver/util.h"
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

template <typename T>
using Graph = absl::flat_hash_map<T, absl::flat_hash_set<T>>;

static void create_graph(HloInstruction* inst, Graph<HloInstruction*>& result) {
  for (const auto& operand : inst->operands()) {
    const bool traverse = !result.contains(operand);
    result[operand].insert(inst);
    if (traverse) {
      create_graph(operand, result);
    }
  }
}

static Graph<HloInstruction*> create_graph(const HloComputation* module) {
  Graph<HloInstruction*> result;
  create_graph(module->root_instruction(), result);
  return result;
}

static Graph<HloInstruction*> transpose(const Graph<HloInstruction*>& graph) {
  Graph<HloInstruction*> result;

  for (const auto& edge : graph) {
    for (const auto& v2 : edge.second) {
      result[v2].insert(edge.first);
    }
  }

  return result;
}

static absl::flat_hash_set<HloInstruction*> get_vertices(
    const Graph<HloInstruction*>& graph) {
  absl::flat_hash_set<HloInstruction*> result;

  for (auto pair : graph) {
    result.insert(pair.first);
    result.merge(pair.second);
  }

  return result;
}

template <typename Predicate>
static absl::flat_hash_set<HloInstruction*> find_consumers(
    const Graph<HloInstruction*>& graph, HloInstruction* inst, Predicate pred,
    bool inclusive, absl::flat_hash_set<HloInstruction*>& visited) {
  absl::flat_hash_set<HloInstruction*> consumers;

  const auto itr = graph.find(inst);
  if (itr != graph.end()) {
    for (const auto& neighbour : itr->second) {
      if (inclusive) {
        consumers.insert(neighbour);
      }
      const bool already_visited = visited.count(neighbour);
      if (pred(neighbour) && !already_visited) {
        consumers.insert(neighbour);
        visited.insert(neighbour);
        consumers.merge(
            find_consumers(graph, neighbour, pred, inclusive, visited));
      }
    }
  }

  return consumers;
}

template <typename Predicate>
static absl::flat_hash_set<HloInstruction*> find_consumers(
    const Graph<HloInstruction*>& graph, HloInstruction* inst, Predicate pred,
    bool inclusive = false) {
  // find_consumers is a depth first traversal - this is a wrapper for it where
  // we create a set of visited instructions to prevent getting stuck in cycles.
  absl::flat_hash_set<HloInstruction*> visited;
  return find_consumers(graph, inst, pred, inclusive, visited);
}

template <typename Predicate>
static absl::flat_hash_set<HloInstruction*> find_vertices(
    const Graph<HloInstruction*>& graph, Predicate pred) {
  absl::flat_hash_set<HloInstruction*> result;

  for (const auto& v : get_vertices(graph)) {
    if (pred(v)) {
      result.insert(v);
    }
  }

  return result;
}

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

// The difference between reduce_to_one and reduce_to_one_with_no_dependencies
// is that the latter also tries to remove any results of reduction that are
// dependent on other reduced elements.
template <typename Predicate>
static absl::optional<HloInstruction*> reduce_to_one_with_no_dependencies(
    const absl::flat_hash_set<HloInstruction*>& values,
    const HloReachabilityMap* reachability_map, Predicate pred) {
  auto result = reduce(values, pred);
  absl::flat_hash_set<HloInstruction*> has_dependency;
  // Check whether this_inst depends on any other instruction from reduction.
  for (auto this_inst : result) {
    if (!is_independent(this_inst, result, reachability_map)) {
      has_dependency.insert(this_inst);
    }
  }
  // Get the result instructions which have no dependencies.
  // TODO consider whether it is worth extending this so that if we have few
  // targets with no dependency between we still allocate it with some layout
  // and add a control dependency.
  absl::flat_hash_set<HloInstruction*> result_no_deps;
  absl::c_set_difference(
      result, has_dependency,
      std::inserter(result_no_deps, std::begin(result_no_deps)));
  return result_no_deps.size() == 1
             ? absl::optional<HloInstruction*>(*std::begin(result_no_deps))
             : absl::nullopt;
}

static std::vector<const HloInstruction*> shortest_path(
    const Graph<HloInstruction*>& graph, HloInstruction* src,
    HloInstruction* dst) {
  absl::flat_hash_map<HloInstruction*, int> dist;
  absl::flat_hash_map<HloInstruction*, HloInstruction*> prev;
  absl::flat_hash_set<HloInstruction*> visited;

  const auto comp = [&](HloInstruction* a, HloInstruction* b) {
    return dist[a] < dist[b];
  };

  std::priority_queue<HloInstruction*, std::vector<HloInstruction*>,
                      decltype(comp)>
      queue(comp);

  const auto vs = get_vertices(graph);
  for (const auto& v : vs) {
    dist[v] = std::numeric_limits<int>::max();
  }

  dist[src] = 0;
  queue.push(src);

  while (!queue.empty() && dist[dst] == std::numeric_limits<int>::max()) {
    const auto top = queue.top();
    queue.pop();
    visited.insert(top);

    const auto itr = graph.find(top);
    std::for_each(itr->second.begin(), itr->second.end(),
                  [&](HloInstruction* v) {
                    if (!visited.contains(v)) {
                      dist[v] = dist[top] + 1;
                      prev[v] = top;
                      queue.push(v);
                    }
                  });
  }

  std::vector<HloInstruction*> path = {dst};
  while (path.back() != src) {
    path.push_back(prev[path.back()]);
  }
  std::reverse(path.begin(), path.end());

  return std::vector<const HloInstruction*>(path.begin(), path.end());
}

// TODO - fix this.  it needs to take into account the indices of the path
// from one op to the next. and probably do something to do with in-place ops
// Returns the tensor index of the last instruction in the path.
static absl::optional<int64> IsPathOk(
    const std::vector<const HloInstruction*>& path,
    bool allow_gte_at_the_end = false) {
  int64 tensor_index = 0;
  for (unsigned i = 0; i < path.size(); i++) {
    auto* inst = path[i];
    // Element-wise ops are ok.
    if (!IsPopOpsElementwise(inst)) {
      switch (inst->opcode()) {
        case HloOpcode::kGetTupleElement:
          // We only allow GTEs at the end of the path
          if (!(allow_gte_at_the_end && i == (path.size() - 1))) {
            return absl::nullopt;
          }
          tensor_index = inst->tuple_index();
        case HloOpcode::kReshape:
        case HloOpcode::kTranspose:
          break;
        default:
          return absl::nullopt;
          break;
      }
    }
  }
  return tensor_index;
};

static absl::optional<int64> IsPrefixPathOk(
    const std::vector<const HloInstruction*>& path) {
  return IsPathOk(path, false);
}

// We allow the suffix path to have a GTE at the end of the path.
static absl::optional<int64> IsSuffixPathOk(
    const std::vector<const HloInstruction*>& path) {
  return IsPathOk(path, true);
}

// TODO - this should probably be in a more central location
static bool IsLayoutSensitiveTarget(const HloInstruction* target) {
  switch (target->opcode()) {
    case HloOpcode::kBatchNormInference:
    case HloOpcode::kBatchNormTraining:
      return true;
    default:
      break;
  }
  return IsPopOpsElementwiseBinary(target);
}

// TODO - this should probably be in a more central location
static absl::optional<int64> IsLayoutSensitiveOperand(
    const HloInstruction* target, const HloInstruction* operand) {
  const auto op_idx = target->operand_index(operand);
  if (IsPopOpsElementwiseBinary(target)) {
    return op_idx;
  }
  switch (target->opcode()) {
    case HloOpcode::kBatchNormInference:
    case HloOpcode::kBatchNormTraining:
      // Only a layout sensitive target on operands index 1 and 2.
      if (op_idx == 1 || op_idx == 2) {
        return op_idx;
      }
      return absl::nullopt;
    default:
      break;
  }
  return absl::nullopt;
}

StatusOr<bool> ForwardAllocation::Run(
    HloComputation* comp, std::set<const HloInstruction*>& ops_with_layout) {
  bool found_targets = false;
  const auto is_param_no_layout_pred = [this](HloInstruction* inst) {
    return inst->opcode() == HloOpcode::kParameter &&
           tensor_allocation_map.find(std::make_pair(inst, 0)) ==
               tensor_allocation_map.end();
  };

  const auto is_layout_producer =
      [&ops_with_layout](const HloInstruction* inst) {
        return ops_with_layout.count(inst);
      };

  const auto g = create_graph(comp);
  const auto g_tr = transpose(g);
  const auto layout_producing_ops = find_vertices(g, is_layout_producer);

  std::unique_ptr<HloReachabilityMap> reachability_map =
      comp->ComputeReachability();

  // Get everything that depends upon an op with a special layout
  Graph<HloInstruction*> layout_op_consumers;
  for (const auto& inst : layout_producing_ops) {
    layout_op_consumers[inst] =
        find_consumers(g, inst, [is_layout_producer](HloInstruction* inst) {
          return !is_layout_producer(inst);
        });
  }

  const auto alloc_dependencies = transpose(layout_op_consumers);
  const auto source_ops = find_vertices(g, is_param_no_layout_pred);

  // Get everything that depends on a source op
  Graph<HloInstruction*> source_consumers;
  for (const auto& inst : source_ops) {
    source_consumers[inst] =
        find_consumers(g, inst,
                       [is_layout_producer, layout_producing_ops,
                        alloc_dependencies](HloInstruction* inst) {
                         return !is_layout_producer(inst) &&
                                !alloc_dependencies.contains(inst) &&
                                !layout_producing_ops.contains(inst);
                       },
                       true);
  }

  for (const auto& edges : source_consumers) {
    const auto& source = edges.first;

    if (!edges.second.empty()) {
      // Target is the op consuming the allocated tensor which is layout
      // sensitive.
      const auto is_valid_target = [&](HloInstruction* a) {
        return alloc_dependencies.contains(a) && IsLayoutSensitiveTarget(a);
      };

      const auto optional_target = reduce_to_one_with_no_dependencies(
          edges.second, reachability_map.get(), is_valid_target);
      if (!optional_target) {
        continue;
      }
      auto* target = *optional_target;
      const auto& itr = alloc_dependencies.find(target);
      if (itr != alloc_dependencies.end() && !itr->second.empty()) {
        // layout_producer is the op which produces the tensor whose layout is
        // important - it cannot have any allocation dependencies.
        // TODO we only allow a single layout producer at the moment.
        const auto is_not_alloc_dependency = [&](HloInstruction* a) {
          return !alloc_dependencies.contains(a);
        };
        const auto optional_layout_producer =
            reduce_to_one(itr->second, is_not_alloc_dependency);
        if (!optional_layout_producer) {
          continue;
        }
        auto* layout_producer = *optional_layout_producer;

        auto prefix = shortest_path(g, source, target);
        auto suffix = shortest_path(g, layout_producer, target);
        // Only some operands are layout sensitive.
        auto optional_op_idx =
            IsLayoutSensitiveOperand(target, prefix.rbegin()[1]);
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
              tensor_allocation_map[src] =
                  TensorTarget(target, op_idx, layout_producer,
                               layout_output_idx, suffix, prefix);
              // Make sure the layout_producer is executed before the source
              // instruction.
              layout_producer->AddControlDependencyTo(source);
              comp->UpdateReachabilityThroughInstruction(
                  source, reachability_map.get());
              found_targets = true;
            }
          }
        }
      }
    }
  }
  return found_targets;
}

ForwardAllocation::ForwardAllocation(CompilerAnnotations& annotations)
    : annotations(annotations),
      tensor_allocation_map(annotations.tensor_allocation_map) {}

StatusOr<bool> ForwardAllocation::Run(HloModule* module) {
  bool found_targets = false;

  // An op with a layout is an op that has been identified by the Allocation
  // Finder to have a layout, a Tensor allocation target or any op that is in
  // the path between the two.
  std::set<const HloInstruction*> ops_with_layout;
  for (auto& ta : tensor_allocation_map) {
    ops_with_layout.insert(ta.first.first);
    ops_with_layout.insert(ta.second.tgt);
    for (auto& inst : ta.second.forward_path) {
      ops_with_layout.insert(inst);
    }
  }

  for (const auto& computation : module->computations()) {
    if (!IsPopOpsCall(computation)) {
      TF_ASSIGN_OR_RETURN(bool found_targets_in_computation,
                          Run(computation, ops_with_layout));
      found_targets |= found_targets_in_computation;
    }
  }

  return found_targets;
}

}  // namespace poplarplugin
}  // namespace xla
