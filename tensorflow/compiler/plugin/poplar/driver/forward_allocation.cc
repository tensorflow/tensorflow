#include "forward_allocation.h"

#include <limits>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"

#include "tensorflow/compiler/plugin/poplar/driver/allocation_finder.h"
#include "tensorflow/compiler/plugin/poplar/driver/classification_predicates.h"
#include "tensorflow/compiler/plugin/poplar/driver/compiler_annotations.h"
#include "tensorflow/compiler/plugin/poplar/driver/util.h"
#include "tensorflow/compiler/xla/service/hlo_casting_utils.h"
#include "tensorflow/compiler/xla/service/hlo_instructions.h"

namespace xla {
namespace poplarplugin {

template <typename T>
using Graph = absl::flat_hash_map<T, absl::flat_hash_set<T>>;
using HloInstPtr = const HloInstruction*;

static Graph<HloInstPtr> create_graph(HloInstPtr inst) {
  Graph<HloInstPtr> result;

  for (const auto& operand : inst->operands()) {
    result[operand].insert(inst);

    for (auto& pair : create_graph(operand)) {
      result[pair.first].merge(pair.second);
    }
  }

  return result;
}

static Graph<HloInstPtr> create_graph(const HloComputation* module) {
  return create_graph(module->root_instruction());
}

static Graph<HloInstPtr> create_graph(const HloModule* module) {
  Graph<HloInstPtr> result;

  for (const auto& computation : module->computations()) {
    for (auto& pair : create_graph(computation)) {
      result[pair.first].merge(pair.second);
    }
  }

  return result;
}

static Graph<HloInstPtr> transpose(const Graph<HloInstPtr>& graph) {
  Graph<HloInstPtr> result;

  for (const auto& edge : graph) {
    for (const auto& v2 : edge.second) {
      result[v2].insert(edge.first);
    }
  }

  return result;
}

static absl::flat_hash_set<HloInstPtr> get_vertices(
    const Graph<HloInstPtr>& graph) {
  absl::flat_hash_set<HloInstPtr> result;

  for (auto pair : graph) {
    result.insert(pair.first);
    result.merge(pair.second);
  }

  return result;
}

template <typename Predicate>
static absl::flat_hash_set<HloInstPtr> find_consumers(
    const Graph<HloInstPtr>& graph, HloInstPtr inst, Predicate pred,
    bool inclusive = false) {
  absl::flat_hash_set<HloInstPtr> consumers;

  const auto itr = graph.find(inst);

  if (itr != graph.end()) {
    for (const auto& neighbour : itr->second) {
      if (inclusive) {
        consumers.insert(neighbour);
      }

      if (pred(neighbour)) {
        consumers.insert(neighbour);
        consumers.merge(find_consumers(graph, neighbour, pred, inclusive));
      }
    }
  }

  return consumers;
}

template <typename Predicate>
static absl::flat_hash_set<HloInstPtr> find_vertices(
    const Graph<HloInstPtr>& graph, Predicate pred) {
  absl::flat_hash_set<HloInstPtr> result;

  for (const auto& v : get_vertices(graph)) {
    if (pred(v)) {
      result.insert(v);
    }
  }

  return result;
}

static std::vector<HloInstPtr> shortest_path(const Graph<HloInstPtr>& graph,
                                             HloInstPtr src, HloInstPtr dst) {
  absl::flat_hash_map<HloInstPtr, int> dist;
  absl::flat_hash_map<HloInstPtr, HloInstPtr> prev;
  absl::flat_hash_set<HloInstPtr> visited;

  const auto comp = [&](HloInstPtr a, HloInstPtr b) {
    return dist[a] < dist[b];
  };

  std::priority_queue<HloInstPtr, std::vector<HloInstPtr>, decltype(comp)>
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
    std::for_each(itr->second.begin(), itr->second.end(), [&](HloInstPtr v) {
      if (!visited.contains(v)) {
        dist[v] = dist[top] + 1;
        prev[v] = top;
        queue.push(v);
      }
    });
  }

  std::vector<HloInstPtr> path = {dst};
  while (path.back() != src) {
    path.push_back(prev[path.back()]);
  }
  std::reverse(path.begin(), path.end());

  return path;
}

ForwardAllocation::ForwardAllocation(CompilerAnnotations& annotations)
    : annotations(annotations),
      tensor_allocation_map(annotations.tensor_allocation_map[0]),
      tensor_allocation_map_second_pass(annotations.tensor_allocation_map[1]) {}

StatusOr<bool> ForwardAllocation::Run(HloModule* module) {
  const auto is_alloc_pred = [this](HloInstPtr inst) {
    return inst->opcode() == HloOpcode::kConvolution ||
           inst->opcode() == HloOpcode::kDot ||
           inst->opcode() == HloOpcode::kDynamicSlice ||
           inst->opcode() == HloOpcode::kDynamicUpdateSlice ||
           inst->opcode() == HloOpcode::kCall ||
           tensor_allocation_map.find(std::make_pair(inst, 0)) !=
               tensor_allocation_map.end();
  };

  const auto is_param_no_layout_pred = [this](HloInstPtr inst) {
    return inst->opcode() == HloOpcode::kParameter &&
           tensor_allocation_map.find(std::make_pair(inst, 0)) ==
               tensor_allocation_map.end();
  };

  const auto g = create_graph(module);
  const auto g_tr = transpose(g);
  const auto alloc_verts = find_vertices(g, is_alloc_pred);

  Graph<HloInstPtr> alloc_consumers;
  for (const auto& v : alloc_verts) {
    alloc_consumers[v] = find_consumers(
        g, v, [is_alloc_pred](HloInstPtr v) { return !is_alloc_pred(v); });

    alloc_consumers[v].insert(v);
  }

  const auto alloc_dependencies = transpose(alloc_consumers);
  const auto param_verts = find_vertices(g, is_param_no_layout_pred);

  Graph<HloInstPtr> param_consumers;
  for (const auto& v : param_verts) {
    param_consumers[v] = find_consumers(
        g, v,
        [is_alloc_pred, alloc_verts, alloc_dependencies](HloInstPtr v) {
          return !is_alloc_pred(v) && !alloc_dependencies.contains(v) &&
                 !alloc_verts.contains(v);
        },
        true);
  }

  for (const auto& edges : param_consumers) {
    const auto& v1 = edges.first;
    const auto inst_reduction = [&](HloInstPtr a, HloInstPtr b) {
      if (alloc_dependencies.contains(a)) {
        return a;
      } else {
        return b;
      }
    };

    if (!edges.second.empty()) {
      const auto mid =
          std::accumulate(std::next(edges.second.begin()), edges.second.end(),
                          *(edges.second.begin()), inst_reduction);

      const auto itr = alloc_dependencies.find(mid);
      if (itr != alloc_dependencies.end() && !itr->second.empty()) {
        const auto target =
            std::accumulate(std::next(itr->second.begin()), itr->second.end(),
                            *(itr->second.begin()), inst_reduction);

        if (target->opcode() == HloOpcode::kConvolution ||
            target->opcode() == HloOpcode::kDot) {
          auto prefix = shortest_path(g, v1, mid);
          auto suffix = shortest_path(g, target, mid);

          auto src = std::make_pair(prefix.front(), 0);
          auto t = TensorTarget(suffix.front(), -1, suffix, prefix);
          tensor_allocation_map_second_pass[src] = t;
        }
      }
    }
  }

  return true;
}

}  // namespace poplarplugin
}  // namespace xla
