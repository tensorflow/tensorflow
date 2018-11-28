#include "forward_allocation.h"

#include <limits>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "absl/types/optional.h"

#include "tensorflow/compiler/plugin/poplar/driver/allocation_finder.h"
#include "tensorflow/compiler/plugin/poplar/driver/classification_predicates.h"
#include "tensorflow/compiler/plugin/poplar/driver/compiler_annotations.h"
#include "tensorflow/compiler/plugin/poplar/driver/util.h"
#include "tensorflow/compiler/plugin/poplar/kernels/custom_kernels_util.h"
#include "tensorflow/compiler/xla/service/hlo_casting_utils.h"
#include "tensorflow/compiler/xla/service/hlo_instructions.h"

namespace xla {
namespace poplarplugin {

template <typename T>
using Graph = absl::flat_hash_map<T, absl::flat_hash_set<T>>;
using HloInstPtr = const HloInstruction*;

static void create_graph(HloInstPtr inst, Graph<HloInstPtr>& result) {
  for (const auto& operand : inst->operands()) {
    if (!result.contains(operand)) {
      result[operand].insert(inst);
      create_graph(operand, result);
    }
  }
}

static void create_graph(const HloComputation* module,
                         Graph<HloInstPtr>& result) {
  return create_graph(module->root_instruction(), result);
}

static Graph<HloInstPtr> create_graph(const HloModule* module) {
  Graph<HloInstPtr> result;

  for (const auto& computation : module->computations()) {
    create_graph(computation, result);
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

// TODO - this should probably be in a more central location
static bool IsLayoutProducer(HloInstPtr inst) {
  switch (inst->opcode()) {
    case HloOpcode::kConvolution:
    case HloOpcode::kDot:
      return true;
    default:
      break;
  }

  if (IsPopOpsCall(inst, "depthwise_conv")) {
    return true;
  }

  if (IPUCustomKernelsUtil::IsPoplibsOp(inst)) {
    // For custom ops, they are layout producers if they have allocating
    // operands.
    auto attribute_map = IPUCustomKernelsUtil::AttributeMap(inst);
    auto statusor =
        attribute_map.GetAttributeAsInt64FlatHashSet("allocating_indexes");
    if (!statusor.ok()) {
      LOG(FATAL) << "Custom Poplibs op " << inst->ToString()
                 << " is missing \'allocating_indexes\' field.";
    }
    return statusor.ValueOrDie().size() > 0;
  }
  return false;
}

// TODO - fix this.  it needs to take into account the indices of the path
// from one op to the next. and probably do something to do with in-place ops
static bool IsPathOk(const std::vector<HloInstPtr>& path) {
  for (auto* inst : path) {
    switch (inst->opcode()) {
      case HloOpcode::kBatchNormInference:
      case HloOpcode::kBatchNormTraining:
      case HloOpcode::kReshape:
      case HloOpcode::kTranspose:
        break;
      case HloOpcode::kCall:
        if (!IsPopOpsBiasAdd(inst)) {
          return false;
        }
        break;
      default:
        if (!inst->IsElementwise()) {
          return false;
        }
        break;
    }
  }
  return true;
};

// TODO - this should probably be in a more central location
static bool IsLayoutSensitiveTarget(HloInstPtr target) {
  switch (target->opcode()) {
    case HloOpcode::kBatchNormInference:
    case HloOpcode::kBatchNormTraining:
      return true;
    default:
      break;
  }
  return IsPopOpsBiasAdd(target);
}

// TODO - this should probably be in a more central location
static absl::optional<int64> IsLayoutSensitiveOperand(HloInstPtr target,
                                                      HloInstPtr operand) {
  const auto op_idx = target->operand_index(operand);
  if (IsPopOpsBiasAdd(target) && op_idx == 1) {
    // Only layout sensitive target on operand index 1
    return 1;
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

ForwardAllocation::ForwardAllocation(CompilerAnnotations& annotations)
    : annotations(annotations),
      tensor_allocation_map(annotations.tensor_allocation_map) {}

StatusOr<bool> ForwardAllocation::Run(HloModule* module) {
  const auto is_param_no_layout_pred = [this](HloInstPtr inst) {
    return inst->opcode() == HloOpcode::kParameter &&
           tensor_allocation_map.find(std::make_pair(inst, 0)) ==
               tensor_allocation_map.end();
  };

  const auto g = create_graph(module);
  const auto g_tr = transpose(g);
  const auto layout_producing_ops = find_vertices(g, IsLayoutProducer);

  // Get everything that depends upon an op with a special layout
  Graph<HloInstPtr> layout_op_consumers;
  for (const auto& inst : layout_producing_ops) {
    layout_op_consumers[inst] = find_consumers(
        g, inst, [](HloInstPtr inst) { return !IsLayoutProducer(inst); });
  }

  const auto alloc_dependencies = transpose(layout_op_consumers);
  const auto source_ops = find_vertices(g, is_param_no_layout_pred);

  // Get everything that depends on a source op
  Graph<HloInstPtr> source_consumers;
  for (const auto& inst : source_ops) {
    source_consumers[inst] = find_consumers(
        g, inst,
        [layout_producing_ops, alloc_dependencies](HloInstPtr inst) {
          return !IsLayoutProducer(inst) &&
                 !alloc_dependencies.contains(inst) &&
                 !layout_producing_ops.contains(inst);
        },
        true);
  }

  for (const auto& edges : source_consumers) {
    const auto& source = edges.first;
    const auto inst_reduction = [&](HloInstPtr a, HloInstPtr b) {
      if (alloc_dependencies.contains(a)) {
        return a;
      } else {
        return b;
      }
    };

    if (!edges.second.empty()) {
      // Target is the op consuming the allocated tensor
      const auto target =
          std::accumulate(std::next(edges.second.begin()), edges.second.end(),
                          *(edges.second.begin()), inst_reduction);

      if (IsLayoutSensitiveTarget(target)) {
        const auto& itr = alloc_dependencies.find(target);
        if (itr != alloc_dependencies.end() && !itr->second.empty()) {
          // layout_producer is the op which produces the tensor whose layout is
          // important
          const auto* layout_producer =
              std::accumulate(std::next(itr->second.begin()), itr->second.end(),
                              *(itr->second.begin()), inst_reduction);

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
            auto src = std::make_pair(source, 0);
            auto t =
                TensorTarget(target, op_idx, layout_producer, suffix, prefix);

            if (IsPathOk(prefix) && IsPathOk(suffix)) {
              if (!source_consumers[source].contains(layout_producer)) {
                tensor_allocation_map[src] = t;

                HloInstruction* s;
                TF_ASSIGN_OR_RETURN(
                    s, module->LaunderConstInstructionFromModule(source));
                HloInstruction* p;
                TF_ASSIGN_OR_RETURN(
                    p,
                    module->LaunderConstInstructionFromModule(layout_producer));

                p->AddControlDependencyTo(s);
              }
            }
          }
        }
      }
    }
  }

  return true;
}

}  // namespace poplarplugin
}  // namespace xla
