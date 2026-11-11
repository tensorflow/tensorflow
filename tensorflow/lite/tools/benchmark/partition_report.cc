/* Copyright 2026 The TensorFlow Authors. All Rights Reserved.

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
#include "tensorflow/lite/tools/benchmark/partition_report.h"

#include <cstddef>
#include <cstdint>
#include <map>
#include <string>
#include <utility>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"
#include "absl/strings/str_join.h"
#include "tensorflow/lite/context_util.h"
#include "tensorflow/lite/core/c/common.h"
#include "tensorflow/lite/core/interpreter.h"
#include "tensorflow/lite/core/subgraph.h"
#include "tensorflow/lite/util.h"

namespace tflite {
namespace benchmark {
namespace {

// A delegate kernel in the execution plan, i.e. one delegate partition.
struct Partition {
  int plan_index;
  std::string delegate_name;
  int num_replaced_nodes;
  int64_t in_bytes = 0;
  int64_t out_bytes = 0;
  // Outputs handed straight to another delegate partition. Not a CPU boundary
  // but still a kernel-to-kernel hop through an interpreter-owned tensor.
  int64_t to_delegate_bytes = 0;
  int num_dynamic_tensors = 0;
};

// A maximal run of consecutive non-delegated nodes in the execution plan.
struct FallbackIsland {
  int first_plan_index;
  int last_plan_index;
  std::map<std::string, int> op_histogram;
};

bool IsDelegateKernel(const TfLiteNode& node) {
  return node.delegate != nullptr;
}

std::string DelegateName(const TfLiteRegistration& registration) {
  return registration.custom_name != nullptr ? registration.custom_name
                                             : "delegate";
}

// Number of original graph nodes replaced by a delegate kernel. The subgraph
// stores either TfLiteDelegateParams or TfLiteOpaqueDelegateParams in
// `builtin_data`, depending on the delegate kind.
int NumReplacedNodes(const TfLiteNode& node) {
  if (node.builtin_data == nullptr) return 1;
  const TfLiteIntArray* nodes_to_replace =
      node.delegate->opaque_delegate_builder != nullptr
          ? static_cast<const TfLiteOpaqueDelegateParams*>(node.builtin_data)
                ->nodes_to_replace
          : static_cast<const TfLiteDelegateParams*>(node.builtin_data)
                ->nodes_to_replace;
  return nodes_to_replace != nullptr ? nodes_to_replace->size : 1;
}

// Adds `tensor_index` to the boundary traffic of `partition`.
void AccumulateBoundaryBytes(const Subgraph& subgraph, int tensor_index,
                             int64_t* bytes, Partition* partition) {
  const TfLiteTensor* tensor = subgraph.tensor(tensor_index);
  if (tensor == nullptr) return;
  if (tensor->allocation_type == kTfLiteDynamic) {
    ++partition->num_dynamic_tensors;
  } else {
    *bytes += static_cast<int64_t>(tensor->bytes);
  }
}

std::string FormatHistogram(const std::map<std::string, int>& histogram) {
  std::vector<std::string> parts;
  parts.reserve(histogram.size());
  for (const auto& [op_name, count] : histogram) {
    parts.push_back(absl::StrCat(op_name, " x", count));
  }
  return absl::StrJoin(parts, ", ");
}

}  // namespace

std::string GeneratePartitionReport(const Interpreter& interpreter) {
  const Subgraph& subgraph = interpreter.primary_subgraph();
  const std::vector<int>& plan = subgraph.execution_plan();
  const int num_plan_entries = static_cast<int>(plan.size());

  std::string report = "==== Delegate partition report (primary subgraph";
  if (interpreter.subgraphs_size() > 1) {
    absl::StrAppend(&report, "; ", interpreter.subgraphs_size() - 1,
                    " other subgraph(s) not shown");
  }
  absl::StrAppend(&report, ") ====\n");

  // Pass 1: classify each plan entry and record who produces / consumes which
  // tensor, so that boundaries can be resolved in plan order below.
  std::vector<bool> is_delegated(num_plan_entries, false);
  absl::flat_hash_map<int, int> producer_plan_index;  // tensor -> plan index
  absl::flat_hash_set<int> consumed_by_fallback;      // tensor indices
  absl::flat_hash_set<int> consumed_by_delegate;      // tensor indices
  for (int i = 0; i < num_plan_entries; ++i) {
    const TfLiteNode& node = subgraph.node_and_registration(plan[i])->first;
    is_delegated[i] = IsDelegateKernel(node);
    for (int t : TfLiteIntArrayView(node.outputs)) {
      if (t >= 0) producer_plan_index[t] = i;
    }
    absl::flat_hash_set<int>& consumers =
        is_delegated[i] ? consumed_by_delegate : consumed_by_fallback;
    for (int t : TfLiteIntArrayView(node.inputs)) {
      if (t >= 0) consumers.insert(t);
    }
  }
  const absl::flat_hash_set<int> graph_inputs(subgraph.inputs().begin(),
                                              subgraph.inputs().end());
  const absl::flat_hash_set<int> graph_outputs(subgraph.outputs().begin(),
                                               subgraph.outputs().end());

  // Pass 2: per-node lines, partitions, islands and boundary traffic.
  std::vector<Partition> partitions;
  std::vector<FallbackIsland> islands;
  int delegated_nodes = 0;
  int fallback_nodes = 0;
  absl::StrAppend(&report, "Execution plan (", num_plan_entries,
                  " entries):\n");
  for (int i = 0; i < num_plan_entries; ++i) {
    const auto& [node, registration] = *subgraph.node_and_registration(plan[i]);
    if (is_delegated[i]) {
      Partition partition;
      partition.plan_index = i;
      partition.delegate_name = DelegateName(registration);
      partition.num_replaced_nodes = NumReplacedNodes(node);
      delegated_nodes += partition.num_replaced_nodes;

      // Inputs produced by a fallback node or fed from outside the graph.
      absl::flat_hash_set<int> seen;
      for (int t : TfLiteIntArrayView(node.inputs)) {
        if (t < 0 || !seen.insert(t).second) continue;
        const auto producer = producer_plan_index.find(t);
        const bool from_fallback = producer != producer_plan_index.end() &&
                                   !is_delegated[producer->second];
        if (from_fallback || graph_inputs.count(t) > 0) {
          AccumulateBoundaryBytes(subgraph, t, &partition.in_bytes, &partition);
        }
      }
      // Outputs consumed by a fallback node or returned from the graph; the
      // rest either feed another delegate partition or are unused.
      seen.clear();
      for (int t : TfLiteIntArrayView(node.outputs)) {
        if (t < 0 || !seen.insert(t).second) continue;
        if (consumed_by_fallback.count(t) > 0 || graph_outputs.count(t) > 0) {
          AccumulateBoundaryBytes(subgraph, t, &partition.out_bytes,
                                  &partition);
        } else if (consumed_by_delegate.count(t) > 0) {
          AccumulateBoundaryBytes(subgraph, t, &partition.to_delegate_bytes,
                                  &partition);
        }
      }

      absl::StrAppendFormat(
          &report, "  [%4d] node %4d DELEGATE(%s) replaces %d node(s)\n", i,
          plan[i], partition.delegate_name, partition.num_replaced_nodes);
      partitions.push_back(std::move(partition));
    } else {
      ++fallback_nodes;
      const std::string op_name = GetOpNameByRegistration(registration);
      if (islands.empty() || islands.back().last_plan_index != i - 1) {
        islands.push_back({i, i, {}});
      }
      islands.back().last_plan_index = i;
      ++islands.back().op_histogram[op_name];
      absl::StrAppendFormat(&report, "  [%4d] node %4d %s\n", i, plan[i],
                            op_name);
    }
  }

  absl::StrAppend(&report, "Partitions: ", partitions.size(),
                  " delegate partition(s) covering ", delegated_nodes,
                  " node(s); ", fallback_nodes, " fallback node(s) in ",
                  islands.size(), " island(s)\n");
  for (size_t k = 0; k < islands.size(); ++k) {
    const FallbackIsland& island = islands[k];
    absl::StrAppendFormat(&report, "  island %d: plan[%d..%d] %d node(s): %s\n",
                          k, island.first_plan_index, island.last_plan_index,
                          island.last_plan_index - island.first_plan_index + 1,
                          FormatHistogram(island.op_histogram));
  }

  int64_t total_in_bytes = 0;
  int64_t total_out_bytes = 0;
  int64_t total_to_delegate_bytes = 0;
  int total_dynamic_tensors = 0;
  absl::StrAppend(&report, "Boundary traffic (CPU <-> delegate):\n");
  for (size_t k = 0; k < partitions.size(); ++k) {
    const Partition& p = partitions[k];
    total_in_bytes += p.in_bytes;
    total_out_bytes += p.out_bytes;
    total_to_delegate_bytes += p.to_delegate_bytes;
    total_dynamic_tensors += p.num_dynamic_tensors;
    absl::StrAppendFormat(&report,
                          "  partition %d: plan[%d] %s (%d node(s)) in=%d B "
                          "out=%d B",
                          k, p.plan_index, p.delegate_name,
                          p.num_replaced_nodes, p.in_bytes, p.out_bytes);
    if (p.to_delegate_bytes > 0) {
      absl::StrAppend(&report, " to_delegate=", p.to_delegate_bytes, " B");
    }
    if (p.num_dynamic_tensors > 0) {
      absl::StrAppend(&report, " (+", p.num_dynamic_tensors,
                      " dynamic tensor(s))");
    }
    absl::StrAppend(&report, "\n");
  }
  const int64_t boundary_bytes = total_in_bytes + total_out_bytes;
  absl::StrAppendFormat(&report,
                        "  total: in=%d B out=%d B all=%d B "
                        "(delegate->delegate: %d B)",
                        total_in_bytes, total_out_bytes, boundary_bytes,
                        total_to_delegate_bytes);
  if (total_dynamic_tensors > 0) {
    absl::StrAppend(&report, " (+", total_dynamic_tensors,
                    " dynamic tensor(s) of unknown size)");
  }
  absl::StrAppend(&report, "\n");

  absl::StrAppendFormat(&report,
                        "PARTITION_REPORT partitions=%d delegated_nodes=%d "
                        "fallback_nodes=%d boundary_bytes=%d "
                        "inter_partition_bytes=%d dynamic_boundary_tensors=%d",
                        partitions.size(), delegated_nodes, fallback_nodes,
                        boundary_bytes, total_to_delegate_bytes,
                        total_dynamic_tensors);
  return report;
}

}  // namespace benchmark
}  // namespace tflite
