/* Copyright 2026 The OpenXLA Authors.

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

#include "xla/hlo/tools/comparison/comparison_hlo_dumper.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <limits>
#include <map>
#include <optional>
#include <queue>
#include <string>
#include <utility>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "json/json.h"
#include "riegeli/bytes/fd_reader.h"
#include "riegeli/records/record_reader.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/hlo/tools/comparison/comparison_result.pb.h"
#include "xla/hlo/tools/comparison/comparison_result_utils.h"
#include "xla/hlo/tools/comparison/original_tensor_summary_utils.h"
#include "xla/hlo/tools/comparison/tensor_summary_util.h"
#include "xla/service/hlo.pb.h"
#include "xla/shape.h"
#include "xla/shape_util.h"
#include "xla/tools/hlo_dump/hlo_dump_utils.h"
#include "xla/tsl/platform/env.h"
#include "xla/tsl/platform/threadpool.h"
#include "tsl/platform/path.h"

namespace xla::numerics::comparison {

absl::Status DumpHloToHtml(
    const std::string& dump_name, absl::string_view hlo_text,
    const absl::flat_hash_map<HloHtmlTensorKey, TensorAnnotation>& annotations,
    const std::string& output_path, std::optional<float> percentage_recoverable,
    std::optional<float> percentage_recovered,
    const xla::StackFrameIndexProto* stack_frame_index,
    const debug_info::GraphData* graph_data) {
  LOG(INFO) << "DumpHloToHtml started for " << dump_name;
  absl::flat_hash_map<debug_info::TensorKey, debug_info::TensorAnnotation>
      mapped_annotations;
  // NOLINTNEXTLINE
  for (const auto& [key, ann] : annotations) {
    debug_info::TensorKey tk;
    tk.instruction_name = key.name;
    for (int idx : key.shape_index) {
      tk.shape_index.push_back(idx);
    }
    debug_info::TensorAnnotation mapped_ann;
    mapped_ann.background_color = ann.background_color;
    mapped_ann.border_style = ann.border_style;
    mapped_ann.tooltip_data = ann.tooltip_data;
    mapped_ann.anchor_id = ann.anchor_id;
    mapped_ann.stack_frame_id = ann.stack_frame_id;
    mapped_annotations[tk] = std::move(mapped_ann);
  }

  LOG(INFO) << "Calling ConvertHloToHtml for " << dump_name;
  debug_info::OriginalValueRecoveryInfo recovery_info;
  if (percentage_recoverable) {
    recovery_info.percentage_recoverable =
        static_cast<double>(*percentage_recoverable);
  }
  if (percentage_recovered) {
    recovery_info.percentage_recovered =
        static_cast<double>(*percentage_recovered);
  }

  std::string html = debug_info::ConvertHloToHtml(
      dump_name, hlo_text, mapped_annotations, recovery_info, stack_frame_index,
      graph_data);

  LOG(INFO) << "Writing HTML to file: " << output_path;
  auto status = tsl::WriteStringToFile(tsl::Env::Default(), output_path, html);
  LOG(INFO) << "DumpHloToHtml finished for " << dump_name << " with status "
            << status;
  return status;
}

FloatBlockSummary CombineSummaries(
    const std::vector<TensorSummaryProto>& summaries) {
  std::vector<FloatBlockSummary> all_blocks;
  for (const auto& summary : summaries) {
    for (const auto& bs : summary.block_summaries()) {
      FloatBlockSummary fbs;
      fbs.min = bs.min();
      fbs.max = bs.max();
      fbs.mean = bs.mean();
      fbs.stddev = bs.stddev();
      fbs.count = bs.count();
      fbs.nan_count = bs.nan_count();
      fbs.pos_inf_count = bs.pos_inf_count();
      fbs.neg_inf_count = bs.neg_inf_count();
      fbs.zero_count = bs.zero_count();
      all_blocks.push_back(fbs);
    }
  }
  if (all_blocks.empty()) {
    return {};
  }
  return ::xla::comparison::CombineBlockSummaries({}, all_blocks);
}

std::string GetTooltipData(const HloNodeComparisonStats* comp,
                           const FloatBlockSummary* baseline,
                           const FloatBlockSummary* target) {
  Json::Value json(Json::objectValue);
  if (comp != nullptr) {
    json["diffScore"]["notComparable"] = comp->not_comparable;
    if (!comp->not_comparable) {
      json["diffScore"]["count"] = static_cast<Json::Int64>(comp->score_count);
      json["diffScore"]["min"] = comp->score_count > 0 ? comp->score_min : 0.0;
      json["diffScore"]["max"] = comp->score_count > 0 ? comp->score_max : 0.0;
      json["diffScore"]["mean"] =
          comp->score_count > 0 ? comp->score_mean / comp->score_count : 0.0;
    }
  }

  auto dump_json = [](const Json::Value& val) {
    Json::StreamWriterBuilder builder;
    builder["indentation"] = "";
    builder["precisionType"] = "significant";
    builder["precision"] = 7;
    return Json::writeString(builder, val);
  };

  if (comp != nullptr && comp->not_comparable) {
    return dump_json(json);
  }

  auto add_metric = [&](absl::string_view label, auto get_val) {
    std::string label_str(label);
    if (baseline != nullptr) {
      json["metrics"][label_str]["baseline"] =
          static_cast<double>(get_val(*baseline));
    }
    if (target != nullptr) {
      json["metrics"][label_str]["target"] =
          static_cast<double>(get_val(*target));
    }
  };

  add_metric("Mean", [](const FloatBlockSummary& s) { return s.mean; });
  add_metric("Stddev", [](const FloatBlockSummary& s) { return s.stddev; });
  add_metric("Min", [](const FloatBlockSummary& s) { return s.min; });
  add_metric("Max", [](const FloatBlockSummary& s) { return s.max; });
  add_metric("Count", [](const FloatBlockSummary& s) { return s.count; });
  add_metric("NaN", [](const FloatBlockSummary& s) { return s.nan_count; });
  add_metric("+Inf",
             [](const FloatBlockSummary& s) { return s.pos_inf_count; });
  add_metric("-Inf",
             [](const FloatBlockSummary& s) { return s.neg_inf_count; });
  add_metric("Zero", [](const FloatBlockSummary& s) { return s.zero_count; });

  return dump_json(json);
}

namespace {

// Unique identifier for an instruction instance in the layout.
struct InstructionInstance {
  std::string name;
  std::vector<ScopeInstruction> scope;

  bool operator==(const InstructionInstance& other) const {
    if (name != other.name) {
      return false;
    }
    if (scope.size() != other.scope.size()) {
      return false;
    }
    for (size_t i = 0; i < scope.size(); ++i) {
      if (scope[i].instruction_name != other.scope[i].instruction_name ||
          scope[i].iteration_index != other.scope[i].iteration_index) {
        return false;
      }
    }
    return true;
  }

  template <typename H>
  friend H AbslHashValue(H h, const InstructionInstance& n) {
    h = H::combine(std::move(h), n.name);
    for (const auto& s : n.scope) {
      h = H::combine(std::move(h), s.instruction_name, s.iteration_index);
    }
    return h;
  }
};

InstructionInstance GetInstInst(const AbsoluteScopedTensorKey& key) {
  return {key.tensor_key.instruction_name, key.scope_instructions};
}

struct CallInfo {
  std::string name;
  std::vector<ScopeInstruction> enclosing_scope;

  bool operator==(const CallInfo& other) const {
    return name == other.name && enclosing_scope == other.enclosing_scope;
  }
  template <typename H>
  friend H AbslHashValue(H h, const CallInfo& c) {
    for (const auto& s : c.enclosing_scope) {
      h = H::combine(std::move(h), s.instruction_name, s.iteration_index);
    }
    return H::combine(std::move(h), c.name);
  }
};

absl::flat_hash_map<AbsoluteScopedTensorKey, NodePosition> ComputeDagLayout(
    const ComputationDagCollection& dag_collection,
    absl::Span<const AbsoluteScopedTensorKey> reported_keys,
    const absl::flat_hash_map<AbsoluteScopedTensorKey,
                              std::vector<AbsoluteScopedTensorKey>>&
        node_consumers,
    const absl::flat_hash_map<AbsoluteScopedTensorKey,
                              std::vector<AbsoluteScopedTensorKey>>&
        node_suppliers) {
  auto local_node_consumers = node_consumers;
  auto local_node_suppliers = node_suppliers;

  absl::flat_hash_map<CallInfo, std::vector<AbsoluteScopedTensorKey>>
      call_to_keys;
  absl::flat_hash_map<std::vector<ScopeInstruction>, std::vector<CallInfo>>
      scope_to_calls;

  auto is_call_like = [&](absl::string_view inst_name) {
    if (auto it = dag_collection.inst_info.find(inst_name);
        it != dag_collection.inst_info.end()) {
      return it->second.opcode == HloOpcode::kCall ||
             it->second.opcode == HloOpcode::kWhile ||
             it->second.opcode == HloOpcode::kConditional;
    }
    return false;
  };

  for (const auto& key : reported_keys) {
    for (size_t i = 0; i < key.scope_instructions.size(); ++i) {
      const auto& s = key.scope_instructions[i];
      std::vector<ScopeInstruction> enc(key.scope_instructions.begin(),
                                        key.scope_instructions.begin() + i);
      CallInfo ci{s.instruction_name, enc};
      call_to_keys[ci].push_back(key);
    }
    CallInfo ci{key.tensor_key.instruction_name, key.scope_instructions};
    call_to_keys[ci].push_back(key);
  }

  // NOLINTNEXTLINE
  for (const auto& [ci, keys] : call_to_keys) {
    scope_to_calls[ci.enclosing_scope].push_back(ci);
  }

  auto reaches = [&](const CallInfo& from_ci, const CallInfo& to_ci) {
    auto& key = call_to_keys[to_ci];
    absl::flat_hash_set<AbsoluteScopedTensorKey> to_set(key.begin(), key.end());
    std::queue<AbsoluteScopedTensorKey> q;
    absl::flat_hash_set<AbsoluteScopedTensorKey> visited;
    for (const auto& k : call_to_keys[from_ci]) {
      q.push(k);
      visited.insert(k);
    }
    while (!q.empty()) {
      auto curr = q.front();
      q.pop();
      if (to_set.contains(curr)) {
        return true;
      }
      if (auto it = local_node_consumers.find(curr);
          it != local_node_consumers.end()) {
        for (const auto& next : it->second) {
          if (visited.insert(next).second) {
            q.push(next);
          }
        }
      }
    }
    return false;
  };

  std::vector<std::vector<ScopeInstruction>> sorted_scopes;
  // NOLINTNEXTLINE
  for (const auto& [scope, _] : scope_to_calls) {
    sorted_scopes.push_back(scope);
  }
  std::sort(sorted_scopes.begin(), sorted_scopes.end());

  for (const auto& scope : sorted_scopes) {
    auto calls = scope_to_calls[scope];
    std::sort(calls.begin(), calls.end(), [&](const auto& a, const auto& b) {
      auto it_a = dag_collection.inst_info.find(a.name);
      auto it_b = dag_collection.inst_info.find(b.name);
      int order_a = it_a != dag_collection.inst_info.end()
                        ? it_a->second.module_order
                        : 0;
      int order_b = it_b != dag_collection.inst_info.end()
                        ? it_b->second.module_order
                        : 0;
      return order_a < order_b;
    });

    for (size_t i = 0; i < calls.size(); ++i) {
      for (size_t j = i + 1; j < calls.size(); ++j) {
        if (is_call_like(calls[i].name) || is_call_like(calls[j].name)) {
          if (!reaches(calls[i], calls[j]) && !reaches(calls[j], calls[i])) {
            // Add edges from all output tensors of calls[i] to all input
            // tensors of calls[j]
            auto from_keys = call_to_keys[calls[i]];
            auto to_keys = call_to_keys[calls[j]];
            std::sort(from_keys.begin(), from_keys.end());
            std::sort(to_keys.begin(), to_keys.end());
            for (const auto& fk : from_keys) {
              for (const auto& tk : to_keys) {
                local_node_consumers[fk].push_back(tk);
                local_node_suppliers[tk].push_back(fk);
              }
            }
          }
        }
      }
    }
  }

  // 1. Group reported_keys by InstructionInstance
  LOG(INFO) << "ComputeDagLayout: Step 1 - Group by InstructionInstance";
  absl::flat_hash_map<InstructionInstance, std::vector<AbsoluteScopedTensorKey>>
      inst_to_keys;
  for (const auto& key : reported_keys) {
    inst_to_keys[GetInstInst(key)].push_back(key);
  }

  // 2. Build Instruction Instance Adjacency Graph
  LOG(INFO) << "ComputeDagLayout: Step 2 - Build Instruction Instance "
               "Adjacency Graph";
  absl::flat_hash_map<InstructionInstance,
                      absl::flat_hash_set<InstructionInstance>>
      inst_adj;
  // Iterate through local_node_consumers in a deterministic order.
  std::vector<AbsoluteScopedTensorKey> node_consumer_keys;
  // NOLINTNEXTLINE
  for (const auto& pair : local_node_consumers) {
    node_consumer_keys.push_back(pair.first);
  }
  std::sort(node_consumer_keys.begin(), node_consumer_keys.end());

  for (const auto& u : node_consumer_keys) {
    const auto& consumers = local_node_consumers.at(u);
    auto u_inst = GetInstInst(u);
    for (const auto& v : consumers) {
      auto v_inst = GetInstInst(v);
      if (!(u_inst == v_inst)) {  // Skip self-loops
        inst_adj[u_inst].insert(v_inst);
      }
    }
  }

  // 2b. Align supplier-less nodes with loop inputs
  LOG(INFO) << "ComputeDagLayout: Step 2b - Align supplier-less nodes with "
               "loop inputs";
  absl::flat_hash_set<InstructionInstance> has_suppliers;
  // NOLINTNEXTLINE
  for (const auto& [u_inst, v_insts] : inst_adj) {
    // NOLINTNEXTLINE
    for (const auto& v_inst : v_insts) {
      if (u_inst.scope == v_inst.scope) {
        has_suppliers.insert(v_inst);
      }
    }
  }

  absl::flat_hash_map<std::vector<ScopeInstruction>,
                      std::vector<InstructionInstance>>
      supplier_less_in_scope;
  // NOLINTNEXTLINE
  for (const auto& [inst, keys] : inst_to_keys) {
    bool is_entry_param = false;
    auto it = dag_collection.inst_info.find(inst.name);
    if (it != dag_collection.inst_info.end()) {
      if (it->second.opcode == HloOpcode::kParameter && inst.scope.empty()) {
        is_entry_param = true;
      }
    }

    if (!has_suppliers.contains(inst) && !is_entry_param) {
      supplier_less_in_scope[inst.scope].push_back(inst);
    }
  }

  // Add virtual dependencies from suppliers to supplier-less nodes in nested
  // scopes.
  std::vector<InstructionInstance> u_keys;
  // NOLINTNEXTLINE
  for (const auto& [u, _] : inst_adj) {
    u_keys.push_back(u);
  }

  for (const auto& u_inst : u_keys) {
    auto& adj = inst_adj[u_inst];
    std::vector<InstructionInstance> v_insts(adj.begin(), adj.end());
    for (const auto& v_inst : v_insts) {
      size_t common_len = 0;
      while (common_len < u_inst.scope.size() &&
             common_len < v_inst.scope.size() &&
             u_inst.scope[common_len] == v_inst.scope[common_len]) {
        common_len++;
      }

      // Any scope levels in v_inst beyond the common prefix indicate entering a
      // scope.
      for (size_t i = common_len; i < v_inst.scope.size(); ++i) {
        std::vector<ScopeInstruction> sub_scope(v_inst.scope.begin(),
                                                v_inst.scope.begin() + i + 1);
        // NOLINTNEXTLINE
        for (const auto& [s_scope, c_insts] : supplier_less_in_scope) {
          if (s_scope.size() >= sub_scope.size() &&
              std::equal(sub_scope.begin(), sub_scope.end(), s_scope.begin())) {
            for (const auto& c_inst : c_insts) {
              adj.insert(c_inst);
            }
          }
        }
      }
    }
  }

  // 2c. Add dependencies between consecutive loop iterations
  LOG(INFO) << "ComputeDagLayout: Step 2c - Add dependencies between "
               "consecutive loop iterations";
  absl::flat_hash_map<std::vector<ScopeInstruction>,
                      std::vector<InstructionInstance>>
      loop_to_insts;
  // NOLINTNEXTLINE
  for (const auto& [inst, keys] : inst_to_keys) {
    for (size_t i = 0; i < inst.scope.size(); ++i) {
      std::vector<ScopeInstruction> prefix(inst.scope.begin(),
                                           inst.scope.begin() + i + 1);
      loop_to_insts[prefix].push_back(inst);
    }
  }

  // NOLINTNEXTLINE
  for (const auto& [prefix, insts] : loop_to_insts) {
    if (!prefix.empty() && prefix.back().iteration_index > 0) {
      std::vector<ScopeInstruction> prev_prefix = prefix;
      CHECK(!prev_prefix.empty());
      prev_prefix.back().iteration_index--;
      auto it = loop_to_insts.find(prev_prefix);
      if (it != loop_to_insts.end()) {
        for (const auto& prev_inst : it->second) {
          for (const auto& curr_inst : insts) {
            inst_adj[prev_inst].insert(curr_inst);
          }
        }
      }
    }
  }

  // 3. Initialize X coordinates for Instruction Instances
  LOG(INFO) << "ComputeDagLayout: Step 3 - Initialize X coordinates";
  absl::flat_hash_map<InstructionInstance, double> inst_x;
  // NOLINTNEXTLINE
  for (const auto& [inst, keys] : inst_to_keys) {
    bool is_entry_param = false;
    auto it = dag_collection.inst_info.find(inst.name);
    if (it != dag_collection.inst_info.end()) {
      if (it->second.opcode == HloOpcode::kParameter && inst.scope.empty()) {
        is_entry_param = true;
      }
    }

    if (is_entry_param) {
      inst_x[inst] = 0.0;
    } else {
      inst_x[inst] = 1.0;
    }
  }

  // 4. Propagate X coordinates (Bellman-Ford style worklist)
  LOG(INFO) << "ComputeDagLayout: Step 4 - Propagate X coordinates";
  std::queue<InstructionInstance> worklist;
  // NOLINTNEXTLINE
  for (const auto& [inst, x] : inst_x) {
    worklist.push(inst);
  }

  // Limit iterations to prevent infinite loops if longer cycles exist in
  // projected graph
  size_t max_iterations = inst_to_keys.size() * inst_to_keys.size();
  size_t iterations = 0;

  while (!worklist.empty() && iterations < max_iterations) {
    if (iterations % 10000 == 0) {
      LOG(INFO) << "ComputeDagLayout: Step 4 - iterations=" << iterations
                << ", worklist_size=" << worklist.size();
    }
    auto u = worklist.front();
    worklist.pop();
    iterations++;

    double u_x = inst_x[u];
    auto it = inst_adj.find(u);
    if (it != inst_adj.end()) {
      // NOLINTNEXTLINE
      for (const auto& v : it->second) {
        auto& x = inst_x[v];
        if (x < u_x + 1.0) {
          x = u_x + 1.0;
          worklist.push(v);
        }
      }
    }
  }

  // Map keys to IDs for fast lookup
  absl::flat_hash_map<AbsoluteScopedTensorKey, int> key_to_id;
  std::vector<AbsoluteScopedTensorKey> id_to_key(reported_keys.size());
  for (int i = 0; i < reported_keys.size(); ++i) {
    key_to_id[reported_keys[i]] = i;
    id_to_key[i] = reported_keys[i];
  }

  // Convert adjacency to ID-based
  std::vector<std::vector<int>> id_consumers(reported_keys.size());
  // NOLINTNEXTLINE
  for (const auto& [u, consumers] : local_node_consumers) {
    auto it_u = key_to_id.find(u);
    if (it_u == key_to_id.end()) {
      continue;
    }
    int u_id = it_u->second;
    for (const auto& v : consumers) {
      auto it_v = key_to_id.find(v);
      if (it_v == key_to_id.end()) {
        continue;
      }
      id_consumers[u_id].push_back(it_v->second);
    }
  }

  std::vector<std::vector<int>> id_suppliers(reported_keys.size());
  // NOLINTNEXTLINE
  for (const auto& [c, suppliers] : local_node_suppliers) {
    auto it_c = key_to_id.find(c);
    if (it_c == key_to_id.end()) {
      continue;
    }
    int c_id = it_c->second;
    for (const auto& s : suppliers) {
      auto it_s = key_to_id.find(s);
      if (it_s == key_to_id.end()) {
        continue;
      }
      id_suppliers[c_id].push_back(it_s->second);
    }
  }

  // 5. Assign X to Nodes and Group into Layers
  LOG(INFO)
      << "ComputeDagLayout: Step 5 - Assign X to Nodes and Group into Layers";
  std::map<int, std::vector<int>> layers;
  std::vector<double> node_x(reported_keys.size());

  for (int i = 0; i < reported_keys.size(); ++i) {
    const auto& key = reported_keys[i];
    double x = inst_x[GetInstInst(key)];
    node_x[i] = x;
    layers[static_cast<int>(x)].push_back(i);
  }

  // 5b. Find Connected Components of Nodes
  LOG(INFO) << "ComputeDagLayout: Step 5b - Find Connected Components";
  std::vector<absl::flat_hash_set<int>> node_undirected_adj(
      reported_keys.size());
  for (int u = 0; u < reported_keys.size(); ++u) {
    for (int v : id_consumers[u]) {
      node_undirected_adj[u].insert(v);
      node_undirected_adj[v].insert(u);
    }
    for (int s : id_suppliers[u]) {
      node_undirected_adj[u].insert(s);
      node_undirected_adj[s].insert(u);
    }
  }

  std::vector<int> node_comp(reported_keys.size(), -1);
  int num_node_components = 0;
  for (int i = 0; i < reported_keys.size(); ++i) {
    if (node_comp[i] != -1) {
      continue;
    }
    std::queue<int> q;
    q.push(i);
    node_comp[i] = num_node_components;
    while (!q.empty()) {
      auto u = q.front();
      q.pop();
      // NOLINTNEXTLINE
      for (int v : node_undirected_adj[u]) {
        if (node_comp[v] == -1) {
          node_comp[v] = num_node_components;
          q.push(v);
        }
      }
    }
    num_node_components++;
  }

  // 6. Y Layout (Iterative Barycenter Heuristic)
  LOG(INFO) << "ComputeDagLayout: Step 6 - Y Layout";
  std::vector<double> node_y(reported_keys.size());
  constexpr double kMinSpacing = 2.0;

  // Initial Y: group by component first, then instruction name
  for (auto& [x, nodes] : layers) {
    std::sort(nodes.begin(), nodes.end(), [&](int a, int b) {
      int comp_a = node_comp[a];
      int comp_b = node_comp[b];
      if (comp_a != comp_b) {
        return comp_a < comp_b;
      }
      return id_to_key[a].tensor_key.instruction_name <
             id_to_key[b].tensor_key.instruction_name;
    });
    for (int i = 0; i < nodes.size(); ++i) {
      node_y[nodes[i]] = i * kMinSpacing;
    }
  }

  // Barycenter Iterations
  constexpr int kBarycenterIterations = 20;
  for (int iter = 0; iter < kBarycenterIterations; ++iter) {
    LOG(INFO) << "ComputeDagLayout: Step 6 - Barycenter Iteration " << iter;
    // Forward Pass (Left to Right)
    for (auto& [x, nodes] : layers) {
      for (int u : nodes) {
        double sum_y = 0.0;
        int count = 0;
        for (int s : id_suppliers[u]) {
          sum_y += node_y[s];
          count++;
        }
        if (count > 0) {
          node_y[u] = sum_y / count;
        }
      }
      // Sort and Space Out with Overlap Resolution per Component independently
      std::sort(nodes.begin(), nodes.end(), [&](int a, int b) {
        int comp_a = node_comp[a];
        int comp_b = node_comp[b];
        if (comp_a != comp_b) {
          return comp_a < comp_b;
        }
        return node_y[a] < node_y[b];
      });
      if (nodes.size() > 1) {
        size_t i = 0;
        while (i < nodes.size()) {
          int c = node_comp[nodes[i]];
          size_t j = i;
          double sum_before = 0;
          while (j < nodes.size() && node_comp[nodes[j]] == c) {
            sum_before += node_y[nodes[j]];
            j++;
          }

          if (j - i > 1) {
            double avg_before = sum_before / (j - i);
            for (size_t k = i + 1; k < j; ++k) {
              if (node_y[nodes[k]] < node_y[nodes[k - 1]] + kMinSpacing) {
                node_y[nodes[k]] = node_y[nodes[k - 1]] + kMinSpacing;
              }
            }
            double sum_after = 0;
            for (size_t k = i; k < j; ++k) {
              sum_after += node_y[nodes[k]];
            }
            double avg_after = sum_after / (j - i);
            double shift = avg_before - avg_after;
            for (size_t k = i; k < j; ++k) {
              node_y[nodes[k]] += shift;
            }
          }
          i = j;
        }
      }
    }

    // Backward Pass (Right to Left)
    auto it = layers.rbegin();
    while (it != layers.rend()) {
      auto& nodes = it->second;
      for (int u : nodes) {
        double sum_y = 0.0;
        int count = 0;
        for (int c : id_consumers[u]) {
          sum_y += node_y[c];
          count++;
        }
        if (count > 0) {
          node_y[u] = sum_y / count;
        }
      }
      // Sort and Space Out with Overlap Resolution per Component independently
      std::sort(nodes.begin(), nodes.end(), [&](int a, int b) {
        int comp_a = node_comp[a];
        int comp_b = node_comp[b];
        if (comp_a != comp_b) {
          return comp_a < comp_b;
        }
        return node_y[a] < node_y[b];
      });
      if (nodes.size() > 1) {
        size_t i = 0;
        while (i < nodes.size()) {
          int c = node_comp[nodes[i]];
          size_t j = i;
          double sum_before = 0;
          while (j < nodes.size() && node_comp[nodes[j]] == c) {
            sum_before += node_y[nodes[j]];
            j++;
          }

          if (j - i > 1) {
            double avg_before = sum_before / (j - i);
            for (size_t k = i + 1; k < j; ++k) {
              if (node_y[nodes[k]] < node_y[nodes[k - 1]] + kMinSpacing) {
                node_y[nodes[k]] = node_y[nodes[k - 1]] + kMinSpacing;
              }
            }
            double sum_after = 0;
            for (size_t k = i; k < j; ++k) {
              sum_after += node_y[nodes[k]];
            }
            double avg_after = sum_after / (j - i);
            double shift = avg_before - avg_after;
            for (size_t k = i; k < j; ++k) {
              node_y[nodes[k]] += shift;
            }
          }
          i = j;
        }
      }

      it++;
    }

    double min_y = std::numeric_limits<double>::max();
    double max_y = std::numeric_limits<double>::lowest();
    for (double y : node_y) {
      min_y = std::min(min_y, y);
      max_y = std::max(max_y, y);
    }
    LOG(INFO) << "ComputeDagLayout: Step 6 - Barycenter Iteration " << iter
              << " completed. min_y = " << min_y << ", max_y = " << max_y;
  }

  // 6b. Post-processing: Global Rigid Shift of Components
  LOG(INFO) << "ComputeDagLayout: Step 6b - Global Rigid Shift of Components";
  std::vector<std::vector<int>> comp_to_keys(num_node_components);
  for (int i = 0; i < reported_keys.size(); ++i) {
    comp_to_keys[node_comp[i]].push_back(i);
  }

  if (num_node_components > 1) {
    struct CompInfo {
      int id;
      double avg_y;
    };
    std::vector<CompInfo> sorted_comps;
    for (int c = 0; c < num_node_components; ++c) {
      const auto& keys = comp_to_keys[c];
      double sum_y = 0;
      for (int key_id : keys) {
        sum_y += node_y[key_id];
      }
      if (!keys.empty()) {
        sorted_comps.push_back({c, sum_y / keys.size()});
      }
    }
    std::sort(sorted_comps.begin(), sorted_comps.end(),
              [](const auto& a, const auto& b) { return a.avg_y < b.avg_y; });

    // Maintain max Y for each layer incrementally
    absl::flat_hash_map<int, double> layer_max_y_prev;

    // Initialize with the first component
    int first_comp = sorted_comps[0].id;
    for (int key_id : comp_to_keys[first_comp]) {
      int x = static_cast<int>(node_x[key_id]);
      auto& prev = layer_max_y_prev[x];
      if (!layer_max_y_prev.contains(x) || node_y[key_id] > prev) {
        prev = node_y[key_id];
      }
    }

    for (size_t i = 1; i < sorted_comps.size(); ++i) {
      int curr_comp = sorted_comps[i].id;
      const auto& curr_keys = comp_to_keys[curr_comp];

      absl::flat_hash_map<int, double> layer_min_y_curr;
      absl::flat_hash_map<int, double> layer_max_y_curr;
      for (int key_id : curr_keys) {
        int x = static_cast<int>(node_x[key_id]);
        auto& curr = layer_min_y_curr[x];
        if (!layer_min_y_curr.contains(x) || node_y[key_id] < curr) {
          curr = node_y[key_id];
        }
        if (!layer_max_y_curr.contains(x) ||
            node_y[key_id] > layer_max_y_curr[x]) {
          layer_max_y_curr[x] = node_y[key_id];
        }
      }

      double min_gap = std::numeric_limits<double>::max();
      bool has_overlap = false;
      // NOLINTNEXTLINE
      for (const auto& [x, min_y] : layer_min_y_curr) {
        if (layer_max_y_prev.contains(x)) {
          double gap = min_y - layer_max_y_prev[x];
          if (gap < min_gap) {
            min_gap = gap;
          }
          has_overlap = true;
        }
      }

      double shift = 0;
      if (has_overlap && min_gap < std::numeric_limits<double>::max()) {
        shift = min_gap - kMinSpacing;
        if (shift > 0) {
          for (int key_id : curr_keys) {
            node_y[key_id] -= shift;
          }
        } else {
          shift = 0;
        }
      }

      // NOLINTNEXTLINE
      for (const auto& [x, max_y] : layer_max_y_curr) {
        double shifted_max_y = max_y - shift;
        auto& prev = layer_max_y_prev[x];
        if (!layer_max_y_prev.contains(x) || shifted_max_y > prev) {
          prev = shifted_max_y;
        }
      }
    }
  }

  // 6c. Enforce minimum Y separation (at least 1 unit) for nodes sharing the
  // same X.
  LOG(INFO)
      << "ComputeDagLayout: Step 6c - Enforce minimum Y separation per layer";
  for (auto& [x, nodes] : layers) {
    if (nodes.size() <= 1) {
      continue;
    }
    std::sort(nodes.begin(), nodes.end(),
              [&](int a, int b) { return node_y[a] < node_y[b]; });
    for (size_t i = 1; i < nodes.size(); ++i) {
      if (node_y[nodes[i]] < node_y[nodes[i - 1]] + 1.0) {
        node_y[nodes[i]] = node_y[nodes[i - 1]] + 1.0;
      }
    }
  }

  // 7. Final Assembly
  LOG(INFO) << "ComputeDagLayout: Step 7 - Final Assembly";
  absl::flat_hash_map<AbsoluteScopedTensorKey, NodePosition> result;
  for (int i = 0; i < reported_keys.size(); ++i) {
    result[reported_keys[i]] = {node_x[i], node_y[i]};
  }

  return result;
}

}  // namespace

absl::flat_hash_map<AbsoluteScopedTensorKey, NodePosition>
ComputeDagLayoutForTesting(
    const ComputationDagCollection& dag_collection,
    absl::Span<const AbsoluteScopedTensorKey> reported_keys,
    const absl::flat_hash_map<AbsoluteScopedTensorKey,
                              std::vector<AbsoluteScopedTensorKey>>&
        node_consumers,
    const absl::flat_hash_map<AbsoluteScopedTensorKey,
                              std::vector<AbsoluteScopedTensorKey>>&
        node_suppliers) {
  return ComputeDagLayout(dag_collection, reported_keys, node_consumers,
                          node_suppliers);
}

void GenerateHloHtmlDumps(int replica_id, const HloModule& baseline_module,
                          const HloModule& target_module,
                          absl::string_view output_dir,
                          absl::string_view comparison_results_path) {
  LOG(INFO) << "Generating HLO HTML dumps for replica " << replica_id << " to "
            << output_dir;

  struct AggregatedStats {
    HloNodeComparisonStats comp;
    std::vector<TensorSummaryProto> own_protos;
    std::vector<TensorSummaryProto> paired_protos;
  };

  absl::flat_hash_map<HloHtmlTensorKey, AggregatedStats> baseline_stats;
  absl::flat_hash_map<HloHtmlTensorKey, AggregatedStats> target_stats;

  absl::flat_hash_map<HloHtmlTensorKey, int64_t> baseline_first_indices;
  absl::flat_hash_map<HloHtmlTensorKey, int64_t> target_first_indices;
  absl::flat_hash_map<AbsoluteScopedTensorKey, int64_t> baseline_node_ids;
  absl::flat_hash_map<AbsoluteScopedTensorKey, int64_t> target_node_ids;

  std::vector<AbsoluteScopedTensorKey> baseline_reported_keys;
  absl::flat_hash_set<AbsoluteScopedTensorKey> baseline_seen;
  std::vector<AbsoluteScopedTensorKey> target_reported_keys;
  absl::flat_hash_set<AbsoluteScopedTensorKey> target_seen;
  absl::flat_hash_map<AbsoluteScopedTensorKey, double> baseline_scores;
  absl::flat_hash_map<AbsoluteScopedTensorKey, double> target_scores;

  LOG(INFO) << "Reading comparison results from " << comparison_results_path;

  riegeli::RecordReader reader{riegeli::FdReader(comparison_results_path)};
  ComparisonResultProto result;
  int64_t index = 0;
  while (reader.ReadRecord(result)) {
    const auto& b_tk_proto = result.baseline_tensor_key().tensor_key();
    const auto& t_tk_proto = result.target_tensor_key().tensor_key();

    HloHtmlTensorKey b_key;
    b_key.name = b_tk_proto.instruction_name();
    for (int64_t i : b_tk_proto.shape_index()) {
      b_key.shape_index.push_back(i);
    }

    HloHtmlTensorKey t_key;
    t_key.name = t_tk_proto.instruction_name();
    for (int64_t i : t_tk_proto.shape_index()) {
      t_key.shape_index.push_back(i);
    }

    if (!b_tk_proto.instruction_name().empty()) {
      if (!baseline_first_indices.contains(b_key)) {
        baseline_first_indices[b_key] = index;
      }
      auto key =
          AbsoluteScopedTensorKey::FromProto(result.baseline_tensor_key());
      if (baseline_seen.insert(key).second) {
        baseline_reported_keys.push_back(key);
      }
      baseline_scores[key] = result.diff_score();
      baseline_node_ids[key] = index;
    }
    if (!t_tk_proto.instruction_name().empty()) {
      if (!target_first_indices.contains(t_key)) {
        target_first_indices[t_key] = index;
      }
      auto key = AbsoluteScopedTensorKey::FromProto(result.target_tensor_key());
      if (target_seen.insert(key).second) {
        target_reported_keys.push_back(key);
      }
      target_scores[key] = result.diff_score();
      target_node_ids[key] = index;
    }

    auto& b_stats = baseline_stats[b_key];
    auto& t_stats = target_stats[t_key];

    double score = result.diff_score();

    for (auto* stats : {&b_stats.comp, &t_stats.comp}) {
      if (score == -1.0) {
        stats->not_comparable = true;
      } else {
        stats->score_count++;
        stats->score_min = std::min(stats->score_min, score);
        stats->score_max = std::max(stats->score_max, score);
        stats->score_mean += score;
      }
    }

    for (const auto& s : result.baseline_tensor_summaries()) {
      b_stats.own_protos.push_back(s);
      t_stats.paired_protos.push_back(s);
    }
    for (const auto& s : result.target_tensor_summaries()) {
      t_stats.own_protos.push_back(s);
      b_stats.paired_protos.push_back(s);
    }
    index++;
  }
  LOG(INFO) << "Finished reading records. Total records: " << index;
  CHECK(reader.Close()) << reader.status();

  absl::flat_hash_map<HloHtmlTensorKey, TensorAnnotation> baseline_annotations;
  absl::flat_hash_map<HloHtmlTensorKey, TensorAnnotation> target_annotations;

  auto build_annotations = [&](const auto& stats_map, const auto& first_indices,
                               bool is_baseline, auto& annotations) {
    // NOLINTNEXTLINE
    for (const auto& [key, stats] : stats_map) {
      TensorAnnotation ann;
      if (stats.comp.not_comparable) {
        ann.background_color = GetColorForScore(-1.0);
      } else if (stats.comp.score_count > 0) {
        ann.background_color = GetColorForScore(stats.comp.score_max);
      }

      std::optional<FloatBlockSummary> own_run;
      if (!stats.own_protos.empty()) {
        own_run = CombineSummaries(stats.own_protos);
      }
      std::optional<FloatBlockSummary> paired_run;
      if (!stats.paired_protos.empty()) {
        paired_run = CombineSummaries(stats.paired_protos);
      }

      auto* baseline_ptr = is_baseline ? (own_run ? &*own_run : nullptr)
                                       : (paired_run ? &*paired_run : nullptr);
      auto* target_ptr = is_baseline ? (paired_run ? &*paired_run : nullptr)
                                     : (own_run ? &*own_run : nullptr);

      ann.tooltip_data = GetTooltipData(&stats.comp, baseline_ptr, target_ptr);

      if (first_indices.contains(key)) {
        ann.anchor_id = absl::StrCat("step", first_indices.at(key));
        annotations[key] = ann;
      }
    }
  };

  LOG(INFO) << "Building baseline annotations...";
  build_annotations(baseline_stats, baseline_first_indices, true,
                    baseline_annotations);
  LOG(INFO) << "Building target annotations...";
  build_annotations(target_stats, target_first_indices, false,
                    target_annotations);

  std::string baseline_out = tsl::io::JoinPath(
      output_dir, absl::StrFormat("replica_%d.baseline.hlo.html", replica_id));
  std::string target_out = tsl::io::JoinPath(
      output_dir, absl::StrFormat("replica_%d.target.hlo.html", replica_id));

  xla::StackFrameIndexProto baseline_sf_index;
  if (baseline_module.ToProto().has_stack_frame_index()) {
    baseline_sf_index = baseline_module.ToProto().stack_frame_index();
  }
  xla::StackFrameIndexProto target_sf_index;
  if (target_module.ToProto().has_stack_frame_index()) {
    target_sf_index = target_module.ToProto().stack_frame_index();
  }

  // Compute DAGs and Layouts
  LOG(INFO) << "Generating baseline computation DAG collection...";
  auto baseline_dag =
      GenerateComputationDagCollection(baseline_module, baseline_reported_keys);

  LOG(INFO) << "Building baseline node adjacency...";
  absl::flat_hash_map<AbsoluteScopedTensorKey,
                      std::vector<AbsoluteScopedTensorKey>>
      baseline_node_consumers;
  absl::flat_hash_map<AbsoluteScopedTensorKey,
                      std::vector<AbsoluteScopedTensorKey>>
      baseline_node_suppliers;
  for (const auto& key : baseline_reported_keys) {
    auto consumers = FindConsumers(key, baseline_dag);
    baseline_node_consumers[key] = consumers;
    for (const auto& c : consumers) {
      baseline_node_suppliers[c].push_back(key);
    }
  }

  LOG(INFO) << "Computing baseline DAG layout...";
  auto baseline_layout =
      ComputeDagLayout(baseline_dag, baseline_reported_keys,
                       baseline_node_consumers, baseline_node_suppliers);
  debug_info::GraphData baseline_graph_data;

  LOG(INFO) << "Generating target computation DAG collection...";
  auto target_dag =
      GenerateComputationDagCollection(target_module, target_reported_keys);

  LOG(INFO) << "Building target node adjacency...";
  absl::flat_hash_map<AbsoluteScopedTensorKey,
                      std::vector<AbsoluteScopedTensorKey>>
      target_node_consumers;
  absl::flat_hash_map<AbsoluteScopedTensorKey,
                      std::vector<AbsoluteScopedTensorKey>>
      target_node_suppliers;
  for (const auto& key : target_reported_keys) {
    auto consumers = FindConsumers(key, target_dag);
    target_node_consumers[key] = consumers;
    for (const auto& c : consumers) {
      target_node_suppliers[c].push_back(key);
    }
  }

  LOG(INFO) << "Computing target DAG layout...";
  auto target_layout =
      ComputeDagLayout(target_dag, target_reported_keys, target_node_consumers,
                       target_node_suppliers);
  debug_info::GraphData target_graph_data;

  auto populate_graph_data =
      [&](const auto& reported_keys, const auto& layout, const auto& scores,
          const auto& node_consumers_arg, const auto& first_indices,
          const auto& node_ids, debug_info::GraphData& graph_data) {
        for (const auto& key : reported_keys) {
          auto pos_it = layout.find(key);
          if (pos_it == layout.end()) {
            continue;
          }
          const auto& pos = pos_it->second;

          auto score_it = scores.find(key);
          double score = (score_it != scores.end()) ? score_it->second : -1.0;

          HloHtmlTensorKey html_key;
          html_key.name = key.tensor_key.instruction_name;
          for (int64_t i : key.tensor_key.shape_index) {
            html_key.shape_index.push_back(static_cast<int>(i));
          }
          int64_t node_id = -1;
          auto id_it = node_ids.find(key);
          if (id_it != node_ids.end()) {
            node_id = id_it->second;
          }

          int64_t anchor_id = -1;
          if (!key.tensor_key.instruction_name.empty()) {
            anchor_id = first_indices.at(html_key);
          }

          std::string key_str = key.ToString();
          graph_data.nodes.push_back(
              {node_id, pos.x, pos.y, score, key_str, anchor_id});

          auto con_it = node_consumers_arg.find(key);
          if (con_it != node_consumers_arg.end()) {
            for (const auto& consumer : con_it->second) {
              int64_t consumer_id = -1;
              auto c_it = node_ids.find(consumer);
              if (c_it != node_ids.end()) {
                consumer_id = c_it->second;
              }
              graph_data.edges.push_back({node_id, consumer_id});
            }
          }
        }
      };

  LOG(INFO) << "Populating baseline graph data...";
  populate_graph_data(baseline_reported_keys, baseline_layout, baseline_scores,
                      baseline_node_consumers, baseline_first_indices,
                      baseline_node_ids, baseline_graph_data);
  LOG(INFO) << "Populating target graph data...";
  populate_graph_data(target_reported_keys, target_layout, target_scores,
                      target_node_consumers, target_first_indices,
                      target_node_ids, target_graph_data);
  auto dump_to_json = [&](const debug_info::GraphData& gd,
                          const std::string& path) {
    std::string json = "{\n  \"nodes\": [\n";
    for (size_t i = 0; i < gd.nodes.size(); ++i) {
      const auto& node = gd.nodes[i];
      json += absl::StrFormat(
          "    {\"id\": %d, \"x\": %g, \"y\": %g, \"diffScore\": %g, \"key\": "
          "\"%s\", \"anchorId\": %d}%s\n",
          node.id, node.x, node.y, node.diff_score, node.key, node.anchor_id,
          (i + 1 == gd.nodes.size() ? "" : ","));
    }
    json += "  ],\n  \"edges\": [\n";
    for (size_t i = 0; i < gd.edges.size(); ++i) {
      const auto& edge = gd.edges[i];
      json += absl::StrFormat(
          "    {\"supplierId\": %d, \"consumerId\": %d}%s\n", edge.supplier_id,
          edge.consumer_id, (i + 1 == gd.edges.size() ? "" : ","));
    }
    json += "  ]\n}\n";
    auto st = tsl::WriteStringToFile(tsl::Env::Default(), path, json);
    LOG(INFO) << "Dumped graph JSON to " << path << " with status " << st;
  };
  dump_to_json(baseline_graph_data,
               tsl::io::JoinPath(
                   output_dir, absl::StrFormat("replica_%d.baseline.graph.json",
                                               replica_id)));
  dump_to_json(target_graph_data,
               tsl::io::JoinPath(output_dir,
                                 absl::StrFormat("replica_%d.target.graph.json",
                                                 replica_id)));
  // Run in parallel
  LOG(INFO) << "Dumping HLO to HTML in parallel...";
  absl::Status b_res;
  absl::Status t_res;
  {
    tsl::thread::ThreadPool pool(tsl::Env::Default(), "DumpHloToHtml", 2);
    pool.Schedule([&]() {
      b_res = DumpHloToHtml(absl::StrCat(baseline_module.name(), " (baseline)"),
                            baseline_module.ToString(), baseline_annotations,
                            baseline_out, std::nullopt, std::nullopt,
                            &baseline_sf_index, &baseline_graph_data);
    });
    pool.Schedule([&]() {
      t_res = DumpHloToHtml(absl::StrCat(target_module.name(), " (target)"),
                            target_module.ToString(), target_annotations,
                            target_out, std::nullopt, std::nullopt,
                            &target_sf_index, &target_graph_data);
    });
  }

  if (!b_res.ok()) {
    LOG(ERROR) << "Failed to generate baseline HLO HTML dump: " << b_res;
  }
  if (!t_res.ok()) {
    LOG(ERROR) << "Failed to generate target HLO HTML dump: " << t_res;
  }

  absl::FPrintF(stderr, "HLO HTML dumps are stored in %s\n", output_dir);
}

void GenerateSingleHloHtmlDump(int replica_id, const HloModule& module,
                               absl::string_view output_dir,
                               absl::string_view recovered_summaries_path) {
  LOG(INFO) << "Generating Single HLO HTML dump for replica " << replica_id
            << " to " << output_dir;

  absl::flat_hash_map<HloHtmlTensorKey, std::vector<TensorSummaryProto>>
      all_summaries;

  riegeli::RecordReader reader{riegeli::FdReader(recovered_summaries_path)};
  RecoveredTensorSummaryProto result;
  while (reader.ReadRecord(result)) {
    const auto& tk = result.tensor_key().tensor_key();
    HloHtmlTensorKey key;
    key.name = tk.instruction_name();
    for (int64_t i : tk.shape_index()) {
      key.shape_index.push_back(i);
    }

    for (const auto& s : result.original_tensor_summary().summaries()) {
      all_summaries[key].push_back(s);
    }
  }
  CHECK(reader.Close()) << reader.status();

  absl::flat_hash_map<HloHtmlTensorKey, TensorAnnotation> annotations;
  // NOLINTNEXTLINE
  for (const auto& [key, protos] : all_summaries) {
    FloatBlockSummary run = CombineSummaries(protos);
    TensorAnnotation ann;
    ann.tooltip_data = GetTooltipData(nullptr, &run, nullptr);
    annotations[key] = ann;
  }

  std::string out_path = tsl::io::JoinPath(
      output_dir, absl::StrFormat("replica_%d.hlo.html", replica_id));

  xla::StackFrameIndexProto sf_index;
  if (module.ToProto().has_stack_frame_index()) {
    sf_index = module.ToProto().stack_frame_index();
  }

  auto status = DumpHloToHtml(module.name(), module.ToString(), annotations,
                              out_path, std::nullopt, std::nullopt, &sf_index);
  if (!status.ok()) {
    LOG(ERROR) << "Failed to generate single HLO HTML dump: " << status;
  }

  absl::FPrintF(stderr, "HLO HTML dump is stored in %s\n", out_path);
}

namespace {

std::vector<LocalGraphNode> GetLocalConsumersInternal(
    const LocalGraphNode& node,
    const absl::flat_hash_map<std::string, const HloInstruction*>&
        name_to_inst) {
  std::vector<LocalGraphNode> consumers;
  if (node.operand_index >= 0) {
    return consumers;  // Call Inputs are sinks locally
  }

  auto it = name_to_inst.find(node.instruction_name);
  if (it == name_to_inst.end()) {
    return consumers;
  }
  const HloInstruction* inst = it->second;

  for (const HloInstruction* user : inst->users()) {
    for (int k = 0; k < user->operand_count(); ++k) {
      if (user->operand(k) == inst) {
        if (user->opcode() == HloOpcode::kCall ||
            user->opcode() == HloOpcode::kWhile ||
            user->opcode() == HloOpcode::kConditional) {
          consumers.push_back({std::string(user->name()), k, node.shape_index});
        } else if (user->opcode() == HloOpcode::kTuple) {
          std::vector<int64_t> new_idx = {k};
          new_idx.insert(new_idx.end(), node.shape_index.begin(),
                         node.shape_index.end());
          consumers.push_back({std::string(user->name()), -1, new_idx});
        } else if (user->opcode() == HloOpcode::kGetTupleElement) {
          int gte_idx = user->tuple_index();
          if (k == 0 && !node.shape_index.empty() &&
              node.shape_index[0] == gte_idx) {
            std::vector<int64_t> new_idx(node.shape_index.begin() + 1,
                                         node.shape_index.end());
            consumers.push_back({std::string(user->name()), -1, new_idx});
          }
        } else {
          ShapeUtil::ForEachSubshape(
              user->shape(),
              [&](const Shape& subshape, const ShapeIndex& index) {
                std::vector<int64_t> idx(index.begin(), index.end());
                consumers.push_back({std::string(user->name()), -1, idx});
              });
        }
      }
    }
  }
  return consumers;
}

}  // namespace

ComputationDagCollection GenerateComputationDagCollection(
    const HloModule& module,
    absl::Span<const AbsoluteScopedTensorKey> reported_keys) {
  ComputationDagCollection dag_collection;
  absl::flat_hash_map<std::string, const HloInstruction*> local_name_to_inst;

  for (const auto* comp : module.computations()) {
    ComputationInfo comp_info;
    comp_info.root_name = comp->root_instruction()->name();
    for (const auto* param : comp->parameter_instructions()) {
      comp_info.parameters.push_back(std::string(param->name()));
    }
    dag_collection.comp_info[comp->name()] = comp_info;

    int module_order = 0;
    for (const auto* inst : comp->instructions()) {
      local_name_to_inst[inst->name()] = inst;

      InstructionInfo info;
      info.parent_computation = comp->name();
      info.opcode = inst->opcode();
      info.module_order = module_order++;
      if (inst->opcode() == HloOpcode::kCall) {
        info.called_computations.push_back(
            std::string(inst->to_apply()->name()));
      } else if (inst->opcode() == HloOpcode::kWhile) {
        info.called_computations.push_back(
            std::string(inst->while_body()->name()));
        info.called_computations.push_back(
            std::string(inst->while_condition()->name()));
      } else if (inst->opcode() == HloOpcode::kConditional) {
        for (int i = 0; i < inst->branch_count(); ++i) {
          info.called_computations.push_back(
              std::string(inst->branch_computation(i)->name()));
        }
      }
      if (inst->opcode() == HloOpcode::kParameter) {
        info.parameter_number = inst->parameter_number();
      }
      dag_collection.inst_info[inst->name()] = info;
    }
  }

  absl::flat_hash_map<const HloComputation*,
                      absl::flat_hash_set<LocalGraphNode>>
      reported_per_comp;
  for (const auto& key : reported_keys) {
    const auto& tk = key.tensor_key;
    auto it = local_name_to_inst.find(tk.instruction_name);
    if (it != local_name_to_inst.end()) {
      const HloInstruction* inst = it->second;
      const HloComputation* comp = inst->parent();
      LocalGraphNode node;
      node.instruction_name = tk.instruction_name;
      node.operand_index = -1;
      node.shape_index.assign(tk.shape_index.begin(), tk.shape_index.end());
      reported_per_comp[comp].insert(node);
      dag_collection.reported_map[node].push_back(key);
    }
  }

  for (const auto* comp : module.computations()) {
    absl::flat_hash_set<LocalGraphNode> key_nodes;

    // 1. Reported
    auto rep_it = reported_per_comp.find(comp);
    if (rep_it != reported_per_comp.end()) {
      // NOLINTNEXTLINE
      for (const auto& n : rep_it->second) {
        key_nodes.insert(n);
      }
    }

    // 2. Parameters
    for (const auto* param : comp->parameter_instructions()) {
      ShapeUtil::ForEachSubshape(
          param->shape(), [&](const Shape& subshape, const ShapeIndex& index) {
            std::vector<int64_t> idx(index.begin(), index.end());
            key_nodes.insert({std::string(param->name()), -1, idx});
          });
    }

    // 3. Root
    const HloInstruction* root = comp->root_instruction();
    ShapeUtil::ForEachSubshape(
        root->shape(), [&](const Shape& subshape, const ShapeIndex& index) {
          std::vector<int64_t> idx(index.begin(), index.end());
          key_nodes.insert({std::string(root->name()), -1, idx});
        });

    // 4. Calls
    for (const auto* inst : comp->instructions()) {
      if (inst->opcode() == HloOpcode::kCall ||
          inst->opcode() == HloOpcode::kWhile ||
          inst->opcode() == HloOpcode::kConditional) {
        // Inputs
        for (int k = 0; k < inst->operand_count(); ++k) {
          const HloInstruction* operand = inst->operand(k);
          ShapeUtil::ForEachSubshape(
              operand->shape(),
              [&](const Shape& subshape, const ShapeIndex& index) {
                std::vector<int64_t> idx(index.begin(), index.end());
                key_nodes.insert({std::string(inst->name()), k, idx});
              });
        }
        // Outputs
        ShapeUtil::ForEachSubshape(
            inst->shape(), [&](const Shape& subshape, const ShapeIndex& index) {
              std::vector<int64_t> idx(index.begin(), index.end());
              key_nodes.insert({std::string(inst->name()), -1, idx});
            });
      }
    }

    SimplifiedGraph& simplified_graph = dag_collection.graphs[comp->name()];

    // NOLINTNEXTLINE
    for (const auto& u : key_nodes) {
      std::queue<LocalGraphNode> q;
      q.push(u);
      absl::flat_hash_set<LocalGraphNode> visited;
      visited.insert(u);

      while (!q.empty()) {
        LocalGraphNode curr = q.front();
        q.pop();

        std::vector<LocalGraphNode> local_consumers =
            GetLocalConsumersInternal(curr, local_name_to_inst);
        for (const auto& v : local_consumers) {
          if (key_nodes.contains(v)) {
            if (!(v == u)) {
              simplified_graph.consumers[u].push_back(v);
              simplified_graph.suppliers[v].push_back(u);
            }
          } else {
            if (!visited.contains(v)) {
              visited.insert(v);
              q.push(v);
            }
          }
        }
      }
    }
  }

  // Compute max_rep_iter_map
  // NOLINTNEXTLINE
  for (const auto& [node, reps] : dag_collection.reported_map) {
    for (const auto& rep : reps) {
      for (const auto& s_inst : rep.scope_instructions) {
        auto [it, inserted] = dag_collection.max_rep_iter_map.try_emplace(
            s_inst.instruction_name, s_inst.iteration_index);
        if (!inserted) {
          it->second = std::max(it->second, s_inst.iteration_index);
        }
      }
    }
  }

  return dag_collection;
}

namespace {

template <typename T1, typename T2>
bool MatchScopesConsumers(const T1& curr_scope, const T2& rep_scope) {
  if (curr_scope.size() != rep_scope.size()) {
    return false;
  }
  for (size_t i = 0; i < curr_scope.size(); ++i) {
    if (curr_scope[i].instruction_name != rep_scope[i].instruction_name) {
      return false;
    }
    int64_t curr_iter = curr_scope[i].iteration_index;
    int64_t rep_iter = rep_scope[i].iteration_index;
    if (curr_iter != rep_iter) {
      if (curr_iter == -1 || rep_iter == -1) {
        continue;
      }
      return false;
    }
  }
  return true;
}

template <typename T1, typename T2>
bool MatchScopesSuppliers(const T1& curr_scope, const T2& rep_scope) {
  if (curr_scope.size() != rep_scope.size()) {
    return false;
  }
  for (size_t i = 0; i < curr_scope.size(); ++i) {
    if (curr_scope[i].instruction_name != rep_scope[i].instruction_name) {
      return false;
    }
    int64_t curr_iter = curr_scope[i].iteration_index;
    int64_t rep_iter = rep_scope[i].iteration_index;
    if (curr_iter != rep_iter) {
      if (curr_iter == -1 || rep_iter == -1) {
        continue;
      }
      return false;
    }
  }
  return true;
}

bool IsCallInstruction(HloOpcode opcode) {
  return opcode == HloOpcode::kCall || opcode == HloOpcode::kWhile ||
         opcode == HloOpcode::kConditional;
}

struct ExecutionDagNode {
  std::vector<ScopeInstruction> scope;
  LocalGraphNode node;

  template <typename H>
  friend H AbslHashValue(H h, const ExecutionDagNode& n) {
    h = H::combine(std::move(h), n.node);
    for (const auto& s : n.scope) {
      h = H::combine(std::move(h), s.instruction_name, s.iteration_index);
    }
    return h;
  }

  bool operator==(const ExecutionDagNode& other) const {
    if (!(node == other.node)) {
      return false;
    }
    if (scope.size() != other.scope.size()) {
      return false;
    }
    for (size_t i = 0; i < scope.size(); ++i) {
      if (scope[i].instruction_name != other.scope[i].instruction_name ||
          scope[i].iteration_index != other.scope[i].iteration_index) {
        return false;
      }
    }
    return true;
  }
};

}  // namespace

std::vector<AbsoluteScopedTensorKey> FindConsumers(
    const AbsoluteScopedTensorKey& X,
    const ComputationDagCollection& dag_collection) {
  std::queue<ExecutionDagNode> q;
  LocalGraphNode start_node;
  start_node.instruction_name = X.tensor_key.instruction_name;
  start_node.operand_index = -1;
  start_node.shape_index.assign(X.tensor_key.shape_index.begin(),
                                X.tensor_key.shape_index.end());

  ExecutionDagNode start_state = {X.scope_instructions, start_node};

  q.push(start_state);

  std::vector<AbsoluteScopedTensorKey> results;
  absl::flat_hash_set<ExecutionDagNode> visited;
  visited.insert(start_state);

  const auto& max_rep_iter_map = dag_collection.max_rep_iter_map;

  auto add_result = [&](const AbsoluteScopedTensorKey& res) {
    results.push_back(res);
  };

  while (!q.empty()) {
    ExecutionDagNode curr = q.front();
    q.pop();

    bool is_start = (curr.node == start_node &&
                     curr.scope.size() == X.scope_instructions.size());
    if (is_start) {
      for (size_t i = 0; i < curr.scope.size(); ++i) {
        if (curr.scope[i].instruction_name !=
                X.scope_instructions[i].instruction_name ||
            curr.scope[i].iteration_index !=
                X.scope_instructions[i].iteration_index) {
          is_start = false;
          break;
        }
      }
    }

    if (!is_start) {
      auto it = dag_collection.reported_map.find(curr.node);
      if (it != dag_collection.reported_map.end()) {
        bool found_match = false;
        for (const auto& rep : it->second) {
          if (MatchScopesConsumers(curr.scope, rep.scope_instructions)) {
            if (!(rep.tensor_key == X.tensor_key &&
                  MatchScopesConsumers(rep.scope_instructions,
                                       X.scope_instructions))) {
              add_result(rep);
            }
            found_match = true;
          }
        }
        if (found_match) {
          continue;  // Stop exploring this branch
        }
      }
    }

    auto info_it = dag_collection.inst_info.find(curr.node.instruction_name);
    if (info_it == dag_collection.inst_info.end()) {
      continue;
    }
    const auto& info = info_it->second;
    const std::string& comp_name = info.parent_computation;

    auto dag_it = dag_collection.graphs.find(comp_name);
    if (dag_it == dag_collection.graphs.end()) {
      continue;
    }
    const auto& simplified_graph = dag_it->second;

    // 1. Local Consumers
    auto con_it = simplified_graph.consumers.find(curr.node);
    if (con_it != simplified_graph.consumers.end()) {
      for (const auto& v : con_it->second) {
        if (v.operand_index >= 0) {
          // Call Input! Enter!
          auto call_info_it = dag_collection.inst_info.find(v.instruction_name);
          if (call_info_it == dag_collection.inst_info.end()) {
            continue;
          }
          const auto& call_info = call_info_it->second;

          std::vector<std::string> called_comp_names =
              call_info.called_computations;

          for (const auto& called_comp_name : called_comp_names) {
            auto comp_info_it = dag_collection.comp_info.find(called_comp_name);
            if (comp_info_it == dag_collection.comp_info.end()) {
              continue;  // Should not happen in well-formed DAG
            }
            const auto& called_comp_info = comp_info_it->second;

            if (v.operand_index >= called_comp_info.parameters.size()) {
              LOG(ERROR) << "Strict alignment failure: operand index "
                         << v.operand_index << " out of bounds for "
                         << called_comp_name;
              continue;
            }
            const std::string& param_name =
                called_comp_info.parameters[v.operand_index];

            ScopeInstruction s = ScopeInstruction::Create(
                v.instruction_name,
                call_info.opcode == HloOpcode::kWhile ? 0 : 0);

            ExecutionDagNode next_curr = curr;
            next_curr.scope.push_back(s);
            next_curr.node = {param_name, -1, v.shape_index};

            if (visited.insert(next_curr).second) {
              q.push(next_curr);
            }
          }
        } else {
          // Normal tensor
          ExecutionDagNode next_curr = curr;
          next_curr.node = v;
          if (visited.insert(next_curr).second) {
            q.push(next_curr);
          }
        }
      }
    }

    // 2. Exit to Parent (if Root)
    auto comp_info_it = dag_collection.comp_info.find(comp_name);
    if (comp_info_it != dag_collection.comp_info.end()) {
      const auto& comp_info = comp_info_it->second;
      if (curr.node.instruction_name == comp_info.root_name) {
        // Loop back logic for kWhile
        if (!curr.scope.empty()) {
          ScopeInstruction s = curr.scope.back();
          auto parent_info_it =
              dag_collection.inst_info.find(s.instruction_name);
          if (parent_info_it != dag_collection.inst_info.end() &&
              parent_info_it->second.opcode == HloOpcode::kWhile) {
            // If we are at the root of the BODY.
            if (!parent_info_it->second.called_computations.empty() &&
                comp_name == parent_info_it->second.called_computations[0]) {
              for (const auto& called_comp_name :
                   parent_info_it->second.called_computations) {
                auto called_comp_info_it =
                    dag_collection.comp_info.find(called_comp_name);
                if (called_comp_info_it != dag_collection.comp_info.end()) {
                  const auto& called_comp_info = called_comp_info_it->second;
                  int64_t bound = 1;
                  auto bound_it = max_rep_iter_map.find(s.instruction_name);
                  if (bound_it != max_rep_iter_map.end() &&
                      bound_it->second >= 0) {
                    bound = bound_it->second + 1;
                  }

                  if (s.iteration_index == -1 || s.iteration_index <= bound) {
                    ExecutionDagNode next_curr = curr;  // Keep scope!
                    if (next_curr.scope.back().iteration_index >= 0) {
                      next_curr.scope.back().iteration_index++;
                    }
                    if (!called_comp_info.parameters.empty()) {
                      next_curr.node = {called_comp_info.parameters[0], -1,
                                        curr.node.shape_index};
                      if (visited.insert(next_curr).second) {
                        q.push(next_curr);
                      }
                    }
                  }
                }
              }
            }
          }
        }
        if (!curr.scope.empty()) {
          ScopeInstruction s = curr.scope.back();
          auto parent_info_it =
              dag_collection.inst_info.find(s.instruction_name);
          bool should_exit = true;
          if (parent_info_it != dag_collection.inst_info.end() &&
              parent_info_it->second.opcode == HloOpcode::kWhile) {
            // Only exit to parent from BODY, not CONDITION.
            if (parent_info_it->second.called_computations.empty() ||
                comp_name != parent_info_it->second.called_computations[0]) {
              should_exit = false;
            } else {
              int64_t max_iter = 0;
              auto max_iter_it = max_rep_iter_map.find(s.instruction_name);
              if (max_iter_it != max_rep_iter_map.end()) {
                max_iter = max_iter_it->second;
              }
              if (s.iteration_index != -1 && s.iteration_index < max_iter) {
                should_exit = false;
              }
            }
          }

          if (should_exit) {
            ExecutionDagNode next_curr = curr;
            next_curr.scope.pop_back();
            next_curr.node = {s.instruction_name, -1, curr.node.shape_index};
            if (visited.insert(next_curr).second) {
              q.push(next_curr);
            }
          }
        }
      }
    }
  }
  return results;
}

std::vector<AbsoluteScopedTensorKey> FindSuppliers(
    const AbsoluteScopedTensorKey& X,
    const ComputationDagCollection& dag_collection) {
  std::queue<ExecutionDagNode> q;
  LocalGraphNode start_node;
  start_node.instruction_name = X.tensor_key.instruction_name;
  start_node.operand_index = -1;
  start_node.shape_index.assign(X.tensor_key.shape_index.begin(),
                                X.tensor_key.shape_index.end());

  ExecutionDagNode start_state = {X.scope_instructions, start_node};

  q.push(start_state);

  std::vector<AbsoluteScopedTensorKey> results;
  absl::flat_hash_set<ExecutionDagNode> visited;
  visited.insert(start_state);

  const auto& max_rep_iter_map = dag_collection.max_rep_iter_map;

  auto add_result = [&](const AbsoluteScopedTensorKey& res) {
    results.push_back(res);
  };

  while (!q.empty()) {
    ExecutionDagNode curr = q.front();
    q.pop();

    bool is_start = (curr.node == start_node &&
                     curr.scope.size() == X.scope_instructions.size());
    if (is_start) {
      for (size_t i = 0; i < curr.scope.size(); ++i) {
        if (curr.scope[i].instruction_name !=
                X.scope_instructions[i].instruction_name ||
            curr.scope[i].iteration_index !=
                X.scope_instructions[i].iteration_index) {
          is_start = false;
          break;
        }
      }
    }

    if (!is_start) {
      auto it = dag_collection.reported_map.find(curr.node);
      if (it != dag_collection.reported_map.end()) {
        bool found_match = false;
        for (const auto& rep : it->second) {
          if (MatchScopesSuppliers(curr.scope, rep.scope_instructions)) {
            if (!(rep.tensor_key == X.tensor_key &&
                  MatchScopesSuppliers(rep.scope_instructions,
                                       X.scope_instructions))) {
              add_result(rep);
            }
            found_match = true;
          }
        }
        if (found_match) {
          continue;  // Stop exploring this branch
        }
      }
    }

    auto info_it = dag_collection.inst_info.find(curr.node.instruction_name);
    if (info_it == dag_collection.inst_info.end()) {
      continue;
    }
    const auto& info = info_it->second;
    const std::string& comp_name = info.parent_computation;

    // 1. Handle Call Output (Enter Call Backward)
    if (curr.node.operand_index == -1 && IsCallInstruction(info.opcode)) {
      std::vector<std::string> called_comp_names;
      if (info.opcode == HloOpcode::kWhile) {
        if (!info.called_computations.empty()) {
          called_comp_names.push_back(
              info.called_computations[0]);  // Only body
        }
      } else {
        called_comp_names = info.called_computations;
      }

      for (const auto& called_comp_name : called_comp_names) {
        auto comp_info_it = dag_collection.comp_info.find(called_comp_name);
        if (comp_info_it == dag_collection.comp_info.end()) {
          continue;
        }
        const auto& called_comp_info = comp_info_it->second;
        const std::string& root_name = called_comp_info.root_name;

        int64_t iter_index = 0;
        if (info.opcode == HloOpcode::kWhile) {
          auto max_iter_it = max_rep_iter_map.find(curr.node.instruction_name);
          if (max_iter_it != max_rep_iter_map.end()) {
            iter_index = max_iter_it->second;
          }
        }

        ScopeInstruction s =
            ScopeInstruction::Create(curr.node.instruction_name, iter_index);

        ExecutionDagNode next_curr = curr;
        next_curr.scope.push_back(s);
        next_curr.node = {root_name, -1, curr.node.shape_index};

        if (visited.insert(next_curr).second) {
          q.push(next_curr);
        }
      }
      continue;  // Call Output is handled by entering, don't follow local
                 // suppliers (it shouldn't have any anyway)
    }

    // 2. Exit to Parent (if Parameter)
    if (info.opcode == HloOpcode::kParameter) {
      // Loop back logic for kWhile
      if (!curr.scope.empty()) {
        ScopeInstruction s = curr.scope.back();
        auto parent_info_it = dag_collection.inst_info.find(s.instruction_name);
        if (parent_info_it != dag_collection.inst_info.end() &&
            parent_info_it->second.opcode == HloOpcode::kWhile) {
          if (s.iteration_index > 0 || s.iteration_index == -1) {
            if (!parent_info_it->second.called_computations.empty()) {
              const std::string& body_comp_name =
                  parent_info_it->second.called_computations[0];
              auto body_comp_info_it =
                  dag_collection.comp_info.find(body_comp_name);
              if (body_comp_info_it != dag_collection.comp_info.end()) {
                const auto& body_comp_info = body_comp_info_it->second;
                ExecutionDagNode next_curr = curr;  // Keep scope!
                if (next_curr.scope.back().iteration_index > 0) {
                  next_curr.scope.back().iteration_index--;
                }
                next_curr.node = {body_comp_info.root_name, -1,
                                  curr.node.shape_index};
                if (visited.insert(next_curr).second) {
                  q.push(next_curr);
                }
              }
            }
          }
        }
      }

      bool should_exit = true;
      if (!curr.scope.empty()) {
        ScopeInstruction s = curr.scope.back();
        auto parent_info_it = dag_collection.inst_info.find(s.instruction_name);
        if (parent_info_it != dag_collection.inst_info.end() &&
            parent_info_it->second.opcode == HloOpcode::kWhile) {
          if (s.iteration_index > 0) {
            should_exit = false;
          }
        }
      }

      if (should_exit && !curr.scope.empty()) {
        ScopeInstruction s = curr.scope.back();
        ExecutionDagNode next_curr = curr;
        next_curr.scope.pop_back();

        int param_no = info.parameter_number;
        LocalGraphNode call_input = {s.instruction_name, param_no,
                                     curr.node.shape_index};

        auto parent_info_it = dag_collection.inst_info.find(s.instruction_name);
        if (parent_info_it != dag_collection.inst_info.end()) {
          const std::string& parent_comp_name =
              parent_info_it->second.parent_computation;
          auto parent_dag_it = dag_collection.graphs.find(parent_comp_name);
          if (parent_dag_it != dag_collection.graphs.end()) {
            auto sup_it = parent_dag_it->second.suppliers.find(call_input);
            if (sup_it != parent_dag_it->second.suppliers.end()) {
              for (const auto& v : sup_it->second) {
                ExecutionDagNode next_next_curr = next_curr;
                next_next_curr.node = v;
                if (visited.insert(next_next_curr).second) {
                  q.push(next_next_curr);
                }
              }
            }
          }
        }
      }
      continue;  // Parameter is handled by exiting, don't follow local
                 // suppliers (it has none)
    }

    // 3. Local Suppliers
    auto dag_it = dag_collection.graphs.find(comp_name);
    if (dag_it == dag_collection.graphs.end()) {
      continue;
    }
    const auto& simplified_graph = dag_it->second;

    auto sup_it = simplified_graph.suppliers.find(curr.node);
    if (sup_it != simplified_graph.suppliers.end()) {
      for (const auto& v : sup_it->second) {
        ExecutionDagNode next_curr = curr;
        next_curr.node = v;
        if (visited.insert(next_curr).second) {
          q.push(next_curr);
        }
      }
    }
  }

  return results;
}

}  // namespace xla::numerics::comparison
