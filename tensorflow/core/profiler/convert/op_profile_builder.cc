/* Copyright 2022 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/core/profiler/convert/op_profile_builder.h"

#include <algorithm>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

// #include "perftools/accelerators/xprof/convert/device_type_utils.h"
#include "absl/container/flat_hash_map.h"
#include "absl/container/node_hash_map.h"
#include "absl/strings/ascii.h"
#include "absl/strings/str_cat.h"
#include "tensorflow/core/lib/gtl/top_n.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/profiler/convert/op_metrics_db_combiner.h"
#include "tensorflow/core/profiler/convert/op_metrics_to_record.h"
#include "tensorflow/core/profiler/convert/xla_op_utils.h"
#include "tensorflow/core/profiler/protobuf/op_metrics.pb.h"
#include "tensorflow/core/profiler/protobuf/op_profile.pb.h"
#include "tensorflow/core/profiler/utils/math_utils.h"
#include "tensorflow/core/profiler/utils/op_metrics_db_utils.h"

namespace tensorflow {
namespace profiler {
namespace {

using op_profile::Metrics;
using op_profile::Node;

// Fill symbol details into a node.
void PopulateSymbolNode(const OpMetrics& op_metrics, Node* node) {
  node->set_name(op_metrics.name());
  Node::XLAInstruction& xla = *node->mutable_xla();
  xla.set_expression(op_metrics.long_name());
  xla.set_category(op_metrics.category());
  xla.set_provenance(op_metrics.provenance());
  if (op_metrics.has_layout()) {
    for (const auto& dimension : op_metrics.layout().dimensions()) {
      auto* dim = xla.mutable_layout()->add_dimensions();
      dim->set_size(dimension.size());
      dim->set_alignment(dimension.alignment());
      dim->set_semantics(absl::AsciiStrToLower(
          LayoutDimensionSemantics_Name(dimension.semantics())));
    }
  }
  xla.set_computation_primitive_size(op_metrics.computation_primitive_size());
}

// Sort the children and only keep the top K children.
template <typename Cmp>
Node TopKChildren(const Node* root, int k, Cmp cmp) {
  tensorflow::gtl::TopN<const Node*, decltype(cmp)> top_n(k, cmp);
  for (const Node& node : root->children()) {
    top_n.push(&node);
  }
  Node output;
  std::unique_ptr<std::vector<const Node*>> extracted_nodes(top_n.Extract());
  for (const Node* node : *extracted_nodes) {
    *output.add_children() = *node;
  }
  return output;
}

// Copy symbol details into a deduplicated node from the top child node.
void CopySymbolDetailsToDeduplicatedNode(Node* top_child_node,
                                         Node* deduplicated_node) {
  deduplicated_node->set_name(
      absl::StrCat(top_child_node->name(), " and its duplicate(s)"));
  Node::XLAInstruction& xla = *deduplicated_node->mutable_xla();
  const Node::XLAInstruction& top_child_node_xla = top_child_node->xla();
  xla.set_expression(top_child_node_xla.expression());
  xla.set_category(top_child_node_xla.category());
  if (IsFusion(top_child_node_xla.category())) return;
  xla.set_provenance(top_child_node_xla.provenance());
  *xla.mutable_layout() = top_child_node_xla.layout();
}

void SortAndPruneChildren(int k, int level, Node* root) {
  // Set the total number of children before pruning.
  root->set_num_children(root->children_size());
  for (Node& node : *root->mutable_children()) {
    SortAndPruneChildren(k, level - 1, &node);
  }
  k = level > 0 ? root->children_size() : k;

  if (root->children_size() > 1) {
    if (root->has_xla() && IsFusion(root->xla().category())) {
      // Sort the children under fusion node by raw flops.
      *root->mutable_children() =
          TopKChildren(root, k, [](const Node* a, const Node* b) {
            return a->metrics().raw_flops() > b->metrics().raw_flops();
          }).children();
    } else {
      *root->mutable_children() =
          TopKChildren(root, k, [](const Node* a, const Node* b) {
            return a->metrics().time() > b->metrics().time();
          }).children();
    }
  }
}

// Finalize deduplicated nodes by copying symbol details from the top child
// node.
void FinalizeDeduplicatedNodes(bool by_program, Node* root) {
  if (by_program) {
    for (Node& program_node : *root->mutable_children()) {
      for (Node& category_node : *program_node.mutable_children()) {
        for (Node& deduplicated_node : *category_node.mutable_children()) {
          // Skip for non deduplicated nodes. Those nodes already have name set.
          if (!deduplicated_node.name().empty() ||
              deduplicated_node.children().empty())
            continue;
          CopySymbolDetailsToDeduplicatedNode(
              deduplicated_node.mutable_children(0), &deduplicated_node);
        }
      }
    }
  } else {
    for (Node& category_node : *root->mutable_children()) {
      for (Node& deduplicated_node : *category_node.mutable_children()) {
        // Skip for non deduplicated nodes. Those nodes already have name set.
        if (!deduplicated_node.name().empty() ||
            deduplicated_node.children().empty())
          continue;
        CopySymbolDetailsToDeduplicatedNode(
            deduplicated_node.mutable_children(0), &deduplicated_node);
      }
    }
  }
}

// Recursively find computation size for HLOs -- applied only for convolutions.
// This is only for convolutions, not other HLOs, categories or whole programs.
// TODO(b/243596435) Find a permanent fix to this problem.
int64_t GetComputationSize(Node node) {
  int64_t computation_size = 0;
  for (const auto& child : node.children()) {
    if (GetComputationSize(child) != 0) {
      computation_size = GetComputationSize(child);
    }
  }
  if (node.has_xla()) {
    if (node.xla().computation_primitive_size() > 0) {
      return node.xla().computation_primitive_size();
    } else {
      return computation_size;
    }
  }
  return 0;
}

// Fills op metrics into a node.
void PopulateOpMetricsNode(const OpMetrics& op_metrics,
                           double peak_gigaflops_per_second_per_core,
                           double peak_gibibytes_per_second_per_core,
                           uint64_t total_time_ps, Node* node) {
  DCHECK_EQ(ChildrenTimePs(op_metrics), 0);

  Metrics* metrics = node->mutable_metrics();
  // The UI computes flops_rate = raw_flops / raw_time
  // and memory_bandwidth = raw_bytes_accessed / raw_time. See:
  // https://github.com/tensorflow/profiler/blob/master/frontend/app/common/utils/utils.ts
  metrics->set_raw_time(op_metrics.time_ps());
  metrics->set_raw_flops(op_metrics.flops());
  metrics->set_raw_bytes_accessed(op_metrics.bytes_accessed());

  // "time" is the op or category fraction of total time.
  metrics->set_time(SafeDivide(op_metrics.time_ps(), total_time_ps));

  // Hack to approximate utilization for INT8/4 convolution HLOs:
  // Since MXU BW is 2x/4x for INT8/4, multiply peak BW by the factor detemrined
  // by the computation size
  if (GetComputationSize(*node) == 8) {
    peak_gigaflops_per_second_per_core *= 2;
  } else if (GetComputationSize(*node) == 4) {
    peak_gigaflops_per_second_per_core *= 4;
  }
  double flops_utilization = SafeDivide(GigaFlopsPerSecondPerCore(op_metrics),
                                        peak_gigaflops_per_second_per_core);
  // The UI expects flops_utilization = flops / time. See:
  // https://github.com/tensorflow/profiler/blob/master/frontend/app/common/utils/utils.ts
  metrics->set_flops(flops_utilization * metrics->time());

  // TODO(b/219984562): Use hierarchical roofline.
  double mem_bw_utilization = SafeDivide(GibiBytesPerSecondPerCore(op_metrics),
                                         peak_gibibytes_per_second_per_core);
  metrics->set_memory_bandwidth(mem_bw_utilization);
}

// Sets the total time on the root node metrics.
void SetTotalTime(uint64_t total_time_ps, Node* root) {
  Metrics* metrics = root->mutable_metrics();
  metrics->set_raw_time(total_time_ps);
  metrics->set_time(1.0);
}

// Recursively insert "fused instruction" nodes (with raw flops).
void InsertFusedInstructions(const OpMetrics& op_metrics, Node* node) {
  if (!op_metrics.has_children()) return;
  for (const auto& child : op_metrics.children().metrics_db()) {
    Node* new_node = node->add_children();
    PopulateSymbolNode(child, new_node);
    new_node->mutable_metrics()->set_raw_flops(child.flops());
    if (child.has_children()) {
      InsertFusedInstructions(child, new_node);
    }
  }
}

}  // namespace

std::string OpProfileBuilder::GenerateProgramName(uint64_t program_id) const {
  DCHECK(program_name_map_ != nullptr);
  auto iter = program_name_map_->find(program_id);
  if (iter == program_name_map_->end()) return "main";
  return HloModuleNameWithProgramId(iter->second, program_id);
}

Node* OpProfileBuilder::AddOpNode(const OpMetrics& op_metrics,
                                  Category* category, Node* deduplicated_node) {
  Node* leaf;
  if (deduplicated_node != nullptr) {
    leaf = deduplicated_node->add_children();
  } else if (category != nullptr) {
    leaf = category->node->add_children();
  } else {
    leaf = root_->add_children();
  }
  PopulateSymbolNode(op_metrics, leaf);
  InsertFusedInstructions(op_metrics, leaf);
  return leaf;
}

Node* OpProfileBuilder::LookupOrAddDeduplicatedNode(const OpMetrics& op_metrics,
                                                    Category* category) {
  Node*& deduplicated_node =
      category->deduplicated_nodes[op_metrics.deduplicated_name()];
  if (deduplicated_node == nullptr) {
    deduplicated_node = category->node->add_children();
  }
  return deduplicated_node;
}

OpProfileBuilder::Category* OpProfileBuilder::LookupOrAddCategoryNode(
    const OpMetrics& op_metrics, Program* program) {
  Category* category;
  Node* category_parent;
  if (program != nullptr) {
    category = &program->categories[op_metrics.category()];
    category_parent = program->node;
  } else {
    category = &category_map_[op_metrics.category()];
    category_parent = root_;
  }
  if (category->node == nullptr) {
    category->node = category_parent->add_children();
    category->node->set_name(op_metrics.category());
  }
  return category;
}

OpProfileBuilder::Program* OpProfileBuilder::LookupOrAddProgramNode(
    const OpMetrics& op_metrics) {
  uint64_t program_id = op_metrics.hlo_module_id();
  Program* program = &programs_map_[program_id];
  if (program->node == nullptr) {
    program->node = root_->add_children();
    program->node->set_name(GenerateProgramName(program_id));
  }
  return program;
}

void OpProfileBuilder::AddOp(const OpMetrics& op_metrics) {
  // Exclude ops with children ops to avoid double counting of flops, bytes and
  // time from children ops.
  if (ChildrenTimePs(op_metrics) > 0) return;

  // The path from the root to the leaf node:
  // e.g. by_program -> cluster_xx -> convolution -> convolution.1 and its
  // deduplicates -> convolution.1
  // We will aggregate the metrics of convolution.1 to all its parent nodes.
  std::vector<Node*> all_paths = {root_};

  if (IsIdleOp(op_metrics)) {
    Node* leaf = AddOpNode(op_metrics);
    all_paths.push_back(leaf);
  } else {
    Program* program = nullptr;
    if (options_.group_by_program) {
      program = LookupOrAddProgramNode(op_metrics);
      all_paths.push_back(program->node);
    }

    Category* category = LookupOrAddCategoryNode(op_metrics, program);
    all_paths.push_back(category->node);

    Node* deduplicated_node = nullptr;
    if (options_.group_by_deduplicated_name &&
        !op_metrics.deduplicated_name().empty()) {
      deduplicated_node = LookupOrAddDeduplicatedNode(op_metrics, category);
      all_paths.push_back(deduplicated_node);
    }

    Node* leaf = AddOpNode(op_metrics, category, deduplicated_node);
    all_paths.push_back(leaf);
  }

  for (auto* node : all_paths) {
    // Per program combiner does not need to update OpMetrics.num_cores
    CombineOpMetrics(op_metrics, &metrics_[node], /*update_num_cores=*/false);
  }
}

void OpProfileBuilder::Finalize(double peak_gigaflops_per_second_per_core,
                                double peak_gibibytes_per_second_per_core,
                                uint64_t total_time_ps) {
  for (const auto& [node, op_metrics] : metrics_) {
    PopulateOpMetricsNode(op_metrics, peak_gigaflops_per_second_per_core,
                          peak_gibibytes_per_second_per_core, total_time_ps,
                          node);
  }
  SetTotalTime(total_time_ps, root_);
  // If grouping by program, we build a two-level pruned tree: the first level
  // is per program and the second level is per category. Otherwise we build a
  // single-level per category pruned tree.
  int level = options_.group_by_program ? 2 : 1;
  SortAndPruneChildren(options_.children_per_node, level, root_);
  if (options_.group_by_deduplicated_name) {
    FinalizeDeduplicatedNodes(options_.group_by_program, root_);
  }
}

OpProfileBuilder::OpProfileBuilder(
    const OpProfileOptions& options,
    tensorflow::profiler::op_profile::Node* root,
    const tensorflow::protobuf::Map<uint64_t, std::string>* program_name_map)
    : options_(options), root_(root), program_name_map_(program_name_map) {
  CHECK(root != nullptr);
  DCHECK(!options_.group_by_program || program_name_map_ != nullptr);
  root->set_name(options_.group_by_program ? "by_program" : "by_category");
}
}  // namespace profiler
}  // namespace tensorflow
