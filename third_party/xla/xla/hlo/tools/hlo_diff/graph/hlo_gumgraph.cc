// Copyright 2025 The OpenXLA Authors
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "xla/hlo/tools/hlo_diff/graph/hlo_gumgraph.h"

#include <algorithm>
#include <cstdint>
#include <memory>
#include <utility>
#include <vector>

#include "absl/base/nullability.h"
#include "absl/container/flat_hash_map.h"
#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/memory/memory.h"
#include "absl/status/status.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/hlo/ir/hlo_print_options.h"
#include "xla/hlo/tools/hlo_diff/graph/analysis/hlo_value_tracing.h"
#include "xla/hlo/tools/hlo_diff/graph/hlo_gumgraph_node.h"
#include "xla/hlo/tools/hlo_diff/graph/utils/hlo_gumgraph_dfs.h"
#include "xla/hlo/tools/hlo_diff/utils/hlo_diff_util.h"
#include "xla/service/call_graph.h"
#include "xla/tsl/platform/errors.h"
#include "xla/tsl/platform/statusor.h"
#include "tsl/platform/fingerprint.h"

namespace xla {
namespace hlo_diff {
namespace {

// Adds an edge between the given parent and child nodes.
void AddEdge(HloInstructionNode* parent, HloInstructionNode* child) {
  parent->children.push_back(child);
  child->parents.push_back(parent);
}

// Creates HloPrintOptions from the given fingerprint options.
HloPrintOptions CreateHloPrintOptions(
    const HloGumgraphFingerprintOptions& fingerprint_options) {
  HloPrintOptions hlo_print_options =
      HloPrintOptions::Fingerprint()
          .set_include_layout_in_shapes(false)
          .set_print_subcomputation_mode(
              HloPrintOptions::PrintSubcomputationMode::kOff)
          .set_print_parameter_number(false);
  if (fingerprint_options.ignore_shape) {
    hlo_print_options.set_print_operand_shape(false);
    hlo_print_options.set_print_result_shape(false);
  }
  return hlo_print_options;
}

}  // namespace

std::pair<HloInstructionNode*, bool> HloGumgraph::AddNode(
    const HloInstruction& instruction, int unique_node_index) {
  auto node = std::make_unique<HloInstructionNode>(HloInstructionNode{
      .instruction = &instruction, .unique_node_index = unique_node_index});
  auto [new_node_it, inserted] =
      instruction_to_node_.try_emplace(&instruction, std::move(node));
  return {new_node_it->second.get(), inserted};
}

absl::Status HloGumgraph::ConstructGraph(const HloModule& hlo_module) {
  LOG(INFO) << "Constructing HloGumgraph";
  int unique_instruction_index = 0;
  for (auto* computation : hlo_module.MakeComputationPostOrder()) {
    for (auto* instruction : computation->MakeInstructionPostOrder()) {
      std::pair<HloInstructionNode*, bool> node_and_inserted =
          AddNode(*instruction, ++unique_instruction_index);
      if (!node_and_inserted.second) {
        return absl::InternalError(absl::StrCat(
            "Instruction: ", instruction->name(), " already in the graph"));
      }

      HloInstructionNode* node = node_and_inserted.first;
      node->props.fingerprint = GetHloInstructionFingerprint(
          instruction, CreateHloPrintOptions(fingerprint_options_));

      switch (instruction->opcode()) {
        case HloOpcode::kCall:
        case HloOpcode::kFusion:
        case HloOpcode::kWhile: {
          // Connect Call, Fusion and While instruction's called computations
          // parameters with the operands of the caller instructions to inline
          // the called computation as they should match 1:1.
          for (auto* called_computation : instruction->called_computations()) {
            for (int i = 0; i < instruction->operands().size(); ++i) {
              HloInstructionNode* parent =
                  GetNode(called_computation->parameter_instruction(i));
              HloInstructionNode* child = GetNode(instruction->operands()[i]);
              if (parent == nullptr || child == nullptr) {
                return absl::InternalError(absl::StrFormat(
                    "Called computation instruction (%s) operand not found "
                    "in the called computation: %s parameters",
                    child->GetName(), parent->GetName()));
              }
              AddEdge(parent, child);
            }
          }
          break;
        }
        case HloOpcode::kConditional: {
          // Connect conditional instruction node with the predicate operand.
          HloInstructionNode* pred_node = GetNode(instruction->operands()[0]);
          if (pred_node == nullptr) {
            return absl::InternalError(absl::StrFormat(
                "Instruction (%s) operand: %s not found in the graph",
                instruction->name(), instruction->operands()[0]->name()));
          }
          AddEdge(node, pred_node);

          // Connect conditional instruction's branch computations parameters
          // with the operands of the caller instructions to inline the branch
          // computations.
          for (int i = 0; i < instruction->branch_count(); ++i) {
            HloComputation* branch_computation =
                instruction->branch_computation(i);
            HloInstructionNode* parent =
                GetNode(branch_computation->parameter_instruction(0));
            HloInstructionNode* child = GetNode(instruction->operands()[i + 1]);
            if (parent == nullptr || child == nullptr) {
              return absl::InternalError(absl::StrFormat(
                  "Branch computation instruction (%s) operand not found "
                  "in the branch computation: %s parameters",
                  child->GetName(), parent->GetName()));
            }
            AddEdge(parent, child);
          }
          break;
        }
        default: {
          for (auto* operand : instruction->operands()) {
            HloInstructionNode* child = GetNode(operand);
            if (child == nullptr) {
              return absl::InternalError(absl::StrFormat(
                  "Instruction (%s) operand: %s not found in the graph",
                  instruction->name(), operand->name()));
            }
            AddEdge(node, child);
          }
        }
      }

      // Connect the root instruction of the called computation with the
      // caller instruction.
      for (auto* called_computation : instruction->called_computations()) {
        HloInstructionNode* called_computation_root_node =
            GetNode(called_computation->root_instruction());
        if (called_computation_root_node == nullptr) {
          return absl::InternalError(absl::StrFormat(
              "Called computation (%s) root: %s not found in the graph",
              called_computation->name(),
              called_computation->root_instruction()->name()));
        }
        AddEdge(node, called_computation_root_node);
      }
    }
  }

  return absl::OkStatus();
}

absl::StatusOr<std::vector<HloInstructionNode*>>
HloGumgraph::PrecomputeGenerations() {
  LOG(INFO) << "Precomputing generations";
  std::vector<HloInstructionNode*> zero_indegrees;
  absl::flat_hash_map<const HloInstructionNode*, int> indegrees;
  for (const auto& [_, node] : instruction_to_node_) {
    if (node->parents.empty()) {
      zero_indegrees.push_back(node.get());
      continue;
    }

    auto [it, inserted] = indegrees.insert({node.get(), node->parents.size()});
    if (!inserted) {
      return absl::InternalError(
          absl::StrCat("Instruction: ", node->instruction->name(),
                       " already inserted in indegree map"));
    }
    indegrees[node.get()] = node->parents.size();
  }
  std::vector<HloInstructionNode*> init_zero_indegrees = zero_indegrees;
  nodes_by_generation_.push_back({&root_});

  int current_generation = 1;
  while (!zero_indegrees.empty()) {
    std::vector<HloInstructionNode*> current_generation_nodes =
        std::move(zero_indegrees);
    zero_indegrees = {};

    for (int i = 0; i < current_generation_nodes.size(); ++i) {
      current_generation_nodes[i]->props.generation = current_generation;
      current_generation_nodes[i]->props.sibling_position = {
          i, static_cast<int64_t>(current_generation_nodes.size())};
      for (HloInstructionNode* child : current_generation_nodes[i]->children) {
        auto it = indegrees.find(child);
        if (it == indegrees.end()) {
          return absl::InternalError(
              absl::StrCat("Instruction: ", child->instruction->name(),
                           " not found in indegree map"));
        }
        --it->second;
        if (it->second == 0) {
          zero_indegrees.push_back(child);
          indegrees.erase(it);
        }
      }
    }
    nodes_by_generation_.push_back(std::move(current_generation_nodes));
    ++current_generation;
  }

  if (!indegrees.empty()) {
    LOG(WARNING) << "Cycle detected in the graph.";
    return absl::InternalError("Cycle detected in the graph");
  }
  return init_zero_indegrees;
}

void HloGumgraph::PrecomputeSizeAndHeight() {
  LOG(INFO) << "Precomputing size and height";
  // TODO(camillesun): Refactor this to use DFS.
  for (auto it = nodes_by_generation_.rbegin();
       it != nodes_by_generation_.rend(); ++it) {
    for (HloInstructionNode* node : *it) {
      int64_t height = 0;
      uint64_t fingerprint = node->props.fingerprint;

      for (const HloInstructionNode* child : node->children) {
        height = std::max(height, child->props.height);
        fingerprint = tsl::FingerprintCat64(fingerprint,
                                            child->props.subgraph_fingerprint);
      }

      node->props.height = height + 1;
      // TODO(b/365855856): graph with different structure can share a same
      // subgraph fingerprint, see test case
      // PreComputationsWorksSubgraphFingerprint. This is unexpected.
      node->props.subgraph_fingerprint = fingerprint;
    }
  }
}

absl::Status HloGumgraph::PrecomputeComputationFingerprint() {
  LOG(INFO) << "Precomputing computation fingerprint";
  TF_RETURN_IF_ERROR(call_graph_->VisitNodes([&](const CallGraphNode& node)
                                                 -> absl::Status {
    absl::flat_hash_map<const HloInstruction*, uint64_t> subgraph_fingerprint;
    const HloComputation* computation = node.computation();
    for (auto* instruction : computation->MakeInstructionPostOrder()) {
      uint64_t fp = GetNode(instruction)->props.fingerprint;
      for (const HloInstruction* operand : instruction->operands()) {
        fp = tsl::FingerprintCat64(
            fp, subgraph_fingerprint.at(GetNode(operand)->instruction));
      }
      subgraph_fingerprint[instruction] = fp;
    }

    computation_to_props_[computation] = CallGraphNodeProps{
        .call_graph_node = &node,
        .fingerprint =
            subgraph_fingerprint.at(computation->root_instruction())};

    return absl::OkStatus();
  }));
  return absl::OkStatus();
}

void HloGumgraph::PrecomputeDfsPosition() {
  LOG(INFO) << "Precomputing DFS position";
  std::vector<const HloInstructionNode*> pre_order_nodes =
      GetAllNodesInDfsOrder(root_, DfsTraversalOrder::kPreOrder,
                            GetNodeCount());
  for (int i = 0; i < pre_order_nodes.size(); ++i) {
    if (pre_order_nodes[i]->is_root) {
      continue;
    }
    instruction_to_node_[pre_order_nodes[i]->instruction]
        ->props.pre_order_graph_position = {
        i, static_cast<int64_t>(pre_order_nodes.size())};
  }
}

absl::StatusOr<std::unique_ptr<const HloGumgraph>> HloGumgraph::Create(
    absl::Nonnull<const HloModule*> hlo_module,
    const HloGumgraphFingerprintOptions& fingerprint_options) {
  CHECK(hlo_module != nullptr) << "Expected a non-null hlo module";
  CHECK(hlo_module->entry_computation() != nullptr)
      << "Expected a non-null entry computation";

  std::unique_ptr<CallGraph> call_graph = CallGraph::Build(hlo_module);
  TF_ASSIGN_OR_RETURN(std::unique_ptr<HloValueTracing> hlo_value_tracing,
                      HloValueTracing::Run(*hlo_module));
  auto graph = absl::WrapUnique(
      new HloGumgraph(*hlo_module, fingerprint_options, std::move(call_graph),
                      std::move(hlo_value_tracing)));

  TF_RETURN_IF_ERROR(graph->ConstructGraph(*hlo_module));
  TF_ASSIGN_OR_RETURN(std::vector<HloInstructionNode*> zero_indegree_nodes,
                      graph->PrecomputeGenerations());
  for (auto* zero_indegree_node : zero_indegree_nodes) {
    AddEdge(&graph->root_, zero_indegree_node);
  }
  graph->PrecomputeSizeAndHeight();
  TF_RETURN_IF_ERROR(graph->PrecomputeComputationFingerprint());
  graph->PrecomputeDfsPosition();

  return graph;
};

}  // namespace hlo_diff
}  // namespace xla
