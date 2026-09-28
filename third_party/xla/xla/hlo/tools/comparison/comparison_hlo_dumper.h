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

#ifndef XLA_HLO_TOOLS_COMPARISON_COMPARISON_HLO_DUMPER_H_
#define XLA_HLO_TOOLS_COMPARISON_COMPARISON_HLO_DUMPER_H_

#include <cstdint>
#include <limits>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/status/status.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/hlo/tools/comparison/comparison_result.pb.h"
#include "xla/hlo/tools/comparison/original_tensor_summary_utils.h"
#include "xla/hlo/tools/comparison/tensor_summary_util.h"
#include "xla/service/hlo.pb.h"
#include "xla/tools/hlo_dump/hlo_dump_utils.h"

namespace xla::numerics::comparison {

struct HloNodeComparisonStats {
  int64_t score_count = 0;
  double score_min = std::numeric_limits<double>::infinity();
  double score_max = -std::numeric_limits<double>::infinity();
  double score_mean = 0.0;
  bool not_comparable = false;
};

struct HloHtmlTensorKey {
  std::string name;
  std::vector<int> shape_index;

  template <typename H>
  friend H AbslHashValue(H h, const HloHtmlTensorKey& k) {
    return H::combine(std::move(h), k.name, k.shape_index);
  }

  bool operator==(const HloHtmlTensorKey& other) const {
    return name == other.name && shape_index == other.shape_index;
  }
};

struct TensorAnnotation {
  std::optional<std::string> background_color;
  std::optional<std::string> border_style;
  std::optional<std::string> tooltip_data;
  std::optional<std::string> anchor_id;
  std::optional<int32_t> stack_frame_id;
};

// Generates an HLO HTML dump for the given HLO text and annotations.
// This function will initialize the Python interpreter if it hasn't been
// already.
absl::Status DumpHloToHtml(
    const std::string& dump_name, absl::string_view hlo_text,
    const absl::flat_hash_map<HloHtmlTensorKey, TensorAnnotation>& annotations,
    const std::string& output_path,
    std::optional<float> percentage_recoverable = std::nullopt,
    std::optional<float> percentage_recovered = std::nullopt,
    const xla::StackFrameIndexProto* stack_frame_index = nullptr,
    const debug_info::GraphData* graph_data = nullptr);

using FloatBlockSummary = ::xla::comparison::FloatBlockSummary;

FloatBlockSummary CombineSummaries(
    const std::vector<TensorSummaryProto>& summaries);

std::string GetTooltipData(const HloNodeComparisonStats* comp,
                           const FloatBlockSummary* baseline,
                           const FloatBlockSummary* target);

void GenerateHloHtmlDumps(int replica_id, const HloModule& baseline_module,
                          const HloModule& target_module,
                          absl::string_view output_dir,
                          absl::string_view comparison_results_path);

void GenerateSingleHloHtmlDump(int replica_id, const HloModule& module,
                               absl::string_view output_dir,
                               absl::string_view recovered_summaries_path);

struct LocalGraphNode {
  std::string instruction_name;
  // If >= 0, this is a Call Input node (k-th operand).
  // If == -1, this is an Output node.
  int operand_index = -1;
  std::vector<int64_t> shape_index;

  bool operator==(const LocalGraphNode& other) const {
    return instruction_name == other.instruction_name &&
           operand_index == other.operand_index &&
           shape_index == other.shape_index;
  }

  template <typename H>
  friend H AbslHashValue(H h, const LocalGraphNode& n) {
    return H::combine(std::move(h), n.instruction_name, n.operand_index,
                      n.shape_index);
  }
};

struct SimplifiedGraph {
  absl::flat_hash_map<LocalGraphNode, std::vector<LocalGraphNode>> consumers;
  absl::flat_hash_map<LocalGraphNode, std::vector<LocalGraphNode>> suppliers;
};

struct InstructionInfo {
  std::string parent_computation;
  HloOpcode opcode;
  std::vector<std::string> called_computations;
  int parameter_number = -1;
  int module_order = 0;
};

struct ComputationInfo {
  std::vector<std::string> parameters;
  std::string root_name;
};

struct ComputationDagCollection {
  absl::flat_hash_map<std::string, SimplifiedGraph> graphs;
  absl::flat_hash_map<std::string, InstructionInfo> inst_info;
  absl::flat_hash_map<std::string, ComputationInfo> comp_info;
  absl::flat_hash_map<LocalGraphNode, std::vector<AbsoluteScopedTensorKey>>
      reported_map;
  absl::flat_hash_map<std::string, int64_t> max_rep_iter_map;
};

ComputationDagCollection GenerateComputationDagCollection(
    const HloModule& module,
    absl::Span<const AbsoluteScopedTensorKey> reported_keys);

std::vector<AbsoluteScopedTensorKey> FindConsumers(
    const AbsoluteScopedTensorKey& X,
    const ComputationDagCollection& dag_collection);

std::vector<AbsoluteScopedTensorKey> FindSuppliers(
    const AbsoluteScopedTensorKey& X,
    const ComputationDagCollection& dag_collection);

struct NodePosition {
  double x;
  double y;
};

absl::flat_hash_map<AbsoluteScopedTensorKey, NodePosition>
ComputeDagLayoutForTesting(
    const ComputationDagCollection& dag_collection,
    absl::Span<const AbsoluteScopedTensorKey> reported_keys,
    const absl::flat_hash_map<AbsoluteScopedTensorKey,
                              std::vector<AbsoluteScopedTensorKey>>&
        node_consumers,
    const absl::flat_hash_map<AbsoluteScopedTensorKey,
                              std::vector<AbsoluteScopedTensorKey>>&
        node_suppliers);

}  // namespace xla::numerics::comparison

#endif  // XLA_HLO_TOOLS_COMPARISON_COMPARISON_HLO_DUMPER_H_
