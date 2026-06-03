// Copyright 2026 The OpenXLA Authors.
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

#ifndef XLA_TOOLS_DEBUG_INFO_HLO_DUMP_HLO_DUMP_UTILS_H_
#define XLA_TOOLS_DEBUG_INFO_HLO_DUMP_HLO_DUMP_UTILS_H_

#include <cstdint>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/service/hlo.pb.h"
#include "xla/shape_util.h"

namespace xla::tools::debug_info {

struct MismatchDetails {
  std::string target_instruction_name;
  std::optional<int64_t> output_shape_index;
  double actual = 0.0;
  double expected = 0.0;
  double rel_error = 0.0;
  std::optional<double> percentage_of_elems_exceeding_abs_error;
  std::optional<double> percentage_of_elems_exceeding_rel_error;
  std::optional<double> percentage_of_elems_exceeding_both_errors;
  std::optional<bool> result_of_reduce;
  std::optional<std::string> custom_description;
};

struct TensorKey {
  std::string instruction_name;
  xla::ShapeIndex shape_index;

  bool operator==(const TensorKey& other) const {
    return instruction_name == other.instruction_name &&
           shape_index == other.shape_index;
  }

  bool operator<(const TensorKey& other) const {
    if (instruction_name != other.instruction_name) {
      return instruction_name < other.instruction_name;
    }
    return shape_index < other.shape_index;
  }

  template <typename H>
  friend H AbslHashValue(H h, const TensorKey& k) {
    return H::combine(std::move(h), k.instruction_name, k.shape_index);
  }

  static TensorKey Create(absl::string_view instruction_name,
                          xla::ShapeIndex shape_index = {}) {
    return TensorKey{std::string(instruction_name), std::move(shape_index)};
  }
};

struct TensorAnnotation {
  std::optional<std::string> background_color;
  std::optional<std::string> border_style;
  std::optional<std::string> tooltip_data;
  std::optional<std::string> anchor_id;
  std::optional<int32_t> stack_frame_id;
};

struct OriginalValueRecoveryInfo {
  std::optional<double> percentage_recoverable;
  std::optional<double> percentage_recovered;
  absl::Span<const std::pair<std::string, int64_t>> histogram;
};

struct GraphNode {
  int64_t id;
  double x;
  double y;
  double diff_score;
  std::string key;
  int64_t anchor_id;
};

struct GraphEdge {
  int64_t supplier_id;
  int64_t consumer_id;
};

struct GraphData {
  std::vector<GraphNode> nodes;
  std::vector<GraphEdge> edges;
};

// Converts HLO text to HTML with syntax highlighting and annotations.
// Mirroring the Python API in utils.py.
std::string ConvertHloToHtml(
    absl::string_view dump_name, absl::string_view hlo_text,
    const absl::flat_hash_map<TensorKey, TensorAnnotation>& annotations,
    OriginalValueRecoveryInfo recovery_info = {},
    const xla::StackFrameIndexProto* stack_frame_index = nullptr,
    const GraphData* graph_data = nullptr);

absl::flat_hash_map<TensorKey, TensorAnnotation> PopulateMismatchAnnotations(
    const HloModule& module, absl::Span<const MismatchDetails> mismatches);

GraphData PopulateMismatchGraphData(
    const HloModule& module, absl::Span<const MismatchDetails> mismatches);

// Generates an interactive HTML dump for an HLO module
// and writes it to the test undeclared outputs directory (or temp dir).
// Returns the absolute path of the written file.
absl::StatusOr<std::string> DumpHloModuleMismatchWithGraphData(
    const HloModule& module, absl::Span<const MismatchDetails> mismatches,
    absl::string_view output_filename);

}  // namespace xla::tools::debug_info

#endif  // XLA_TOOLS_DEBUG_INFO_HLO_DUMP_HLO_DUMP_UTILS_H_
