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

#ifndef XLA_TOOLS_COMPARE_LITERALS_COMPARE_MODEL_LITERALS_H_
#define XLA_TOOLS_COMPARE_LITERALS_COMPARE_MODEL_LITERALS_H_

#include <cstdint>
#include <map>
#include <string>
#include <vector>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "xla/tools/compare_literals/compare_literals.h"

namespace xla::compare_literals {

// Key identifying a specific literal output on a device replica.
struct LiteralKey {
  int64_t literal_id = 0;
  int64_t device_id = 0;

  bool operator<(const LiteralKey& o) const {
    if (literal_id != o.literal_id) {
      return literal_id < o.literal_id;
    }
    return device_id < o.device_id;
  }
  bool operator==(const LiteralKey& o) const {
    return literal_id == o.literal_id && device_id == o.device_id;
  }
};

// Level 1: Numerical comparison and error statistics between two literals.
struct LiteralComparisonStats {
  absl::Status status = absl::OkStatus();
  int64_t exact_matches = 0;
  double exact_match_pct = 0.0;
  int64_t mismatches = 0;
  int64_t nan_mismatches = 0;
  int64_t inf_mismatches = 0;
  double max_abs_error = 0.0;
  double max_rel_error = 0.0;
  double mean_rel_error = 0.0;
  double suggested_abs_error = 0.0;
  double suggested_rel_error = 0.0;
};

// Level 2: Numerical profile for a model output literal, aggregated across
// devices.
struct OutputLiteralStats {
  int64_t literal_index = 0;
  std::string literal_name;
  std::string shape_str;
  std::string element_type;
  int64_t element_count = 0;

  int64_t num_devices = 0;
  int64_t failed_devices = 0;
  LiteralComparisonStats aggregated_device_stats;

  std::map<int64_t, LiteralComparisonStats> device_stats;
};

// Level 3: Summary metrics across all outputs in the model.
struct ModelSummaryStats {
  int64_t total_literals = 0;
  int64_t exact_match_literals = 0;
  int64_t within_tolerance_literals = 0;
  int64_t differing_literals = 0;
  int64_t failed_device_comparisons = 0;
  int64_t nan_inf_mismatch_literals = 0;
  double worst_abs_error = 0.0;
  int64_t worst_abs_literal = -1;
  double worst_rel_error = 0.0;
  int64_t worst_rel_literal = -1;
};

// Result of comparing all literals across directories.
struct ModelComparisonResult {
  std::string golden_dir;
  std::string test_dir;
  std::vector<int64_t> devices;
  ModelSummaryStats summary;
  std::vector<OutputLiteralStats> output_stats;
  std::vector<LiteralKey> missing_in_golden;
  std::vector<LiteralKey> missing_in_test;

  std::string ToJson() const;

  std::string ToTsv() const;

  std::string ToDeviceTsv() const;

  std::string SummaryToString() const;
};

struct ModelComparisonOptions {
  int num_threads = 16;
  std::vector<int64_t> target_devices;
  ComparisonOptions comparison_options = {
      0.0,  // abs_error_bound
      0.0,  // rel_error_bound
      10,   // max_mismatches_to_record
      0.5,  // heatmap_yellow_pct
  };
};

// Scans golden_dir and test_dir, pairs up corresponding device/literal files,
// executes comparisons in parallel, and produces a ModelComparisonResult.
absl::StatusOr<ModelComparisonResult> CompareModelDirectories(
    absl::string_view golden_dir, absl::string_view test_dir,
    const ModelComparisonOptions& options = {});

// Writes output files (JSON, TSV, device TSV) based on non-empty paths.
absl::Status WriteModelComparisonOutputs(
    const ModelComparisonResult& result, absl::string_view json_path,
    absl::string_view tsv_path, absl::string_view device_tsv_path = "");

}  // namespace xla::compare_literals

#endif  // XLA_TOOLS_COMPARE_LITERALS_COMPARE_MODEL_LITERALS_H_
