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

#include "xla/tools/compare_literals/compare_model_literals.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <map>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "absl/algorithm/container.h"
#include "absl/base/no_destructor.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/status_macros.h"
#include "absl/status/statusor.h"
#include "absl/strings/match.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "re2/re2.h"
#include "xla/tools/compare_literals/compare_literals.h"
#include "xla/tsl/platform/env.h"
#include "xla/tsl/platform/threadpool.h"
#include "tsl/platform/path.h"

namespace xla::compare_literals {

std::optional<LiteralKey> ParseLiteralFilename(absl::string_view filename) {
  static const absl::NoDestructor<RE2> kLiteralRegex(
      R"((?:^|[._])literal_(\d+)\.pb$)");
  static const absl::NoDestructor<RE2> kDeviceRegex(
      R"((?:^|[._])device_(\d+)[._])");

  LiteralKey parsed;
  if (!RE2::PartialMatch(filename, *kLiteralRegex, &parsed.literal_id)) {
    return std::nullopt;
  }
  RE2::PartialMatch(filename, *kDeviceRegex, &parsed.device_id);
  return parsed;
}

namespace {

absl::StatusOr<std::map<LiteralKey, std::string>> LoadLiteralsFromDirectory(
    absl::string_view dir, absl::Span<const int64_t> target_devices,
    absl::string_view dir_label) {
  tsl::Env* env = tsl::Env::Default();
  if (!env->FileExists(dir).ok()) {
    return absl::NotFoundError(
        absl::StrCat(dir_label, " directory does not exist: ", dir));
  }

  std::vector<std::string> files;
  ABSL_RETURN_IF_ERROR(env->GetChildren(std::string(dir), &files))
      << "Failed to list " << dir_label << " directory: " << dir;

  std::map<LiteralKey, std::string> literal_map;
  for (const std::string& fname : files) {
    const auto parsed = ParseLiteralFilename(fname);
    if (!parsed.has_value()) {
      continue;
    }
    if (!target_devices.empty() &&
        !absl::c_linear_search(target_devices, parsed->device_id)) {
      continue;
    }
    auto [it, inserted] =
        literal_map.try_emplace(*parsed, tsl::io::JoinPath(dir, fname));
    if (!inserted) {
      LOG(WARNING) << "Duplicate literal file in " << dir_label
                   << "_dir: literal " << parsed->literal_id << ", device "
                   << parsed->device_id << " (" << it->second << " vs " << fname
                   << ")";
    }
  }
  return literal_map;
}

struct ComparisonTask {
  int64_t literal_id = 0;
  int64_t device_id = 0;
  std::string golden_path;
  std::string test_path;
};

struct TaskOutcome {
  ComparisonTask task;
  absl::StatusOr<ComparisonResult> result;
};

}  // namespace

absl::StatusOr<ModelComparisonResult> CompareModelDirectories(
    absl::string_view golden_dir, absl::string_view test_dir,
    const ModelComparisonOptions& options) {
  ABSL_ASSIGN_OR_RETURN(
      auto golden_map,
      LoadLiteralsFromDirectory(golden_dir, options.target_devices, "Golden"));
  ABSL_ASSIGN_OR_RETURN(
      auto test_map,
      LoadLiteralsFromDirectory(test_dir, options.target_devices, "Test"));

  ModelComparisonResult model_result;
  model_result.golden_dir = std::string(golden_dir);
  model_result.test_dir = std::string(test_dir);

  std::vector<ComparisonTask> tasks;
  std::vector<int64_t> discovered_devices;

  for (const auto& [key, g_path] : golden_map) {
    const auto it = test_map.find(key);
    if (it == test_map.end()) {
      LOG(WARNING) << "Missing in test_dir: literal " << key.literal_id
                   << ", device " << key.device_id;
      model_result.missing_in_test.push_back(key);
      model_result.summary.issues.push_back(
          absl::StrFormat("Literal %v, device %v: missing in test directory",
                          key.literal_id, key.device_id));
      continue;
    }
    ComparisonTask task;
    task.literal_id = key.literal_id;
    task.device_id = key.device_id;
    task.golden_path = g_path;
    task.test_path = it->second;
    tasks.push_back(std::move(task));
    if (!absl::c_linear_search(discovered_devices, key.device_id)) {
      discovered_devices.push_back(key.device_id);
    }
  }

  absl::c_sort(discovered_devices);
  model_result.devices = std::move(discovered_devices);

  for (const auto& [key, unused_path] : test_map) {
    if (golden_map.find(key) == golden_map.end()) {
      LOG(WARNING) << "Missing in golden_dir: literal " << key.literal_id
                   << ", device " << key.device_id;
      model_result.missing_in_golden.push_back(key);
      model_result.summary.issues.push_back(
          absl::StrFormat("Literal %v, device %v: missing in golden directory",
                          key.literal_id, key.device_id));
    }
  }

  if (golden_map.empty() && test_map.empty()) {
    return absl::NotFoundError(
        absl::StrCat("No matching literal files found between ", golden_dir,
                     " and ", test_dir));
  }

  std::vector<TaskOutcome> outcomes(tasks.size());
  if (!tasks.empty()) {
    for (size_t i = 0; i < tasks.size(); ++i) {
      outcomes[i].task = std::move(tasks[i]);
    }

    const int num_threads = std::min(static_cast<int>(tasks.size()),
                                     std::max(1, options.num_threads));
    tsl::thread::ThreadPool thread_pool(tsl::Env::Default(), "CompareModel",
                                        num_threads);

    for (size_t i = 0; i < tasks.size(); ++i) {
      thread_pool.Schedule([&outcomes, &options, i]() {
        outcomes[i].result = CompareLiteralFiles(outcomes[i].task.golden_path,
                                                 outcomes[i].task.test_path,
                                                 options.comparison_options);
      });
    }
  }

  // Group outcomes by literal_id
  std::map<int64_t, std::vector<TaskOutcome>> grouped_outcomes;
  for (auto& outcome : outcomes) {
    grouped_outcomes[outcome.task.literal_id].push_back(std::move(outcome));
  }

  for (auto& [lit_id, dev_outcomes] : grouped_outcomes) {
    OutputLiteralStats entry;
    entry.literal_index = lit_id;
    entry.literal_name = absl::StrCat("literal_", lit_id);

    double sum_mean_rel = 0.0;
    int valid_devices = 0;

    entry.aggregated_device_stats.exact_match_pct = 100.0;
    int64_t shape_device_id = -1;

    for (const auto& outcome : dev_outcomes) {
      const int64_t dev_id = outcome.task.device_id;
      if (!outcome.result.ok()) {
        LOG(ERROR) << "Comparison failed for literal " << lit_id << " device "
                   << dev_id << ": " << outcome.result.status();
        ++entry.failed_devices;
        entry.aggregated_device_stats.exact_match_pct = 0.0;
        if (absl::StrContains(outcome.result.status().message(),
                              "Shapes must be equal")) {
          // If shape does not match we report this in the summary.
          entry.shape_mismatch = true;
        }
        LiteralComparisonStats dev_res;
        dev_res.status = outcome.result.status();
        dev_res.shape_str = "-";
        dev_res.element_type = "-";
        entry.device_stats.emplace(dev_id, std::move(dev_res));
        model_result.summary.issues.push_back(
            absl::StrFormat("Literal %v, device %v: comparison failed: %s",
                            lit_id, dev_id, outcome.result.status().message()));
        continue;
      }

      const ComparisonResult& comp = *outcome.result;
      if (entry.shape_str.empty()) {
        entry.shape_str = comp.shape_str;
        entry.element_type = comp.element_type;
        entry.element_count = comp.total_elements;
        shape_device_id = dev_id;
      } else if (comp.element_type != entry.element_type) {
        // If shape does not match we report this in the summary.
        entry.shape_mismatch = true;
        model_result.summary.issues.push_back(absl::StrFormat(
            "Literal %v: element type mismatch across devices (device %v has "
            "%s vs device %v has %s)",
            lit_id, dev_id, comp.element_type, shape_device_id,
            entry.element_type));
      } else if (comp.shape_str != entry.shape_str) {
        // If shape does not match we report this in the summary.
        entry.shape_mismatch = true;
        model_result.summary.issues.push_back(absl::StrFormat(
            "Literal %v: shape mismatch across devices (device %v has %s vs "
            "device %v has %s)",
            lit_id, dev_id, comp.shape_str, shape_device_id, entry.shape_str));
      }

      const double match_pct =
          (comp.total_elements > 0)
              ? (100.0 * comp.exact_matches / comp.total_elements)
              : 100.0;

      double sugg_abs = 0.0;
      double sugg_rel = 0.0;
      if (comp.suggested_error_spec.has_value()) {
        sugg_abs = comp.suggested_error_spec->abs_bound;
        sugg_rel = comp.suggested_error_spec->rel_bound;
      }

      LiteralComparisonStats dev_res;
      dev_res.status = absl::OkStatus();
      dev_res.shape_str = comp.shape_str;
      dev_res.element_type = comp.element_type;
      dev_res.element_count = comp.total_elements;
      dev_res.exact_matches = comp.exact_matches;
      dev_res.exact_match_pct = match_pct;
      dev_res.mismatches = comp.mismatches;
      dev_res.nan_mismatches = comp.nan_mismatches;
      dev_res.inf_mismatches = comp.inf_mismatches;
      dev_res.max_abs_error = comp.max_abs_error;
      dev_res.max_rel_error = comp.max_rel_error;
      dev_res.mean_rel_error = comp.histogram.mean_rel_error;
      dev_res.suggested_abs_error = sugg_abs;
      dev_res.suggested_rel_error = sugg_rel;
      entry.device_stats.emplace(dev_id, std::move(dev_res));

      entry.aggregated_device_stats.exact_match_pct =
          std::min(entry.aggregated_device_stats.exact_match_pct, match_pct);
      entry.aggregated_device_stats.nan_mismatches += comp.nan_mismatches;
      entry.aggregated_device_stats.inf_mismatches += comp.inf_mismatches;
      entry.aggregated_device_stats.max_abs_error = std::max(
          entry.aggregated_device_stats.max_abs_error, comp.max_abs_error);
      entry.aggregated_device_stats.max_rel_error = std::max(
          entry.aggregated_device_stats.max_rel_error, comp.max_rel_error);
      entry.aggregated_device_stats.suggested_abs_error =
          std::max(entry.aggregated_device_stats.suggested_abs_error, sugg_abs);
      entry.aggregated_device_stats.suggested_rel_error =
          std::max(entry.aggregated_device_stats.suggested_rel_error, sugg_rel);

      sum_mean_rel += comp.histogram.mean_rel_error;
      ++valid_devices;
    }

    entry.num_devices = entry.device_stats.size();
    if (valid_devices == 0) {
      entry.aggregated_device_stats.exact_match_pct = 0.0;
    } else {
      entry.aggregated_device_stats.mean_rel_error =
          sum_mean_rel / valid_devices;
    }

    model_result.output_stats.push_back(std::move(entry));
  }

  absl::c_sort(model_result.output_stats,
               [](const OutputLiteralStats& a, const OutputLiteralStats& b) {
                 return a.literal_index < b.literal_index;
               });

  model_result.summary.total_literals = model_result.output_stats.size();
  for (const auto& out : model_result.output_stats) {
    model_result.summary.failed_device_comparisons += out.failed_devices;
    if (out.shape_mismatch) {
      ++model_result.summary.shape_mismatch_literals;
    }
    if (out.aggregated_device_stats.nan_mismatches > 0 ||
        out.aggregated_device_stats.inf_mismatches > 0) {
      ++model_result.summary.nan_inf_mismatch_literals;
    }
    const bool is_exact =
        !out.shape_mismatch && out.num_devices > 0 && out.failed_devices == 0 &&
        absl::c_all_of(out.device_stats, [](const auto& pair) {
          const auto& dev = pair.second;
          return dev.status.ok() && dev.exact_matches == dev.element_count &&
                 dev.nan_mismatches == 0 && dev.inf_mismatches == 0;
        });
    const bool is_within_tol =
        !is_exact && !out.shape_mismatch && out.num_devices > 0 &&
        out.failed_devices == 0 &&
        absl::c_all_of(out.device_stats, [](const auto& pair) {
          const auto& dev = pair.second;
          return dev.status.ok() && dev.mismatches == 0 &&
                 dev.nan_mismatches == 0 && dev.inf_mismatches == 0;
        });
    if (is_exact) {
      ++model_result.summary.exact_match_literals;
    } else if (is_within_tol) {
      ++model_result.summary.within_tolerance_literals;
    } else {
      ++model_result.summary.differing_literals;
    }
    if (out.aggregated_device_stats.max_abs_error >
        model_result.summary.worst_abs_error) {
      model_result.summary.worst_abs_error =
          out.aggregated_device_stats.max_abs_error;
      model_result.summary.worst_abs_literal = out.literal_index;
    }
    if (out.aggregated_device_stats.max_rel_error >
        model_result.summary.worst_rel_error) {
      model_result.summary.worst_rel_error =
          out.aggregated_device_stats.max_rel_error;
      model_result.summary.worst_rel_literal = out.literal_index;
    }
  }

  return model_result;
}

std::string ModelComparisonResult::ToTsv() const {
  std::string tsv =
      "literal\tshape\ttype\telements\tdevices\tfailed_devices\tmin_exact_pct\t"
      "max_abs_err\tmax_rel_err\tmean_rel_err\tsugg_abs_err\tsugg_rel_err\t"
      "nan_count\tinf_count\n";

  for (const auto& e : output_stats) {
    absl::StrAppendFormat(
        &tsv,
        "%v\t%s\t%s\t%v\t%v\t%v\t%.4f\t%.6e\t%.6e\t%.6e\t%.6e\t%.6e\t%v\t%v\n",
        e.literal_index, e.shape_str, e.element_type, e.element_count,
        e.num_devices, e.failed_devices,
        e.aggregated_device_stats.exact_match_pct,
        e.aggregated_device_stats.max_abs_error,
        e.aggregated_device_stats.max_rel_error,
        e.aggregated_device_stats.mean_rel_error,
        e.aggregated_device_stats.suggested_abs_error,
        e.aggregated_device_stats.suggested_rel_error,
        e.aggregated_device_stats.nan_mismatches,
        e.aggregated_device_stats.inf_mismatches);
  }
  return tsv;
}

std::string ModelComparisonResult::ToDeviceTsv() const {
  std::string tsv =
      "literal\tdevice\tcomparison_ok\terror_message\tshape\ttype\telements\t"
      "exact_matches\texact_pct\tmax_abs_err\tmax_rel_err\tmean_rel_err\t"
      "sugg_abs_err\tsugg_rel_err\tnan_count\tinf_count\n";

  for (const auto& e : output_stats) {
    for (const auto& [device_id, dev] : e.device_stats) {
      std::string sanitized_err = dev.status.ok() ? "" : dev.status.ToString();
      absl::c_replace(sanitized_err, '\t', ' ');
      absl::c_replace(sanitized_err, '\n', ' ');
      absl::StrAppendFormat(
          &tsv,
          "%v\t%v\t%s\t%s\t%s\t%s\t%v\t%v\t%.4f\t%.6e\t%.6e\t%.6e\t%.6e\t%."
          "6e\t%v\t%v\n",
          e.literal_index, device_id, dev.status.ok() ? "true" : "false",
          sanitized_err.empty() ? "-" : sanitized_err, dev.shape_str,
          dev.element_type, dev.element_count, dev.exact_matches,
          dev.exact_match_pct, dev.max_abs_error, dev.max_rel_error,
          dev.mean_rel_error, dev.suggested_abs_error, dev.suggested_rel_error,
          dev.nan_mismatches, dev.inf_mismatches);
    }
  }
  return tsv;
}

std::string ModelComparisonResult::SummaryToString() const {
  std::string summary_str;
  absl::StrAppend(&summary_str, "Model Comparison Summary:\n");
  absl::StrAppend(&summary_str, "  Golden Dir: ", golden_dir, "\n");
  absl::StrAppend(&summary_str, "  Test Dir:   ", test_dir, "\n");
  absl::StrAppendFormat(&summary_str,
                        "  Total Literals: %v across %v device(s)\n",
                        summary.total_literals, devices.size());
  absl::StrAppendFormat(
      &summary_str, "  Exact Match Literals: %v (%.2f%%)\n",
      summary.exact_match_literals,
      summary.total_literals == 0
          ? 0.0
          : 100.0 * summary.exact_match_literals / summary.total_literals);
  if (summary.within_tolerance_literals > 0) {
    absl::StrAppendFormat(&summary_str, "  Within Tolerance:     %v (%.2f%%)\n",
                          summary.within_tolerance_literals,
                          summary.total_literals == 0
                              ? 0.0
                              : 100.0 * summary.within_tolerance_literals /
                                    summary.total_literals);
  }
  absl::StrAppendFormat(
      &summary_str, "  Differing Literals:   %v (%.2f%%)\n",
      summary.differing_literals,
      summary.total_literals == 0
          ? 0.0
          : 100.0 * summary.differing_literals / summary.total_literals);
  if (summary.shape_mismatch_literals > 0) {
    absl::StrAppendFormat(&summary_str, "  Shape Mismatch Literals:    %v\n",
                          summary.shape_mismatch_literals);
  }
  if (summary.failed_device_comparisons > 0) {
    absl::StrAppendFormat(&summary_str, "  Failed Device Comparisons:  %v\n",
                          summary.failed_device_comparisons);
  }
  if (summary.nan_inf_mismatch_literals > 0) {
    absl::StrAppendFormat(&summary_str, "  NaN/Inf Mismatch Literals:  %v\n",
                          summary.nan_inf_mismatch_literals);
  }
  if (!missing_in_test.empty()) {
    absl::StrAppendFormat(
        &summary_str,
        "  Missing in Test Dir:        %zu literal/device pair(s)\n",
        missing_in_test.size());
  }
  if (!missing_in_golden.empty()) {
    absl::StrAppendFormat(
        &summary_str,
        "  Missing in Golden Dir:      %zu literal/device pair(s)\n",
        missing_in_golden.size());
  }
  if (summary.worst_abs_literal >= 0) {
    absl::StrAppendFormat(&summary_str,
                          "  Worst Absolute Error: %.6e (literal_%v)\n",
                          summary.worst_abs_error, summary.worst_abs_literal);
  }
  if (summary.worst_rel_literal >= 0) {
    absl::StrAppendFormat(&summary_str,
                          "  Worst Relative Error: %.6e (literal_%v)\n",
                          summary.worst_rel_error, summary.worst_rel_literal);
  }
  if (!summary.issues.empty()) {
    absl::StrAppendFormat(&summary_str, "\nIssues (%zu):\n",
                          summary.issues.size());
    constexpr size_t kMaxIssuesToPrint = 20;
    const size_t num_to_print =
        std::min(summary.issues.size(), kMaxIssuesToPrint);
    for (size_t i = 0; i < num_to_print; ++i) {
      absl::StrAppend(&summary_str, "  - ", summary.issues[i], "\n");
    }
    if (summary.issues.size() > kMaxIssuesToPrint) {
      absl::StrAppendFormat(&summary_str, "  ... and %zu more issue(s)\n",
                            summary.issues.size() - kMaxIssuesToPrint);
    }
  }
  return summary_str;
}

absl::Status WriteModelComparisonOutputs(const ModelComparisonResult& result,
                                         absl::string_view tsv_path,
                                         absl::string_view device_tsv_path) {
  tsl::Env* env = tsl::Env::Default();

  auto create_parent_dir = [env](absl::string_view file_path) -> absl::Status {
    if (file_path.empty()) {
      return absl::OkStatus();
    }
    const absl::string_view dir = tsl::io::Dirname(file_path);
    if (!dir.empty() && !env->FileExists(dir).ok()) {
      ABSL_RETURN_IF_ERROR(env->RecursivelyCreateDir(dir))
          << "Failed to create directory: " << dir;
    }
    return absl::OkStatus();
  };

  if (!tsv_path.empty()) {
    ABSL_RETURN_IF_ERROR(create_parent_dir(tsv_path));
    ABSL_RETURN_IF_ERROR(tsl::WriteStringToFile(env, tsv_path, result.ToTsv()))
        << "Failed to write TSV output to: " << tsv_path;
  }
  if (!device_tsv_path.empty()) {
    ABSL_RETURN_IF_ERROR(create_parent_dir(device_tsv_path));
    ABSL_RETURN_IF_ERROR(
        tsl::WriteStringToFile(env, device_tsv_path, result.ToDeviceTsv()))
        << "Failed to write Device TSV output to: " << device_tsv_path;
  }
  return absl::OkStatus();
}

}  // namespace xla::compare_literals
