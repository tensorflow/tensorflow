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
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <map>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "absl/algorithm/container.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/status_macros.h"
#include "absl/status/statusor.h"
#include "absl/strings/match.h"
#include "absl/strings/numbers.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"
#include "absl/strings/string_view.h"
#include "third_party/jsoncpp/include/json/value.h"
#include "third_party/jsoncpp/include/json/writer.h"
#include "xla/tools/compare_literals/compare_literals.h"
#include "xla/tsl/platform/env.h"
#include "xla/tsl/platform/threadpool.h"
#include "tsl/platform/path.h"

namespace xla::compare_literals {
namespace {

constexpr absl::string_view kLiteralPrefix = "literal_";
constexpr absl::string_view kDevicePrefix = "device_";

// Parses filename patterns such as:
// "output.hlo_0.task_0.device_7.literal_27.pb",
// "output.device_1.literal_3.pb", or "literal_5.pb".
std::optional<LiteralKey> ParseLiteralFilename(absl::string_view filename) {
  if (!absl::EndsWith(filename, ".pb")) {
    return std::nullopt;
  }
  const size_t lit_pos = filename.rfind(kLiteralPrefix);
  if (lit_pos == absl::string_view::npos) {
    return std::nullopt;
  }

  const absl::string_view after_lit =
      filename.substr(lit_pos + kLiteralPrefix.size());
  const size_t dot_pos = after_lit.find_first_of("._");
  const absl::string_view lit_num_str = (dot_pos == absl::string_view::npos)
                                            ? after_lit
                                            : after_lit.substr(0, dot_pos);
  LiteralKey parsed;
  if (!absl::SimpleAtoi(lit_num_str, &parsed.literal_id) ||
      parsed.literal_id < 0) {
    return std::nullopt;
  }

  const size_t dev_pos = filename.rfind(kDevicePrefix);
  if (dev_pos != absl::string_view::npos) {
    const absl::string_view after_dev =
        filename.substr(dev_pos + kDevicePrefix.size());
    const size_t dev_dot = after_dev.find_first_of("._");
    const absl::string_view dev_num_str = (dev_dot == absl::string_view::npos)
                                              ? after_dev
                                              : after_dev.substr(0, dev_dot);
    if (!absl::SimpleAtoi(dev_num_str, &parsed.device_id) ||
        parsed.device_id < 0) {
      return std::nullopt;
    }
  } else {
    parsed.device_id = 0;
  }
  return parsed;
}

void SetJsonDouble(Json::Value& parent, const char* key, double val) {
  if (std::isnan(val)) {
    parent[key] = "NaN";
  } else if (std::isinf(val)) {
    parent[key] = (val > 0) ? "Infinity" : "-Infinity";
  } else {
    parent[key] = val;
  }
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
  tsl::Env* env = tsl::Env::Default();

  if (!env->FileExists(golden_dir).ok()) {
    return absl::NotFoundError(
        absl::StrCat("Golden directory does not exist: ", golden_dir));
  }
  if (!env->FileExists(test_dir).ok()) {
    return absl::NotFoundError(
        absl::StrCat("Test directory does not exist: ", test_dir));
  }

  std::vector<std::string> golden_files;
  ABSL_RETURN_IF_ERROR(env->GetChildren(std::string(golden_dir), &golden_files))
      << "Failed to list golden directory: " << golden_dir;
  absl::c_sort(golden_files);

  std::vector<std::string> test_files;
  ABSL_RETURN_IF_ERROR(env->GetChildren(std::string(test_dir), &test_files))
      << "Failed to list test directory: " << test_dir;
  absl::c_sort(test_files);

  // Map key is LiteralKey (literal_id, device_id)
  std::map<LiteralKey, std::string> golden_map;
  for (const std::string& fname : golden_files) {
    const auto parsed = ParseLiteralFilename(fname);
    if (!parsed.has_value()) {
      continue;
    }
    if (!options.target_devices.empty() &&
        !absl::c_linear_search(options.target_devices, parsed->device_id)) {
      continue;
    }
    auto [it, inserted] =
        golden_map.try_emplace(*parsed, tsl::io::JoinPath(golden_dir, fname));
    if (!inserted) {
      LOG(WARNING) << "Duplicate literal file in golden_dir: literal "
                   << parsed->literal_id << ", device " << parsed->device_id
                   << " (" << it->second << " vs " << fname << ")";
    }
  }

  std::map<LiteralKey, std::string> test_map;
  for (const std::string& fname : test_files) {
    const auto parsed = ParseLiteralFilename(fname);
    if (!parsed.has_value()) {
      continue;
    }
    if (!options.target_devices.empty() &&
        !absl::c_linear_search(options.target_devices, parsed->device_id)) {
      continue;
    }
    auto [it, inserted] =
        test_map.try_emplace(*parsed, tsl::io::JoinPath(test_dir, fname));
    if (!inserted) {
      LOG(WARNING) << "Duplicate literal file in test_dir: literal "
                   << parsed->literal_id << ", device " << parsed->device_id
                   << " (" << it->second << " vs " << fname << ")";
    }
  }

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
    tsl::thread::ThreadPool thread_pool(env, "CompareModel", num_threads);

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

    for (const auto& outcome : dev_outcomes) {
      const int64_t dev_id = outcome.task.device_id;
      if (!outcome.result.ok()) {
        LOG(ERROR) << "Comparison failed for literal " << lit_id << " device "
                   << dev_id << ": " << outcome.result.status();
        ++entry.failed_devices;
        entry.aggregated_device_stats.exact_match_pct = 0.0;
        LiteralComparisonStats dev_res;
        dev_res.status = outcome.result.status();
        entry.device_stats.emplace(dev_id, std::move(dev_res));
        continue;
      }

      const ComparisonResult& comp = *outcome.result;
      if (entry.shape_str.empty()) {
        entry.shape_str = comp.shape_str;
        entry.element_type = comp.element_type;
        entry.element_count = comp.total_elements;
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
    if (out.aggregated_device_stats.nan_mismatches > 0 ||
        out.aggregated_device_stats.inf_mismatches > 0) {
      ++model_result.summary.nan_inf_mismatch_literals;
    }
    const bool is_exact =
        out.num_devices > 0 && out.failed_devices == 0 &&
        absl::c_all_of(out.device_stats, [&out](const auto& pair) {
          const auto& dev = pair.second;
          return dev.status.ok() && dev.exact_matches == out.element_count &&
                 dev.nan_mismatches == 0 && dev.inf_mismatches == 0;
        });
    const bool is_within_tol =
        !is_exact && out.num_devices > 0 && out.failed_devices == 0 &&
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

std::string ModelComparisonResult::ToJson() const {
  Json::Value root(Json::objectValue);
  root["golden_dir"] = golden_dir;
  root["test_dir"] = test_dir;
  root["total_literals"] = summary.total_literals;

  Json::Value summary_val(Json::objectValue);
  summary_val["total_literals"] = summary.total_literals;
  summary_val["exact_match_literals"] = summary.exact_match_literals;
  summary_val["within_tolerance_literals"] = summary.within_tolerance_literals;
  summary_val["differing_literals"] = summary.differing_literals;
  summary_val["failed_device_comparisons"] = summary.failed_device_comparisons;
  summary_val["nan_inf_mismatch_literals"] = summary.nan_inf_mismatch_literals;
  SetJsonDouble(summary_val, "worst_abs_error", summary.worst_abs_error);
  summary_val["worst_abs_literal"] = summary.worst_abs_literal;
  SetJsonDouble(summary_val, "worst_rel_error", summary.worst_rel_error);
  summary_val["worst_rel_literal"] = summary.worst_rel_literal;
  root["summary"] = summary_val;

  root["devices"] = Json::Value(Json::arrayValue);
  for (int64_t d : devices) {
    root["devices"].append(d);
  }

  if (!missing_in_test.empty()) {
    root["missing_in_test"] = Json::Value(Json::arrayValue);
    for (const auto& item : missing_in_test) {
      Json::Value m(Json::objectValue);
      m["literal"] = item.literal_id;
      m["device"] = item.device_id;
      root["missing_in_test"].append(std::move(m));
    }
  }

  if (!missing_in_golden.empty()) {
    root["missing_in_golden"] = Json::Value(Json::arrayValue);
    for (const auto& item : missing_in_golden) {
      Json::Value m(Json::objectValue);
      m["literal"] = item.literal_id;
      m["device"] = item.device_id;
      root["missing_in_golden"].append(std::move(m));
    }
  }

  root["literals"] = Json::Value(Json::arrayValue);
  for (const auto& e : output_stats) {
    Json::Value lit(Json::objectValue);
    lit["index"] = e.literal_index;
    lit["name"] = e.literal_name;
    lit["shape"] = e.shape_str;
    lit["element_type"] = e.element_type;
    lit["element_count"] = e.element_count;

    Json::Value agg(Json::objectValue);
    agg["num_devices"] = e.num_devices;
    agg["failed_devices"] = e.failed_devices;
    agg["min_exact_match_pct"] = e.aggregated_device_stats.exact_match_pct;
    SetJsonDouble(agg, "max_abs_error",
                  e.aggregated_device_stats.max_abs_error);
    SetJsonDouble(agg, "max_rel_error",
                  e.aggregated_device_stats.max_rel_error);
    SetJsonDouble(agg, "mean_rel_error",
                  e.aggregated_device_stats.mean_rel_error);
    SetJsonDouble(agg, "suggested_abs_error",
                  e.aggregated_device_stats.suggested_abs_error);
    SetJsonDouble(agg, "suggested_rel_error",
                  e.aggregated_device_stats.suggested_rel_error);
    agg["nan_mismatches"] = e.aggregated_device_stats.nan_mismatches;
    agg["inf_mismatches"] = e.aggregated_device_stats.inf_mismatches;
    lit["aggregate"] = std::move(agg);

    lit["devices"] = Json::Value(Json::arrayValue);
    for (const auto& [device_id, dev] : e.device_stats) {
      Json::Value d(Json::objectValue);
      d["device"] = device_id;
      d["comparison_ok"] = dev.status.ok();
      if (!dev.status.ok()) {
        d["error_message"] = dev.status.ToString();
      }
      d["exact_matches"] = dev.exact_matches;
      d["exact_match_pct"] = dev.exact_match_pct;
      d["mismatches"] = dev.mismatches;
      SetJsonDouble(d, "max_abs_error", dev.max_abs_error);
      SetJsonDouble(d, "max_rel_error", dev.max_rel_error);
      SetJsonDouble(d, "mean_rel_error", dev.mean_rel_error);
      SetJsonDouble(d, "suggested_abs_error", dev.suggested_abs_error);
      SetJsonDouble(d, "suggested_rel_error", dev.suggested_rel_error);
      d["nan_mismatches"] = dev.nan_mismatches;
      d["inf_mismatches"] = dev.inf_mismatches;
      lit["devices"].append(std::move(d));
    }

    root["literals"].append(std::move(lit));
  }

  Json::StreamWriterBuilder builder;
  builder["indentation"] = "  ";
  return Json::writeString(builder, root);
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
          sanitized_err.empty() ? "-" : sanitized_err, e.shape_str,
          e.element_type, e.element_count, dev.exact_matches,
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
  if (summary.failed_device_comparisons > 0) {
    absl::StrAppendFormat(&summary_str, "  Failed Device Comparisons:  %v\n",
                          summary.failed_device_comparisons);
  }
  if (summary.nan_inf_mismatch_literals > 0) {
    absl::StrAppendFormat(&summary_str, "  NaN/Inf Mismatch Literals: %v\n",
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
  return summary_str;
}

absl::Status WriteModelComparisonOutputs(const ModelComparisonResult& result,
                                         absl::string_view json_path,
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

  if (!json_path.empty()) {
    ABSL_RETURN_IF_ERROR(create_parent_dir(json_path));
    ABSL_RETURN_IF_ERROR(tsl::WriteStringToFile(env, json_path, result.ToJson()))
        << "Failed to write JSON output to: " << json_path;
  }
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
