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

#include "xla/hlo/tools/comparison/comparison_manager.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "absl/algorithm/container.h"
#include "absl/base/thread_annotations.h"
#include "absl/cleanup/cleanup.h"
#include "absl/container/inlined_vector.h"
#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/status_macros.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "absl/synchronization/mutex.h"
#include "absl/synchronization/notification.h"
#include "absl/types/span.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/ir/hlo_print_options.h"
#include "xla/hlo/tools/comparison/comparison_manager.pb.h"
#include "xla/hlo/tools/comparison/comparison_options.pb.h"
#include "xla/hlo/tools/comparison/comparison_service.pb.h"
#include "xla/hlo/tools/hlo_diff/hlo_gumgraph_diff.h"
#include "xla/shape.h"
#include "xla/shape_util.h"
#include "xla/tsl/platform/env.h"
#include "tsl/platform/path.h"

namespace xla::numerics::comparison {

using ComparisonVariant = ComparisonOptions::ComparisonVariant;

namespace {
ComparisonVariant FlipVariant(ComparisonVariant variant) {
  return variant == ComparisonOptions::COMPARISON_VARIANT_TARGET
             ? ComparisonOptions::COMPARISON_VARIANT_BASELINE
             : ComparisonOptions::COMPARISON_VARIANT_TARGET;
};

// Dumps the HLO module to the given directory. The filename is in the format of
// <dump_dir>/<module_name>.<variant>.hlo.txt.
//
// Note that we don't use anything in `compiler/xla/service/dump.h` because we
// don't want to couple comparison tool with the XLA dump system.
absl::Status DumpHloModule(const HloModule& module, ComparisonVariant variant,
                           absl::string_view dump_dir) {
  std::string filename = tsl::io::JoinPath(
      dump_dir, absl::StrCat(module.name(), ".",
                             ComparisonVariantToString(variant), ".hlo.txt"));
  auto options = HloPrintOptions::ShortParsable();
  options.set_print_large_constants(true);
  options.set_print_metadata(true);
  std::string module_text = module.ToString(options);
  absl::Status create_dir_status = tsl::Env::Default()->RecursivelyCreateDir(
      std::string(tsl::io::Dirname(filename)));
  if (!create_dir_status.ok() && !absl::IsAlreadyExists(create_dir_status)) {
    return create_dir_status;
  }
  return tsl::WriteStringToFile(tsl::Env::Default(), filename, module_text);
}

// Copies the tensor summary from `src` to `dst`. If `log_samples` is true, the
// samples will also be copied.
void CopyTensorSummary(const TensorSummary& src, TensorSummary* dst,
                       bool log_samples) {
  *dst->mutable_shape() = src.shape();
  *dst->mutable_metadata() = src.metadata();
  dst->set_mean(src.mean());
  dst->set_stddev(src.stddev());
  dst->set_min(src.min());
  dst->set_max(src.max());
  dst->set_checksum(src.checksum());
  dst->set_non_zero_mean(src.non_zero_mean());
  dst->set_non_zero_stddev(src.non_zero_stddev());
  if (log_samples) {
    *dst->mutable_samples() = src.samples();
  }
}

// Returns true if the two floats are considered different. If both values are
// NaN, they are considered identical, unlike the comparison operator !=. This
// is because having two nan doesn't mean there is any divergence between
// two runs.
//
// Note that using `MathUtil::AlmostEquals` is not appropriate here because
// the goal of the comparison tool is to detect any numerical divergence between
// two runs, even if they are very close to each other.
bool areFloatsReallyDifferent(float a, float b) {
  if (std::isnan(a) && std::isnan(b)) {
    return false;
  }
  return a != b;
}

}  // namespace

TensorComparisonResult ComparisonManager::CompareTensor(
    const TensorSummary& baseline, const TensorSummary& target) const {
  TensorComparisonResult result;

  *result.mutable_baseline_original_positions() =
      baseline.metadata().original_positions();
  *result.mutable_target_original_positions() =
      target.metadata().original_positions();

  CopyTensorSummary(target, result.mutable_target_summary(), log_samples_);

  absl::StatusOr<Shape> baseline_shape = Shape::FromProto(baseline.shape());
  absl::StatusOr<Shape> target_shape = Shape::FromProto(target.shape());
  bool shape_match = true;
  if (baseline_shape.ok() && target_shape.ok()) {
    shape_match = ShapeUtil::Equal(*baseline_shape, *target_shape);
  } else {
    LOG(WARNING) << "[Comparison Tool] Baseline and target tensor shapes "
                    "cannot be parsed: "
                 << baseline_shape.status() << " vs " << target_shape.status();
    shape_match = false;
  }
  result.set_shape_match(shape_match);
  bool summary_match = true;
  if (!shape_match ||
      areFloatsReallyDifferent(target.mean(), baseline.mean()) ||
      areFloatsReallyDifferent(target.stddev(), baseline.stddev()) ||
      areFloatsReallyDifferent(target.min(), baseline.min()) ||
      areFloatsReallyDifferent(target.max(), baseline.max()) ||
      target.checksum() != baseline.checksum() ||
      areFloatsReallyDifferent(target.non_zero_mean(),
                               baseline.non_zero_mean()) ||
      areFloatsReallyDifferent(target.non_zero_stddev(),
                               baseline.non_zero_stddev())) {
    summary_match = false;
  }

  if (!absl::c_equal(
          target.samples(), baseline.samples(),
          [](float a, float b) { return !areFloatsReallyDifferent(a, b); })) {
    summary_match = false;
  }
  if (summary_match) {
    result.set_summary_match(true);
  } else {
    TensorComparisonResult::TensorSummaryDelta& summary_delta =
        *result.mutable_summary_delta();

    CopyTensorSummary(baseline, summary_delta.mutable_baseline_summary(),
                      log_samples_);

    const auto& baseline_samples = baseline.samples();
    const auto& target_samples = target.samples();

    if (baseline_samples.size() != target_samples.size()) {
      LOG(WARNING)
          << "[Comparison Tool] Baseline and target tensor samples have "
             "different sizes ("
          << baseline_samples.size() << " vs " << target_samples.size()
          << "). Sample delta statistics will be NaN.";
      summary_delta.set_sample_delta_max(
          std::numeric_limits<float>::quiet_NaN());  // Ensure max is also NaN
      summary_delta.set_sample_delta_mean(
          std::numeric_limits<float>::quiet_NaN());
      summary_delta.set_sample_delta_stddev(
          std::numeric_limits<float>::quiet_NaN());
      return result;
    }

    std::vector<double> sample_deltas;
    // Reserve space to avoid multiple reallocations.
    sample_deltas.reserve(baseline_samples.size());

    for (int i = 0; i < baseline_samples.size(); ++i) {
      double baseline_val = baseline_samples.Get(i);
      double target_val = target_samples.Get(i);
      // Only consider the delta if neither sample value is NaN.
      if (!std::isnan(baseline_val) && !std::isnan(target_val)) {
        sample_deltas.push_back(target_val - baseline_val);
      }
    }

    // If, after filtering, there are no valid sample deltas (e.g., all pairs
    // had NaNs, or original samples were empty), set all sample delta
    // statistics to NaN.
    if (sample_deltas.empty()) {
      summary_delta.set_sample_delta_max(
          std::numeric_limits<float>::quiet_NaN());
      summary_delta.set_sample_delta_mean(
          std::numeric_limits<float>::quiet_NaN());
      summary_delta.set_sample_delta_stddev(
          std::numeric_limits<float>::quiet_NaN());
      return result;
    }

    // Calculations proceed only with valid, non-NaN-derived deltas.
    double max_abs_delta = 0.0;  // For max of absolute deltas
    double sum_of_deltas = 0.0;
    for (double delta : sample_deltas) {
      sum_of_deltas += delta;
      max_abs_delta = std::max(max_abs_delta, std::abs(delta));
    }
    summary_delta.set_sample_delta_max(static_cast<float>(max_abs_delta));

    // sample_deltas is guaranteed non-empty here.
    double mean_of_deltas =
        sum_of_deltas / static_cast<double>(sample_deltas.size());
    summary_delta.set_sample_delta_mean(static_cast<float>(mean_of_deltas));

    // Calculate standard deviation of sample_deltas (population stddev).
    double sum_of_squared_differences = 0.0;
    for (double delta : sample_deltas) {
      sum_of_squared_differences +=
          (delta - mean_of_deltas) * (delta - mean_of_deltas);
    }
    // sample_deltas is guaranteed non-empty here.
    double stddev_of_deltas = std::sqrt(
        sum_of_squared_differences / static_cast<double>(sample_deltas.size()));
    summary_delta.set_sample_delta_stddev(static_cast<float>(stddev_of_deltas));
  }

  return result;
}

std::optional<ComparisonManager::TensorComparisonKey>
ComparisonManager::FlipKey(absl::string_view hlo_module_name,
                           const TensorComparisonKey& key) const
    ABSL_EXCLUSIVE_LOCKS_REQUIRED(mutex_) {
  const auto state = comparison_states_.find(hlo_module_name);
  if (state == comparison_states_.end()) {
    LOG(ERROR) << "[Comparison Tool] HLO module " << hlo_module_name
               << " is not found in the comparison states when flipping key.";
    return std::nullopt;
  }
  absl::InlinedVector<TensorPositionStruct, 2> positions;
  positions.reserve(key.positions.size());

  auto process_positions = [&](const auto& map_view) -> bool {
    for (const auto& position : key.positions) {
      TensorPositionStruct flipped_position = position;
      auto it = map_view.find(position.instruction_name);
      if (it == map_view.end()) {
        return false;
      }
      flipped_position.instruction_name = it->second;
      positions.emplace_back(flipped_position);
    }
    return true;
  };

  bool success;
  if (key.variant == ComparisonOptions::COMPARISON_VARIANT_TARGET) {
    success = process_positions(state->second.instruction_diff_map_left);
  } else {
    success = process_positions(state->second.instruction_diff_map_right);
  }

  if (!success) {
    return std::nullopt;
  }

  return std::make_optional<TensorComparisonKey>(FlipVariant(key.variant),
                                                 std::move(positions));
}

absl::Status ComparisonManager::RegisterHloModule(ComparisonVariant variant,
                                                  const HloModule& module) {
  absl::MutexLock lock(mutex_);
  LOG(INFO) << "[Comparison Tool] Registering HLO module " << module.name()
            << " with id: " << module.unique_id()
            << " with variant: " << ComparisonVariantToString(variant);
  comparison_states_[module.name()].variants.insert(variant);
  absl::string_view module_name = module.name();
  auto [other_pair, inserted] =
      hlo_module_map_.try_emplace(module_name, &module, variant);
  ABSL_RETURN_IF_ERROR(DumpHloModule(module, variant, hlo_module_dump_dir_));
  if (inserted) {
    // If insertion is successful, it means the other HLO module is not added
    // yet. So we just return and the next add would trigger the comparison.
    return absl::OkStatus();
  }
  LOG(INFO) << "[Comparison Tool] Comparing HLO module " << module.name()
            << " with id: " << module.unique_id();
  if (other_pair->second.variant == variant) {
    // if the same variant is registered again due to various reasons (e.g. RPC
    // retries), we just ignore it and return.
    return absl::OkStatus();
  }
  const HloModule* baseline_module;
  const HloModule* target_module;
  if (variant == ComparisonOptions::COMPARISON_VARIANT_TARGET) {
    baseline_module = other_pair->second.module;
    target_module = &module;
  } else {
    baseline_module = &module;
    target_module = other_pair->second.module;
  }
  hlo_module_map_.erase(module_name);

  ABSL_ASSIGN_OR_RETURN(hlo_diff::HloGumgraphDiffResults diff_results,
                   hlo_diff::ComputeDiff(*baseline_module, *target_module));

  auto& state = comparison_states_[module_name];
  for (const auto& instructions :
       {diff_results.diff_result->unchanged_instructions,
        diff_results.diff_result->changed_instructions}) {
    // NOLINTNEXTLINE
    for (const auto& [baseline_inst, target_inst] : instructions) {
      LOG(INFO) << "[Comparison Tool] diff pair: " << baseline_inst->name()
                << " vs " << target_inst->name();
      state.instruction_diff_map_left.insert(
          std::make_pair(std::string(target_inst->name()),
                         std::string(baseline_inst->name())));
      state.instruction_diff_map_right.insert(
          std::make_pair(std::string(baseline_inst->name()),
                         std::string(target_inst->name())));
    }
  }
  LOG(INFO) << "[Comparison Tool] Completed diffing HLO module "
            << module.name() << " with id: " << module.unique_id();
  state.ready_notification->Notify();
  return absl::OkStatus();
}

absl::Status ComparisonManager::WaitUntilHloModuleIsReady(
    absl::string_view hlo_module_name) {
  absl::Notification* notification = nullptr;
  {
    absl::MutexLock lock(mutex_);
    LOG(INFO) << "[Comparison Tool] Waiting for HLO module to be ready: "
              << hlo_module_name;
    auto& state = comparison_states_[hlo_module_name];
    notification = state.ready_notification.get();
  }
  notification->WaitForNotification();
  return absl::OkStatus();
}

absl::Status ComparisonManager::RegisterHloModuleRun(
    ComparisonVariant variant, int32_t logical_device_id, uint64_t run_id,
    absl::string_view hlo_module_name) {
  absl::MutexLock lock(mutex_);
  LOG(INFO) << "[Comparison Tool] Registering HLO module " << hlo_module_name
            << " with run id: " << run_id
            << " with variant: " << ComparisonVariantToString(variant);
  HloRun hlo_run = {/*logical_device_id=*/logical_device_id,
                    /*run_id=*/run_id,
                    /*variant=*/variant};
  hlo_module_name_by_hlo_run_[hlo_run] = hlo_module_name;
  comparison_states_[hlo_module_name].hlo_runs.insert(hlo_run);
  if (!any_modules_running_notification_.HasBeenNotified()) {
    any_modules_running_notification_.Notify();
  }
  return absl::OkStatus();
}

absl::Status ComparisonManager::WaitUntilAnyHloModuleIsRunning() {
  any_modules_running_notification_.WaitForNotification();
  return absl::OkStatus();
}

absl::Status ComparisonManager::RecordTensor(ComparisonVariant variant,
                                             absl::string_view hlo_module_name,
                                             const TensorSummary& summary) {
  absl::MutexLock lock(mutex_);
  LOG(INFO) << "[Comparison Tool] RecordTensor in module " << hlo_module_name
            << " with variant " << ComparisonVariantToString(variant)
            << " and summary metadata:\n"
            << summary.metadata().DebugString() << "\nshape:\n"
            << summary.shape().DebugString();
  TensorComparisonKey key(
      variant, absl::MakeConstSpan(summary.metadata().original_positions()));
  auto other_key = FlipKey(hlo_module_name, key);
  if (!other_key.has_value()) {
    LOG(INFO) << "[Comparison Tool] Skipping comparison because failed to find "
                 "the corresponding tensor for key:\n"
              << key;
    return absl::OkStatus();
  }
  auto& state = comparison_states_[hlo_module_name];
  auto& map = state.tensor_summaries;
  if (!map.contains(*other_key)) {
    // the other tensor is not present. Store this one and wait for the other
    // tensor to be ready.
    map[key] = summary;
    return absl::OkStatus();
  }
  LOG(INFO) << "[Comparison Tool] Both tensors are present. Comparing them.";
  // both tensors are present. Retrieve them, compare and invoke the callback.
  const TensorSummary* baseline_tensor_summary;
  const TensorSummary* target_tensor_summary;
  if (variant == ComparisonOptions::COMPARISON_VARIANT_TARGET) {
    baseline_tensor_summary = &map[*other_key];
    target_tensor_summary = &summary;
  } else {
    baseline_tensor_summary = &summary;
    target_tensor_summary = &map[*other_key];
  }
  const TensorComparisonResult result =
      CompareTensor(*baseline_tensor_summary, *target_tensor_summary);
  // Remove the entries from the map to reduce memory footprint.
  map.erase(key);
  map.erase(*other_key);
  state.stats.set_num_tensors_compared(state.stats.num_tensors_compared() + 1);
  if (!result.summary_match()) {
    state.stats.set_num_tensors_with_delta(
        state.stats.num_tensors_with_delta() + 1);
  }

  return OnComparisonResult(hlo_module_name, result);
}

absl::Status ComparisonManager::FinishHloModuleRun(
    ComparisonVariant variant, int32_t logical_device_id, uint64_t run_id,
    const ModuleStats& module_stats) {
  absl::MutexLock lock(mutex_);
  HloRun hlo_run = {/*logical_device_id=*/logical_device_id,
                    /*run_id=*/run_id,
                    /*variant=*/variant};
  auto it = hlo_module_name_by_hlo_run_.find(hlo_run);
  if (it == hlo_module_name_by_hlo_run_.end()) {
    LOG(ERROR)
        << "[Comparison Tool] Attempted to finish an HLO module that was never "
           "added with run: "
        << hlo_run;
    return absl::OkStatus();
  }
  absl::string_view hlo_module_name = it->second;
  LOG(INFO) << "[Comparison Tool] Finishing HLO module " << hlo_module_name
            << " with variant: " << ComparisonVariantToString(variant);
  auto pair = comparison_states_.find(hlo_module_name);
  if (pair == comparison_states_.end()) {
    // The module is already finished. We tolerate redundant finish calls due to
    // RPC retries.
    return absl::OkStatus();
  }
  auto& comparison_state = pair->second;
  if (variant == ComparisonOptions::COMPARISON_VARIANT_TARGET) {
    *comparison_state.stats.mutable_target_stats() = module_stats;
  } else {
    *comparison_state.stats.mutable_baseline_stats() = module_stats;
  }
  comparison_state.hlo_runs.erase(hlo_run);
  if (!comparison_state.hlo_runs.empty()) {
    return absl::OkStatus();
  }
  absl::Notification* finish_notification =
      comparison_state.finish_notification.get();
  bool all_tracked_modules_fully_finished = true;
  // NOLINTNEXTLINE
  for (const auto& [unused_name, state] : comparison_states_) {
    if (!state.hlo_runs.empty()) {
      all_tracked_modules_fully_finished = false;
      break;
    }
  }
  // NOLINTNEXTLINE
  for (auto& unmatched_pair : comparison_state.tensor_summaries) {
    if (unmatched_pair.first.variant ==
        ComparisonOptions::COMPARISON_VARIANT_TARGET) {
      comparison_state.stats.set_num_unmatched_target_tensors(
          comparison_state.stats.num_unmatched_target_tensors() + 1);
    } else {
      comparison_state.stats.set_num_unmatched_baseline_tensors(
          comparison_state.stats.num_unmatched_baseline_tensors() + 1);
    }
  }
  absl::Cleanup cleanup([this, all_tracked_modules_fully_finished] {
    if (all_tracked_modules_fully_finished) {
      all_modules_finished_notification_.Notify();
    }
  });
  absl::Status status =
      OnHloModuleFinished(hlo_module_name, comparison_state.stats);
  finish_notification->Notify();
  comparison_states_.erase(hlo_module_name);
  return status;
}

absl::Status ComparisonManager::WaitUntilAllModulesAreFinished() {
  all_modules_finished_notification_.WaitForNotification();
  absl::MutexLock lock(mutex_);
  comparison_states_.clear();
  return absl::OkStatus();
}
}  // namespace xla::numerics::comparison
