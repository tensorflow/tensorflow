/* Copyright 2017 The OpenXLA Authors.

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

#include "xla/hlo/pass/hlo_pass_pipeline.h"

#include <cstddef>
#include <memory>
#include <optional>
#include <string>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "absl/hash/hash.h"
#include "absl/status/status.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"
#include "absl/strings/str_join.h"
#include "absl/strings/string_view.h"
#include "xla/tsl/platform/status_macros.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/pass/hlo_pass_interface.h"
#include "xla/hlo/tools/hlo_diff/hlo_diff_result.h"
#include "xla/hlo/tools/hlo_diff/hlo_gumgraph_diff.h"
#include "xla/service/dump.h"
#include "xla/status_macros.h"
#include "xla/tsl/platform/env.h"
#include "xla/tsl/platform/errors.h"
#include "xla/tsl/platform/logging.h"
#include "xla/tsl/platform/statusor.h"
#include "xla/util.h"
#include "xla/xla.pb.h"
#include "tsl/profiler/lib/scoped_annotation.h"
#include "tsl/profiler/lib/traceme.h"
#include "tsl/profiler/lib/traceme_encode.h"

namespace xla {

namespace {

void RecordPassStartMetadata(HloModule& module, const std::string& pass_name,
                             const std::string& pipeline_name) {
  module.metadata()->RecordPassStart();
  // An HloPassMetadata was just created so absl::Status should always be OK.
  CHECK_OK(module.metadata()->set_current_pass_name(pass_name));
  CHECK_OK(module.metadata()->set_current_pass_pipeline_name(pipeline_name));
}

absl::Status AttemptRecordPassEndMetadata(HloModule& module,
                                          const std::string& pass_name,
                                          bool module_changed) {
  // Module id is set here instead of RecordPassStartMetadata because it may
  // change in the middle of the pass, and we want the final id.
  RETURN_IF_ERROR(
      module.metadata()->set_current_pass_module_id(module.unique_id()));
  RETURN_IF_ERROR(
      module.metadata()->set_current_pass_module_changed(module_changed));
  RETURN_IF_ERROR(module.metadata()->RecordPassEnd());
  return absl::OkStatus();
}

void RecordPassEndMetadata(HloModule& module, const std::string& pass_name,
                           bool module_changed) {
  absl::Status status =
      AttemptRecordPassEndMetadata(module, pass_name, module_changed);
  if (!status.ok()) {
    LOG(FATAL) << status;
  }
}
}  // namespace

template <typename HloT>
absl::Status HloPassPipeline::RunInvariantCheckers(
    HloT hlo, absl::string_view after_pass_name,
    const absl::flat_hash_set<absl::string_view>& execution_threads) {
  tsl::profiler::TraceMe traceme("RunInvariantCheckers");
  for (auto& invariant_checker : invariant_checkers_) {
    VLOG(1) << "    Invariant checker " << invariant_checker->name();
    absl::StatusOr<bool> changed_status =
        RunHelper<HloT>(invariant_checker.get(), hlo, execution_threads);
    VLOG(1) << "    Invariant checker done " << invariant_checker->name();
    if (!changed_status.ok()) {
      VLOG(2) << "Failed invariant check:";
      XLA_VLOG_LINES(2, hlo->ToString());
      return tsl::errors::CreateWithUpdatedMessage(
          changed_status.status(),
          absl::StrCat(changed_status.status().message(), "\n\nFailed after ",
                       after_pass_name));
    }
    TF_RET_CHECK(!changed_status.value())
        << "invariant checkers must not change the graph";
  }
  return absl::OkStatus();
}

namespace {
std::string UniqueId(const HloModule& mod) {
  return std::to_string(mod.unique_id());
}

template <typename HloT>
static void VerifyPassChangedReport(const HloT hlo, bool pass_changed,
                                    const DebugOptions& debug_options,
                                    absl::string_view pass_name,
                                    absl::string_view pipeline_name,
                                    size_t hash_before) {
  size_t hash_after = absl::HashOf(*hlo);
  // Fail if pass changed HLO but has reported that it didn't.
  if (!pass_changed && hash_after != hash_before &&
      debug_options.xla_unsupported_crash_on_hlo_pass_silent_hlo_change()) {
    LOG(FATAL) << absl::StrFormat(
        "Pass '%s' in pipeline '%s' reported that it did not change the "
        "HLO but the hash of HLO was changed from %d to %d. HLO text "
        "after:\n%s",
        pass_name, pipeline_name, hash_before, hash_after, hlo->ToString());
  }
  // Fail if pass did not change HLO but has reported that it did.
  if (pass_changed && hash_after == hash_before &&
      debug_options.xla_unsupported_crash_on_hlo_pass_noop_change()) {
    LOG(FATAL) << absl::StrFormat(
        "Pass '%s' in pipeline '%s' reported that it changed the HLO but "
        "the hash of HLO was not updated. HLO text after:\n%s",
        pass_name, pipeline_name, hlo->ToString());
  }
}

absl::Status TrackModifyingPasses(HloModule& hlo,
                                  const HloModule& before_module,
                                  const DebugOptions& debug_options,
                                  absl::string_view pass_name) {
  ASSIGN_OR_RETURN(hlo_diff::HloGumgraphDiffResults diff_results,
                   hlo_diff::ComputeDiff(before_module, hlo));
  if (!diff_results.diff_result) {
    return absl::OkStatus();
  }

  DebugOptions::ModifyingPassTrackMode track_mode =
      debug_options.xla_track_modifying_passes();
  absl::flat_hash_map<const HloInstruction*, const HloInstruction*>
      right_to_left;
  if (track_mode == DebugOptions::FULL_PATH) {
    for (const auto& [left, right] :
         diff_results.diff_result->changed_instructions) {  // NOLINT
      right_to_left[right] = left;
    }
  }

  for (const auto& [inst_const, diff_type] :
       diff_results.diff_result->right_diff_codes) {  // NOLINT
    if (diff_type != hlo_diff::DiffType::kChanged &&
        diff_type != hlo_diff::DiffType::kUnmatched) {
      continue;
    }

    HloComputation* comp =
        hlo.GetComputationWithName(inst_const->parent()->name());
    if (comp == nullptr) {
      continue;
    }
    HloInstruction* inst = comp->GetInstructionWithName(inst_const->name());
    if (inst == nullptr) {
      continue;
    }

    if (track_mode == DebugOptions::LAST_PASS) {
      inst->set_frontend_attribute("_xla_last_modifying_pass", pass_name);
      continue;
    }

    std::optional<std::string> attr;
    if (auto it = right_to_left.find(inst_const);
        diff_type == hlo_diff::DiffType::kChanged &&
        it != right_to_left.end()) {
      attr = it->second->get_frontend_attribute("_xla_modifying_passes");
    } else {
      attr = inst_const->get_frontend_attribute("_xla_modifying_passes");
    }

    inst->set_frontend_attribute(
        "_xla_modifying_passes",
        attr ? absl::StrCat(*attr, ",", pass_name) : pass_name);
  }
  return absl::OkStatus();
}

}  // namespace

template <typename HloT>
absl::StatusOr<bool> HloPassPipeline::RunPassesInternal(
    HloT hlo, const DebugOptions& debug_options,
    const absl::flat_hash_set<absl::string_view>& execution_threads) {
  auto* env = tsl::Env::Default();
  std::unique_ptr<tsl::ThreadNote> thread_note;
  thread_note = env->AddThreadNote(absl::StrCat(
      "Running HLO pass pipeline on module ", hlo->name(), ": ", name()));
  auto passes = GetEnabledPasses(debug_options);
  // Copy string by value since debug options could get clobbered in an hlo
  // module group pass.
  std::string dump_regex = debug_options.xla_dump_hlo_pass_re();
  static constexpr absl::string_view kPipelineStart = "pipeline-start";
  static constexpr absl::string_view kPipelineEnd = "pipeline-end";
  std::string pipeline_name = std::string(name());
  tsl::profiler::ScopedAnnotation annotation{[&] {
    return absl::StrFormat("XlaPassPipeline:#name=%s,module=%s,program_id=%s#",
                           pipeline_name, hlo->name(), UniqueId(*hlo));
  }};

  RETURN_IF_ERROR(
      RunInvariantCheckers<HloT>(hlo, kPipelineStart, execution_threads));

  RecordPassStartMetadata(*hlo, std::string(kPipelineStart), pipeline_name);
  MaybeDumpHloAndSaveFilenames(*hlo,
                               /*after_pass_name=*/kPipelineStart,
                               /*before_pass_name=*/passes.empty()
                                   ? kPipelineEnd
                                   : passes.front()->name());
  RecordPassEndMetadata(*hlo, std::string(kPipelineStart),
                        /*module_changed=*/false);

  bool changed = false;
  bool verify_pass_changed_report =
      debug_options.xla_unsupported_crash_on_hlo_pass_silent_hlo_change() ||
      debug_options.xla_unsupported_crash_on_hlo_pass_noop_change();
  for (int i = 0, sz = passes.size(); i < sz; i++) {
    HloPassInterface* pass = passes[i];
    std::string pass_name = std::string(pass->name());
    XLA_SCOPED_LOGGING_TIMER(absl::StrCat("HLO pass: ", pass_name));
    tsl::profiler::ScopedAnnotation annotation{[&] {
      return absl::StrFormat("XlaPass:#name=%s,module=%s,program_id=%s#",
                             pass_name, hlo->name(), UniqueId(*hlo));
    }};
    VLOG(1) << "  HLO pass " << pass_name;
    std::optional<size_t> hash_before = std::nullopt;
    if (verify_pass_changed_report || VLOG_IS_ON(2)) {
      hash_before = absl::HashOf(*hlo);
      VLOG(2) << "  Module hash " << hash_before.value();
    }
    VLOG(2) << "  Number of instructions: " << hlo->instruction_count();
    tsl::profiler::TraceMe traceme(pass->name());
    if (!pass->IsPassPipeline()) {
      compilation_stats_->StartPass(pass_name);
    }
    RecordPassStartMetadata(*hlo, pass_name, pipeline_name);
    std::unique_ptr<HloModule> before_module;
    if (debug_options.xla_track_modifying_passes() !=
        DebugOptions::TRACK_NONE) {
      before_module = hlo->Clone(absl::StrCat("before_", pass_name));
    }

    auto status_or_changed = RunHelper<HloT>(pass, hlo, execution_threads);
    if (auto status = status_or_changed.status(); !status.ok()) {
      compilation_stats_->RecordPassError(
          pass_name, absl::StatusCodeToString(status.code()));
    }
    ASSIGN_OR_RETURN(bool pass_changed, status_or_changed);
    if (verify_pass_changed_report) {
      VerifyPassChangedReport<HloT>(hlo, pass_changed, debug_options, pass_name,
                                    pipeline_name, hash_before.value());
    }
    if (pass_changed && before_module) {
      RETURN_IF_ERROR(
          TrackModifyingPasses(*hlo, *before_module, debug_options, pass_name));
    }
    if (!dump_regex.empty() && (pass_changed || dump_regex != ".*")) {
      MaybeDumpHloAndSaveFilenames(*hlo,
                                   /*after_pass_name=*/pass_name,
                                   /*before_pass_name=*/i + 1 >= passes.size()
                                       ? kPipelineEnd
                                       : passes[i + 1]->name());
    }
    RecordPassEndMetadata(*hlo, pass_name, pass_changed);
    changed |= pass_changed;
    if (pass_changed) {
      VLOG(3) << "  Pass caused changes " << pass_name;
      auto status =
          RunInvariantCheckers<HloT>(hlo, pass_name, execution_threads);
      if (!status.ok()) {
        compilation_stats_->RecordPassError(
            pass_name, absl::StatusCodeToString(status.code()));
      }
      RETURN_IF_ERROR(status);
    }
    if (!pass->IsPassPipeline()) {
      compilation_stats_->EndPass(pass_name);
    }
  }
  return changed;
}

std::vector<HloPassInterface*> HloPassPipeline::GetEnabledPasses(
    const DebugOptions& debug_options) {
  if (debug_options.xla_disable_all_hlo_passes()) {
    VLOG(1) << "*All* passes disabled by --xla_disable_all_hlo_passes.";
    return {};
  }

  absl::flat_hash_set<std::string> disabled_pass_names(
      debug_options.xla_disable_hlo_passes().begin(),
      debug_options.xla_disable_hlo_passes().end());

  absl::flat_hash_set<std::string> enabled_pass_names(
      debug_options.xla_enable_hlo_passes_only().begin(),
      debug_options.xla_enable_hlo_passes_only().end());

  if (!disabled_pass_names.empty()) {
    VLOG(1) << "Passes disabled by --xla_disable_hlo_passes: "
            << absl::StrJoin(disabled_pass_names, ", ");
  }

  if (!enabled_pass_names.empty()) {
    VLOG(1) << "Passes enabled by --xla_enable_hlo_passes_only: "
            << absl::StrJoin(enabled_pass_names, ", ");
  }

  CHECK(disabled_pass_names.empty() || enabled_pass_names.empty());

  if (disabled_pass_names.contains(name())) {
    // Disable the full pass.
    VLOG(1) << "Disable the full pass: " << name();
    return {};
  }

  if (enabled_pass_names.contains(name())) {
    VLOG(1) << "Enable the full pass: " << name();
    // Enable the full pass.
    enabled_pass_names.clear();
  }

  std::vector<HloPassInterface*> enabled_passes;
  if (!enabled_pass_names.empty()) {
    for (auto& pass : passes_) {
      if (enabled_pass_names.contains(pass->name())) {
        enabled_passes.push_back(pass.get());
      }
    }
  } else {
    for (auto& pass : passes_) {
      if (!disabled_pass_names.contains(pass->name())) {
        enabled_passes.push_back(pass.get());
      }
    }
  }
  return enabled_passes;
}

void HloPassPipeline::MaybeDumpHloAndSaveFilenames(
    HloModule& module, absl::string_view after_pass_name,
    absl::string_view before_pass_name) {
  for (const std::string& filename : DumpHloModuleBetweenPassesIfEnabled(
           name(), before_pass_name, after_pass_name, module)) {
    absl::Status status =
        module.metadata()->add_current_pass_dump_filename(filename);
    if (!status.ok()) {
      LOG(FATAL) << status;
    }
  }
}

absl::StatusOr<bool> HloPassPipeline::RunImpl(
    HloModule* module,
    const absl::flat_hash_set<absl::string_view>& execution_threads) {
  run_called_ = true;

  VLOG(1) << "Running HLO pass pipeline on module " << module->name() << ": "
          << name();

  tsl::profiler::TraceMe traceme([&] {
    return tsl::profiler::TraceMeEncode(
        absl::StrCat(name(), " (", module->name(), ")"),
        {{"module", module->name()}});
  });
  // Copy debug options by value as passes may modify module config.
  DebugOptions debug_options = module->config().debug_options();
  return RunPassesInternal(module, debug_options, execution_threads);
}

absl::StatusOr<bool> HloPassPipeline::RunImpl(
    std::unique_ptr<HloModule>& module,
    const absl::flat_hash_set<absl::string_view>& execution_threads) {
  run_called_ = true;

  VLOG(1) << "Running HLO pass pipeline on module " << module->name() << ": "
          << name();

  tsl::profiler::TraceMe traceme([&] {
    return tsl::profiler::TraceMeEncode(
        absl::StrCat(name(), " (", module->name(), ")"),
        {{"module", module->name()}});
  });
  // Copy debug options by value as passes may modify module config.
  DebugOptions debug_options = module->config().debug_options();
  return RunPassesInternal<std::unique_ptr<HloModule>&>(module, debug_options,
                                                        execution_threads);
}

}  // namespace xla
