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

#include <cstdint>
#include <string>
#include <string_view>
#include <vector>

#include "absl/container/flat_hash_set.h"
#include "absl/status/status.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"
#include "absl/strings/str_join.h"
#include "absl/strings/string_view.h"
#include "xla/hlo/ir/hlo_module_group.h"
#include "xla/hlo/pass/hlo_pass_interface.h"
#include "xla/service/dump.h"
#include "xla/status_macros.h"
#include "xla/util.h"
#include "tsl/platform/errors.h"
#include "tsl/platform/logging.h"
#include "tsl/platform/status.h"
#include "tsl/platform/statusor.h"
#include "tsl/profiler/lib/scoped_annotation.h"

namespace xla {

namespace {
using absl::Status;
using absl::StatusOr;
using std::string;
using std::string_view;

int16_t kLogLevel = 3;

void RecordPassStartMetadata(HloModule& module, string_view pass_name,
                             string_view pipeline_name) {
  module.metadata()->RecordPassStart();
  string pass_str = string(pass_name);
  string pipeline_str = string(pipeline_name);
  // An HloPassMetadata was just created so absl::Status should always be OK.
  TF_CHECK_OK(module.metadata()->set_current_pass_name(pass_str));
  TF_CHECK_OK(module.metadata()->set_current_pass_pipeline_name(pipeline_str));
}

void RecordPassStartMetadata(HloModuleGroup& module_group,
                             string_view pass_name, string_view pipeline_name) {
  for (HloModule* module : module_group.modules()) {
    RecordPassStartMetadata(*module, pass_name, pipeline_name);
  }
}

absl::Status AttemptRecordPassEndMetadata(HloModule& module,
                                          bool module_changed) {
  // Module id is set here instead of RecordPassStartMetadata because it may
  // change in the middle of the pass, and we want the final id.
  TF_RETURN_IF_ERROR(
      module.metadata()->set_current_pass_module_id(module.unique_id()));
  TF_RETURN_IF_ERROR(
      module.metadata()->set_current_pass_module_changed(module_changed));
  TF_RETURN_IF_ERROR(module.metadata()->RecordPassEnd());
  return absl::OkStatus();
}

void RecordPassEndMetadata(HloModule& module, bool module_changed) {
  absl::Status status = AttemptRecordPassEndMetadata(module, module_changed);
  if (!status.ok()) {
    VLOG(kLogLevel) << status;
  }
}

Status AttemptRecordPassEndMetadata(HloModuleGroup& module_group,
                                    bool module_changed) {
  for (HloModule* module : module_group.modules()) {
    for (HloModule* other_module : module_group.modules()) {
      TF_RETURN_IF_ERROR(
          module->metadata()->add_current_pass_module_group_module_id(
              other_module->unique_id()));
    }
    TF_RETURN_IF_ERROR(AttemptRecordPassEndMetadata(*module, module_changed));
  }
  return absl::OkStatus();
}

void RecordPassEndMetadata(HloModuleGroup& module_group, bool module_changed) {
  Status status = AttemptRecordPassEndMetadata(module_group, module_changed);
  if (!status.ok()) {
    VLOG(kLogLevel) << status;
  }
}

// Helpers which run the given passes on the given HLO construct. Only
// computations with specified `execution_threads` are considered by the pass,
// empty thread list means all `execution_threads` are considered. These
// helpers enable templating of the core of the pipeline logic by providing
// HloModule and HloModuleGroup specific methods with the same name.
StatusOr<bool> RunHelper(
    HloPassInterface* pass, HloModule* module,
    const absl::flat_hash_set<absl::string_view>& execution_threads) {
  TF_ASSIGN_OR_RETURN(bool changed, pass->Run(module, execution_threads));
  module->Cleanup();
  return changed;
}

StatusOr<bool> RunHelper(
    HloPassInterface* pass, HloModuleGroup* module_group,
    const absl::flat_hash_set<absl::string_view>& execution_threads) {
  TF_ASSIGN_OR_RETURN(bool changed,
                      pass->RunOnModuleGroup(module_group, execution_threads));
  module_group->Cleanup();
  return changed;
}

// Maybe dumps the given module or module group depending on flag values
// contained in DebugOptions of module config. If it is dumped, saves the
// filenames of the dumps into module metadata.
void MaybeDumpHloAndSaveFilenames(HloModule& module, string_view pipeline_name,
                                  string_view after_pass_name,
                                  string_view before_pass_name) {
  for (const string& filename : DumpHloModuleBetweenPassesIfEnabled(
           pipeline_name, before_pass_name, after_pass_name, module)) {
    absl::Status status =
        module.metadata()->add_current_pass_dump_filename(filename);
    if (!status.ok()) {
      VLOG(kLogLevel) << status;
    }
  }
}

void MaybeDumpHloAndSaveFilenames(HloModuleGroup& module_group,
                                  absl::string_view pipeline_name,
                                  absl::string_view after_pass_name,
                                  absl::string_view before_pass_name) {
  for (HloModule* module : module_group.modules()) {
    MaybeDumpHloAndSaveFilenames(*module, pipeline_name, after_pass_name,
                                 before_pass_name);
  }
}

string UniqueId(const HloModule& mod) {
  return std::to_string(mod.unique_id());
}
string UniqueId(const HloModuleGroup& group) {
  return absl::StrJoin(group.modules(), "-",
                       [](string* out, const HloModule* mod) {
                         out->append(std::to_string(mod->unique_id()));
                       });
}

static constexpr absl::string_view kPipelineStart = "pipeline-start";
static constexpr absl::string_view kPipelineEnd = "pipeline-end";

}  // namespace

template <typename HloT>
absl::Status HloPassPipeline::BeforePipeline(
    HloT* hlo, const absl::flat_hash_set<absl::string_view>& threads,
    std::vector<HloPassInterface*>& passes) {
  string_view pipeline_name = name();

  tsl::profiler::ScopedAnnotation annotation{[&] {
    return absl::StrFormat("XlaPassPipeline:#name=%s,module=%s,program_id=%s#",
                           pipeline_name, hlo->name(), UniqueId(*hlo));
  }};

  TF_RETURN_IF_ERROR(RunInvariantCheckers(hlo, kPipelineStart, threads));

  RecordPassStartMetadata(*hlo, kPipelineStart, pipeline_name);
  auto before_pass_name =
      passes.empty() ? kPipelineEnd : passes.front()->name();
  MaybeDumpHloAndSaveFilenames(*hlo, pipeline_name, kPipelineStart,
                               before_pass_name);
  RecordPassEndMetadata(*hlo, false);
  return absl::OkStatus();
}

template <typename HloT>
void HloPassPipeline::BeforeEachPass(HloT* hlo, HloPassInterface* pass) {
  string_view pass_name = pass->name();
  XLA_SCOPED_LOGGING_TIMER(absl::StrCat("HLO pass: ", pass_name));

  tsl::profiler::ScopedAnnotation annotation{[&] {
    return absl::StrFormat("XlaPass:#name=%s,module=%s,program_id=%s#",
                           pass_name, hlo->name(), UniqueId(*hlo));
  }};

  VLOG(kLogLevel) << "HLO pass " << pass_name << " / Module hash "
                  << absl::HashOf(*hlo);
  if (!pass->IsPassPipeline()) {
    compilation_stats_->StartPass(pass_name);
  }
  RecordPassStartMetadata(*hlo, pass_name, name());
}

template <typename HloT>
absl::StatusOr<bool> HloPassPipeline::AfterEachPass(
    HloT* hlo, const absl::flat_hash_set<absl::string_view>& threads,
    HloPassInterface* pass, string_view next_pass_name, string_view dump_regex,
    absl::StatusOr<bool> pass_result) {
  string_view pass_name = pass->name();
  if (!pass_result.ok()) {
    compilation_stats_->RecordPassError(
        pass_name, absl::StatusCodeToString(pass_result.status().code()));
    return pass_result;
  }
  bool pass_changed = pass_result.value();

  if (!dump_regex.empty() && (pass_changed || dump_regex != ".*")) {
    MaybeDumpHloAndSaveFilenames(*hlo, name(), pass_name, next_pass_name);
  }
  RecordPassEndMetadata(*hlo, pass_changed);

  if (pass_changed) {
    VLOG(kLogLevel) << "Pass caused changes " << pass_name;
    absl::Status status = RunInvariantCheckers(hlo, pass_name, threads);
    if (!status.ok()) {
      compilation_stats_->RecordPassError(
          pass_name, absl::StatusCodeToString(status.code()));
      return status;
    }
  }
  if (!pass->IsPassPipeline()) {
    compilation_stats_->EndPass(pass_name);
  }
  return pass_result;
}

template <typename HloT>
absl::StatusOr<bool> HloPassPipeline::RunPassesInternal(
    HloT* hlo, const DebugOptions& debug_options,
    const absl::flat_hash_set<absl::string_view>& threads) {
  auto passes = GetEnabledPasses(debug_options);
  // Copy string by value since debug options could get clobbered in an hlo
  // module group pass.
  string_view dump_regex = debug_options.xla_dump_hlo_pass_re();

  TF_RETURN_IF_ERROR(BeforePipeline(hlo, threads, passes));

  bool changed = false;
  for (int i = 0; i < passes.size(); i++) {
    HloPassInterface* pass = passes[i];
    string_view next_pass_name =
        i + 1 >= passes.size() ? kPipelineEnd : passes[i + 1]->name();

    BeforeEachPass(hlo, pass);
    absl::StatusOr<bool> result = RunHelper(pass, hlo, threads);
    TF_ASSIGN_OR_RETURN(
        bool pass_changed,
        AfterEachPass(hlo, threads, pass, next_pass_name, dump_regex, result));
    changed |= pass_changed;
  }
  return changed;
}

template <typename HloT>
Status HloPassPipeline::RunInvariantCheckers(
    HloT* hlo, string_view after_pass_name,
    const absl::flat_hash_set<absl::string_view>& execution_threads) {
  for (auto& invariant_checker : invariant_checkers_) {
    absl::StatusOr<bool> result =
        RunHelper(invariant_checker.get(), hlo, execution_threads);

    string prefix = absl::StrFormat("Invariant checker '%s' after pass '%s' ",
                                    invariant_checker->name(), after_pass_name);
    if (!result.ok()) {
      string message = absl::StrCat(prefix, " failed with error ",
                                    result.status().message());
      VLOG(kLogLevel) << message << " on HLO: \n" << hlo->ToString();
      return tsl::errors::CreateWithUpdatedMessage(result.status(), message);
    }
    TF_RET_CHECK(!result.value())
        << prefix << ": invariant checkers must not change the graph";

    VLOG(kLogLevel) << prefix << " finished successfully.";
  }
  return absl::OkStatus();
}

std::vector<HloPassInterface*> HloPassPipeline::GetEnabledPasses(
    const DebugOptions& options) {
  if (options.xla_disable_all_hlo_passes()) {
    VLOG(1) << "*All* passes disabled by --xla_disable_all_hlo_passes.";
    return {};
  }

  absl::flat_hash_set<string> disabled_pass_names(
      options.xla_disable_hlo_passes().begin(),
      options.xla_disable_hlo_passes().end());

  absl::flat_hash_set<string> enabled_pass_names(
      options.xla_enable_hlo_passes_only().begin(),
      options.xla_enable_hlo_passes_only().end());

  if (!disabled_pass_names.empty()) {
    VLOG(1) << "Passes disabled by --xla_disable_hlo_passes: "
            << absl::StrJoin(disabled_pass_names, ", ");
  }

  if (!enabled_pass_names.empty()) {
    VLOG(1) << "Passes enabled by --xla_enable_hlo_passes_only: "
            << absl::StrJoin(enabled_pass_names, ", ");
  }

  CHECK(disabled_pass_names.empty() || enabled_pass_names.empty())
      << "Cannot specify both --xla_disable_hlo_passes and "
         "--xla_enable_hlo_passes_only";

  if (disabled_pass_names.contains(name())) {
    VLOG(1) << "Disable whole pipeline: " << name();
    return {};
  }

  if (enabled_pass_names.contains(name())) {
    VLOG(1) << "Enable whole pipeline: " << name();
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

absl::StatusOr<bool> HloPassPipeline::Run(
    HloModule* module,
    const absl::flat_hash_set<absl::string_view>& execution_threads) {
  run_called_ = true;

  VLOG(1) << "Running HLO pass pipeline on module " << module->name() << ": "
          << name();

  return RunPassesInternal(module, module->config().debug_options(),
                           execution_threads);
}

absl::StatusOr<bool> HloPassPipeline::RunOnModuleGroup(
    HloModuleGroup* module_group,
    const absl::flat_hash_set<absl::string_view>& execution_threads) {
  run_called_ = true;

  VLOG(1) << "Running HLO pass pipeline on module group "
          << module_group->name() << ": " << name();

  if (module_group->modules().empty()) {
    VLOG(1) << "Module group is empty. Nothing to do.";
    return false;
  }

  return RunPassesInternal(module_group,
                           module_group->module(0).config().debug_options(),
                           execution_threads);
}

}  // namespace xla
