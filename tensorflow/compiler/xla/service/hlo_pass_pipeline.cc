/* Copyright 2017 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/compiler/xla/service/hlo_pass_pipeline.h"

#include <functional>

#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "absl/strings/str_format.h"
#include "absl/strings/str_join.h"
#include "tensorflow/compiler/xla/service/dump.h"
#include "tensorflow/compiler/xla/service/hlo_graph_dumper.h"
#include "tensorflow/compiler/xla/service/hlo_proto_util.h"
#include "tensorflow/compiler/xla/status_macros.h"
#include "tensorflow/compiler/xla/types.h"
#include "tensorflow/compiler/xla/util.h"
#include "tensorflow/core/platform/logging.h"

namespace xla {

template <typename HloT>
Status HloPassPipeline::RunInvariantCheckers(
    HloT* hlo, absl::string_view after_pass_name) {
  for (auto& invariant_checker : invariant_checkers_) {
    VLOG(1) << "    Invariant checker " << invariant_checker->name();
    StatusOr<bool> changed_status = RunHelper(invariant_checker.get(), hlo);
    VLOG(1) << "    Invariant checker done " << invariant_checker->name();
    if (!changed_status.ok()) {
      VLOG(2) << "Failed invariant check:";
      XLA_VLOG_LINES(2, hlo->ToString());
      return Status(changed_status.status().code(),
                    absl::StrCat(changed_status.status().error_message(),
                                 "\n\nFailed after ", after_pass_name));
    }
    TF_RET_CHECK(!changed_status.ValueOrDie())
        << "invariant checkers must not change the graph";
  }
  return Status::OK();
}

template <typename HloT>
StatusOr<bool> HloPassPipeline::RunPassesInternal(
    HloT* hlo, absl::Span<HloPassInterface* const> passes) {
  string last_pass_name = "pipeline-start";
  TF_RETURN_IF_ERROR(RunInvariantCheckers(hlo, last_pass_name));
  bool changed = false;
  for (HloPassInterface* pass : passes) {
    absl::string_view pass_name = pass->name();
    VLOG(1) << "  HLO pass " << pass_name;
    MaybeDumpHlo(*hlo,
                 /*after_pass_name=*/last_pass_name,
                 /*before_pass_name=*/pass_name);
    if (!pass->IsPassPipeline()) {
      compilation_stats_->StartPass(pass_name);
    }
    TF_ASSIGN_OR_RETURN(bool pass_changed, RunHelper(pass, hlo));
    changed |= pass_changed;
    TF_RETURN_IF_ERROR(RunInvariantCheckers(hlo, pass_name));
    last_pass_name = string(pass_name);
    if (!pass->IsPassPipeline()) {
      compilation_stats_->EndPass(pass_name);
    }
  }
  MaybeDumpHlo(*hlo,
               /*after_pass_name=*/last_pass_name,
               /*before_pass_name=*/"pipeline-end");
  return changed;
}

std::vector<HloPassInterface*> HloPassPipeline::GetEnabledPasses(
    const DebugOptions& debug_options) {
  if (debug_options.xla_disable_all_hlo_passes()) {
    VLOG(1) << "*All* passes disabled by --xla_disable_all_hlo_passes.";
    return {};
  }

  absl::flat_hash_set<string> disabled_pass_names(
      debug_options.xla_disable_hlo_passes().begin(),
      debug_options.xla_disable_hlo_passes().end());

  absl::flat_hash_set<string> enabled_pass_names(
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

void HloPassPipeline::MaybeDumpHlo(const HloModule& module,
                                   absl::string_view after_pass_name,
                                   absl::string_view before_pass_name) {
  DumpHloModuleBetweenPassesIfEnabled(name(), before_pass_name, after_pass_name,
                                      module);
}

void HloPassPipeline::MaybeDumpHlo(const HloModuleGroup& module_group,
                                   absl::string_view after_pass_name,
                                   absl::string_view before_pass_name) {
  for (const HloModule* module : module_group.modules()) {
    MaybeDumpHlo(*module, after_pass_name, before_pass_name);
  }
}

StatusOr<bool> HloPassPipeline::Run(HloModule* module) {
  run_called_ = true;

  VLOG(1) << "Running HLO pass pipeline on module " << module->name() << ": "
          << name();

  return RunPassesInternal(module,
                           GetEnabledPasses(module->config().debug_options()));
}

StatusOr<bool> HloPassPipeline::RunOnModuleGroup(HloModuleGroup* module_group) {
  run_called_ = true;

  VLOG(1) << "Running HLO pass pipeline on module group "
          << module_group->name() << ": " << name();

  if (module_group->modules().empty()) {
    VLOG(1) << "Module group is empty. Nothing to do.";
    return false;
  }

  return RunPassesInternal(
      module_group,
      GetEnabledPasses(module_group->module(0).config().debug_options()));
}

}  // namespace xla
