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

namespace {

void RecordPassStartMetadata(HloModule& module, const std::string& pass_name,
                             const std::string& pipeline_name) {
  module.metadata()->RecordPassStart();
  // An HloPassMetadata was just created so Status should always be OK.
  TF_CHECK_OK(module.metadata()->set_current_pass_name(pass_name));
  TF_CHECK_OK(module.metadata()->set_current_pass_pipeline_name(pipeline_name));
}

void RecordPassStartMetadata(HloModuleGroup& module_group,
                             const std::string& pass_name,
                             const std::string& pipeline_name) {
  for (HloModule* module : module_group.modules()) {
    RecordPassStartMetadata(*module, pass_name, pipeline_name);
  }
}

Status AttemptRecordPassEndMetadata(HloModule& module,
                                    const std::string& pass_name,
                                    bool module_changed) {
  // Module id is set here instead of RecordPassStartMetadata because it may
  // change in the middle of the pass, and we want the final id.
  TF_RETURN_IF_ERROR(
      module.metadata()->set_current_pass_module_id(module.unique_id()));
  TF_RETURN_IF_ERROR(
      module.metadata()->set_current_pass_module_changed(module_changed));
  TF_RETURN_IF_ERROR(module.metadata()->RecordPassEnd());
  return Status::OK();
}

void RecordPassEndMetadata(HloModule& module, const std::string& pass_name,
                           bool module_changed) {
  Status status =
      AttemptRecordPassEndMetadata(module, pass_name, module_changed);
  if (!status.ok()) {
    LOG(FATAL) << status;
  }
}

Status AttemptRecordPassEndMetadata(HloModuleGroup& module_group,
                                    const std::string& pass_name,
                                    bool module_changed) {
  for (HloModule* module : module_group.modules()) {
    for (HloModule* other_module : module_group.modules()) {
      TF_RETURN_IF_ERROR(
          module->metadata()->add_current_pass_module_group_module_id(
              other_module->unique_id()));
    }
    TF_RETURN_IF_ERROR(
        AttemptRecordPassEndMetadata(*module, pass_name, module_changed));
  }
  return Status::OK();
}

void RecordPassEndMetadata(HloModuleGroup& module_group,
                           const std::string& pass_name, bool module_changed) {
  Status status =
      AttemptRecordPassEndMetadata(module_group, pass_name, module_changed);
  if (!status.ok()) {
    LOG(FATAL) << status;
  }
}

void SetInstructionMetadata(HloModule& module) {
  StatusOr<int64> pass_id = module.metadata()->current_pass_id();
  if (!pass_id.ok()) {
    LOG(FATAL) << pass_id.status();
  }
  for (xla::HloComputation* computation : module.computations()) {
    for (xla::HloInstruction* instruction : computation->instructions()) {
      if (instruction->metadata().creation_pass_id() == 0) {
        instruction->set_creation_pass_id(*pass_id);
      }
      if (instruction->metadata().logical_creation_pass_id() == 0) {
        instruction->set_logical_creation_pass_id(*pass_id);
      }
    }
  }
}

void SetInstructionMetadata(HloModuleGroup& module_group) {
  for (HloModule* module : module_group.modules()) {
    SetInstructionMetadata(*module);
  }
}

}  // namespace

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
  static constexpr absl::string_view kPipelineStart = "pipeline-start";
  static constexpr absl::string_view kPipelineEnd = "pipeline-end";
  std::string pipeline_name = std::string(name());

  TF_RETURN_IF_ERROR(RunInvariantCheckers(hlo, kPipelineStart));

  RecordPassStartMetadata(*hlo, std::string(kPipelineStart), pipeline_name);
  SetInstructionMetadata(*hlo);
  MaybeDumpHloAndSaveFilenames(*hlo,
                               /*after_pass_name=*/kPipelineStart,
                               /*before_pass_name=*/passes.empty()
                                   ? kPipelineEnd
                                   : passes.front()->name());
  RecordPassEndMetadata(*hlo, std::string(kPipelineStart),
                        /*module_changed=*/false);

  bool changed = false;
  for (int i = 0; i < passes.size(); i++) {
    HloPassInterface* pass = passes[i];
    XLA_SCOPED_LOGGING_TIMER(absl::StrCat("HLO pass: ", pass->name()));
    std::string pass_name = std::string(pass->name());
    VLOG(1) << "  HLO pass " << pass_name;
    VLOG(2) << "  Module hash " << hlo->Hash();
    if (!pass->IsPassPipeline()) {
      compilation_stats_->StartPass(pass_name);
    }
    RecordPassStartMetadata(*hlo, pass_name, pipeline_name);
    TF_ASSIGN_OR_RETURN(bool pass_changed, RunHelper(pass, hlo));
    SetInstructionMetadata(*hlo);
    MaybeDumpHloAndSaveFilenames(*hlo,
                                 /*after_pass_name=*/pass_name,
                                 /*before_pass_name=*/i + 1 >= passes.size()
                                     ? kPipelineEnd
                                     : passes[i + 1]->name());
    RecordPassEndMetadata(*hlo, pass_name, pass_changed);
    changed |= pass_changed;
    if (pass_changed) {
      VLOG(3) << "  Pass caused changes " << pass->name();
    }
    TF_RETURN_IF_ERROR(RunInvariantCheckers(hlo, pass_name));
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

void HloPassPipeline::MaybeDumpHloAndSaveFilenames(
    HloModule& module, absl::string_view after_pass_name,
    absl::string_view before_pass_name) {
  for (const std::string& filename : DumpHloModuleBetweenPassesIfEnabled(
           name(), before_pass_name, after_pass_name, module)) {
    Status status = module.metadata()->add_current_pass_dump_filename(filename);
    if (!status.ok()) {
      LOG(FATAL) << status;
    }
  }
}

void HloPassPipeline::MaybeDumpHloAndSaveFilenames(
    HloModuleGroup& module_group, absl::string_view after_pass_name,
    absl::string_view before_pass_name) {
  for (HloModule* module : module_group.modules()) {
    MaybeDumpHloAndSaveFilenames(*module, after_pass_name, before_pass_name);
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
