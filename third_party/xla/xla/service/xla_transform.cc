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

#include "xla/service/xla_transform.h"

#include <memory>
#include <utility>
#include <vector>

#include "absl/base/attributes.h"
#include "absl/base/const_init.h"
#include "absl/base/no_destructor.h"
#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "absl/synchronization/mutex.h"
#include "xla/tsl/platform/status_macros.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/ir/hlo_schedule.h"
#include "xla/service/hlo_verifier.h"
#include "xla/tsl/platform/logging.h"
#include "xla/tsl/platform/statusor.h"

namespace xla {

namespace {
absl::flat_hash_map<HloXlaTransform::PipelineStage,
                    std::vector<std::shared_ptr<HloXlaTransform>>>&
GetHloXlaTransformsInternal() {
  static absl::NoDestructor<
      absl::flat_hash_map<HloXlaTransform::PipelineStage,
                          std::vector<std::shared_ptr<HloXlaTransform>>>>
      out;
  return *out;
}

ABSL_CONST_INIT absl::Mutex transforms_mutex(absl::kConstInit);
}  // namespace

void RegisterHloXlaTransform(HloXlaTransform::PipelineStage stage,
                             std::shared_ptr<HloXlaTransform> transform) {
  absl::MutexLock transforms_lock(transforms_mutex);
  auto& transforms = GetHloXlaTransformsInternal();
  transforms[stage].emplace_back(std::move(transform));
}

std::vector<std::shared_ptr<HloXlaTransform>> GetHloXlaTransforms(
    HloXlaTransform::PipelineStage stage) {
  absl::MutexLock transforms_lock(transforms_mutex);
  auto& transforms = GetHloXlaTransformsInternal();
  return transforms[stage];
}

bool ClearHloXlaTransforms() {
  absl::MutexLock transforms_lock(transforms_mutex);
  auto& transforms = GetHloXlaTransformsInternal();
  if (transforms.empty()) {
    return false;
  }
  transforms.clear();
  return true;
}

bool ClearHloXlaTransform(HloXlaTransform::PipelineStage stage,
                          absl::string_view name) {
  absl::MutexLock transforms_lock(transforms_mutex);
  auto& transforms_map = GetHloXlaTransformsInternal();
  auto it = transforms_map.find(stage);
  if (it == transforms_map.end()) {
    return false;
  }
  auto& stage_transforms = it->second;
  for (auto transform_it = stage_transforms.begin();
       transform_it != stage_transforms.end(); ++transform_it) {
    if ((*transform_it)->name() == name) {
      stage_transforms.erase(transform_it);
      if (stage_transforms.empty()) {
        transforms_map.erase(it);
      }
      return true;
    }
  }
  return false;
}

absl::StatusOr<bool> ApplyXlaTransformsToModule(
    HloXlaTransform::PipelineStage stage, xla::HloModule* module) {
  std::vector<std::shared_ptr<HloXlaTransform>> transforms;
  {
    absl::MutexLock transforms_lock(transforms_mutex);
    auto& transforms_map = GetHloXlaTransformsInternal();
    auto it = transforms_map.find(stage);
    if (it == transforms_map.end()) {
      return false;
    }
    transforms = it->second;
  }
  bool changed = false;
  for (auto& transform : transforms) {
    auto status_or_bool = transform->Transform(module);
    if (!status_or_bool.status().ok()) {
      return status_or_bool.status();
    }
    changed |= status_or_bool.value();
  }
  return changed;
}

absl::StatusOr<bool> ApplyXlaTransforms::RunImpl(
    HloModule* module,
    const absl::flat_hash_set<absl::string_view>& execution_threads) {
  VLOG(1) << "ApplyXlaTransforms ENTRY";
  XLA_VLOG_LINES(1, module->ToString());
  ASSIGN_OR_RETURN(bool changed, ApplyXlaTransformsToModule(stage_, module));
  if (changed) {
    HloVerifier verifier(/*layout_sensitive=*/false,
                         /*allow_mixed_precision=*/true);
    auto verifier_status = verifier.Run(module);
    if (!verifier_status.status().ok()) {
      return verifier_status.status();
    }
  }
  VLOG(1) << "ApplyXlaTransforms EXIT";
  XLA_VLOG_LINES(1, module->ToString());
  return changed;
}

absl::Status UpdateHloModuleFromProto(HloModule* module,
                                      const HloModuleProto& transformed_proto) {
  ASSIGN_OR_RETURN(auto temp_module, HloModule::CreateFromProto(
                                         transformed_proto, module->config()));

  // Capture schedule from temp_module if it has one.
  absl::flat_hash_map<HloComputation*, HloInstructionSequence> comp_to_sequence;
  if (temp_module->has_schedule()) {
    absl::flat_hash_map<int64_t, HloComputation*> temp_comp_map;
    for (HloComputation* comp : temp_module->computations()) {
      temp_comp_map[comp->unique_id()] = comp;
    }
    for (const auto& [comp_id, sequence] :
         temp_module->schedule().sequences()) {
      comp_to_sequence[temp_comp_map[comp_id]] = sequence;
    }
  }

  HloComputation* new_entry = temp_module->entry_computation();
  module->MoveComputationsFrom(temp_module.get());
  module->ReplaceEntryComputation(new_entry);
  module->mutable_config().SetComputationLayoutIfExists(
      new_entry->ComputeProgramShape());

  RETURN_IF_ERROR(module->RemoveUnusedComputations());

  // Restore schedule if we captured one.
  if (!comp_to_sequence.empty()) {
    HloSchedule new_schedule(module);
    absl::flat_hash_set<HloComputation*> remaining_computations(
        module->computations().begin(), module->computations().end());
    for (auto& [comp, sequence] : comp_to_sequence) {
      if (remaining_computations.contains(comp)) {
        sequence.update_id_sequence();
        new_schedule.set_sequence(comp, std::move(sequence));
      }
    }
    RETURN_IF_ERROR(module->set_schedule(std::move(new_schedule)));
  }

  return absl::OkStatus();
}

}  // namespace xla
