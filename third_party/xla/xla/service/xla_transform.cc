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

#include <atomic>
#include <cstdint>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "absl/base/attributes.h"
#include "absl/base/const_init.h"
#include "absl/base/no_destructor.h"
#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "absl/status/status.h"
#include "absl/status/status_macros.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "absl/synchronization/mutex.h"
#include "xla/debug_options_flags.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/service/compiler.h"
#include "xla/service/hlo.pb.h"
#include "xla/service/hlo_module_config.h"
#include "xla/service/hlo_verifier.h"
#include "xla/service/platform_util.h"
#include "xla/stream_executor/platform.h"
#include "xla/stream_executor/platform_manager.h"
#include "xla/tsl/platform/logging.h"
#include "xla/xla.pb.h"

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
    ABSL_ASSIGN_OR_RETURN(bool transform_changed, transform->Transform(module));
    changed |= transform_changed;
  }
  return changed;
}

ApplyXlaTransforms::ApplyXlaTransforms(
    HloXlaTransform::PipelineStage stage,
    std::unique_ptr<TargetVerifierMetadata> target_metadata)
    : stage_(stage) {
  static std::atomic<int64_t> next_id{0};
  name_ = absl::StrCat("apply-xla-transforms-", next_id.fetch_add(1));
  if (target_metadata == nullptr) {
    target_metadata = std::make_unique<DefaultVerifierMetadata>(
        HloVerifierOpts{}.WithLayoutSensitive(false).WithAllowMixedPrecision(
            true));
  }
  verifier_ = std::make_unique<HloVerifier>(std::move(target_metadata), name_);
}

ApplyXlaTransforms::~ApplyXlaTransforms() = default;

absl::StatusOr<bool> ApplyXlaTransforms::RunImpl(
    HloModule* module,
    const absl::flat_hash_set<absl::string_view>& execution_threads) {
  VLOG(1) << "ApplyXlaTransforms ENTRY";
  XLA_VLOG_LINES(1, module->ToString());
  ABSL_ASSIGN_OR_RETURN(bool changed, ApplyXlaTransformsToModule(stage_, module));
  if (changed) {
    ABSL_RETURN_IF_ERROR(verifier_->Run(module, execution_threads).status());
  }
  VLOG(1) << "ApplyXlaTransforms EXIT";
  XLA_VLOG_LINES(1, module->ToString());
  return changed;
}

absl::Status UpdateHloModuleFromProto(HloModule* module,
                                      const HloModuleProto& transformed_proto) {
  return module->UpdateFromProto(transformed_proto);
}

absl::StatusOr<xla::HloModuleMetadataProto> GetHloPassPipelineTrace(
    const xla::HloModuleProto& proto) {
  xla::DebugOptions debug_options = xla::GetDebugOptionsFromFlags();
  ABSL_ASSIGN_OR_RETURN(
      xla::HloModuleConfig config,
      xla::HloModule::CreateModuleConfigFromProto(proto, debug_options));
  ABSL_ASSIGN_OR_RETURN(std::unique_ptr<xla::HloModule> module,
                   xla::HloModule::CreateFromProto(proto, config));

  stream_executor::Platform* platform = nullptr;
  auto default_platform_or = xla::PlatformUtil::GetDefaultPlatform();
  if (default_platform_or.ok()) {
    platform = default_platform_or.value();
  } else {
    std::vector<std::string> platforms = {"TPU", "CUDA", "Host"};
    for (const auto& p_name : platforms) {
      auto p_or = stream_executor::PlatformManager::PlatformWithName(p_name);
      if (p_or.ok()) {
        platform = p_or.value();
        break;
      }
    }
  }
  if (platform == nullptr) {
    return absl::NotFoundError("No platform found for compilation");
  }

  ABSL_ASSIGN_OR_RETURN(std::unique_ptr<xla::Compiler> compiler,
                   xla::Compiler::GetForPlatform(platform->id()));

  xla::Compiler::CompileOptions compile_options;
  // We pass nullptr as executor, as the TPU compiler's RunHloPasses
  // doesn't require it (it gets target from module config).
  ABSL_ASSIGN_OR_RETURN(
      std::unique_ptr<xla::HloModule> optimized_module,
      compiler->RunHloPasses(std::move(module), /*executor=*/nullptr,
                             compile_options));

  return optimized_module->metadata()->proto();
}

}  // namespace xla
