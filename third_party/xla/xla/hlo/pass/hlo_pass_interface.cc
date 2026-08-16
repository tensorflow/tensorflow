/* Copyright 2025 The OpenXLA Authors.

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

#include "xla/hlo/pass/hlo_pass_interface.h"

#include <memory>
#include <string>

#include "absl/container/flat_hash_set.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/tsl/platform/env.h"
#include "xla/tsl/platform/logging.h"
#include "xla/xla.pb.h"

namespace xla {

namespace {

bool ShouldRunPass(HloModule* module, absl::string_view pass_name,
                   bool is_pipeline) {
  const DebugOptions& debug_options = module->config().debug_options();
  const std::string& starting_pass =
      debug_options.xla_run_hlo_passes_starting_from();
  if (starting_pass.empty() || module->hlo_passes_started()) {
    return true;
  }

  if (pass_name == starting_pass) {
    module->set_hlo_passes_started(true);
    VLOG(1) << "Starting HLO passes from " << starting_pass;
    return true;
  }

  // If the pass is a pipeline, we should go into the pipeline & skip individual
  // passes within the pipeline, not the pipeline itself.
  return is_pipeline;
}

}  // namespace

absl::StatusOr<bool> HloPassInterface::Run(
    HloModule* module,
    const absl::flat_hash_set<absl::string_view>& execution_threads) {
  if (!ShouldRunPass(module, name(), IsPassPipeline())) {
    return false;
  }

  auto* env = tsl::Env::Default();
  std::unique_ptr<tsl::ThreadNote> thread_note;
  thread_note = env->AddThreadNote(absl::StrCat("Running HLO pass on module ",
                                                module->name(), ": ", name()));
  return RunImpl(module, execution_threads);
}

absl::StatusOr<bool> HloPassInterface::Run(
    std::unique_ptr<HloModule>& module,
    const absl::flat_hash_set<absl::string_view>& execution_threads) {
  if (!ShouldRunPass(module.get(), name(), IsPassPipeline())) {
    return false;
  }

  auto* env = tsl::Env::Default();
  std::unique_ptr<tsl::ThreadNote> thread_note;
  thread_note = env->AddThreadNote(absl::StrCat("Running HLO pass on module ",
                                                module->name(), ": ", name()));
  return RunImpl(module, execution_threads);
}

}  // namespace xla
