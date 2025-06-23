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

#include "xla/service/gpu/transforms/thunk_pass_pipeline.h"

#include <memory>

#include "absl/log/log.h"
#include "absl/status/statusor.h"
#include "xla/backends/gpu/runtime/sequential_thunk.h"
#include "xla/service/hlo_module_config.h"
#include "xla/stream_executor/device_description.h"
#include "xla/tsl/platform/statusor.h"

namespace xla {
namespace gpu {

absl::StatusOr<bool> ThunkPassPipeline::Run(
    SequentialThunk* root_thunk, const HloModuleConfig& hlo_module_config,
    const se::DeviceDescription& device_info) {
  bool changed = false;
  for (const auto& pass : passes_) {
    VLOG(1) << "Running ThunkPass: " << pass->name();
    TF_ASSIGN_OR_RETURN(bool pass_changed,
                        pass->Run(root_thunk, hlo_module_config, device_info));
    changed |= pass_changed;
  }
  return changed;
}

}  // namespace gpu
}  // namespace xla
