/* Copyright 2019 The OpenXLA Authors.

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

#include "xla/tools/prepare_reference_module.h"

#include <functional>
#include <memory>

#include "absl/status/statusor.h"
#include "xla/debug_options_flags.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/transforms/despecializer.h"
#include "xla/service/hlo_module_config.h"
#include "xla/stream_executor/platform.h"
#include "xla/xla.pb.h"
#include "tsl/platform/errors.h"
#include "tsl/platform/status.h"

namespace xla {

absl::StatusOr<std::unique_ptr<HloModule>> PrepareReferenceModule(
    const HloModule& test_module, HloRunnerInterface* test_runner,
    const std::function<void(HloModuleConfig*)>& config_modifier_hook,
    const std::function<absl::Status(const HloModule&, HloRunnerInterface*,
                                     HloModule*)>& module_modifier_hook,
    bool skip_despecialization) {
  DebugOptions debug_options = GetDebugOptionsFromFlags();
  // The combination of fast math and optimizations leads to unsound code
  // transformations (see third_party/tensorflow/compiler/xla/xla.proto for
  // details).  The test platform should not change this from the default.
  debug_options.set_xla_cpu_enable_fast_math(false);
  debug_options.set_xla_gpu_enable_fast_min_max(false);
  HloModuleConfig reference_config = test_module.config();
  reference_config.set_debug_options(debug_options);
  if (config_modifier_hook) {
    config_modifier_hook(&reference_config);
  }
  std::unique_ptr<HloModule> reference_module =
      test_module.Clone("reference", reference_config);
  if (module_modifier_hook) {
    TF_RETURN_IF_ERROR(
        module_modifier_hook(test_module, test_runner, reference_module.get()));
  } else if (!skip_despecialization) {
    TF_RETURN_IF_ERROR(Despecializer().Run(reference_module.get()).status());
  }
  return std::move(reference_module);
}
};  // namespace xla
