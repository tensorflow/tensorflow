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

#include "xla/service/hlo_module_executable.h"

#include <memory>
#include <utility>

#include "absl/strings/string_view.h"
#include "xla/hlo/ir/hlo_input_output_alias_config.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/service/computation_layout.h"
#include "xla/service/hlo_module_config.h"
#include "xla/shape.h"
#include "xla/stream_executor/kernel_stats.h"

namespace xla {

HloModuleExecutable::HloModuleExecutable(std::shared_ptr<HloModule> hlo_module)
    : hlo_module_(std::move(hlo_module)) {}

const HloModuleConfig& HloModuleExecutable::module_config() const {
  return module().config();
}

Shape HloModuleExecutable::result_shape() const {
  return module().config().entry_computation_layout().result_shape();
}

ComputationLayout HloModuleExecutable::compute_computation_layout() const {
  return module().compute_computation_layout();
}

absl::string_view HloModuleExecutable::name() const {
  if (has_module()) {
    return module().name();
  }
  return "<unknown executable>";
}

const HloInputOutputAliasConfig&
HloModuleExecutable::input_output_alias_config() const {
  return module().input_output_alias_config();
}

const HloBufferDonorConfig& HloModuleExecutable::buffer_donor_config() const {
  return module().buffer_donor_config();
}

void HloModuleExecutable::set_module_stats(ModuleStats module_stats) {
  module_stats_ = std::move(module_stats);
}

}  // namespace xla
