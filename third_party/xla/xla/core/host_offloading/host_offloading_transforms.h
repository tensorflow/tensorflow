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

#ifndef XLA_CORE_HOST_OFFLOADING_HOST_OFFLOADING_TRANSFORMS_H_
#define XLA_CORE_HOST_OFFLOADING_HOST_OFFLOADING_TRANSFORMS_H_

#include "absl/status/status.h"
#include "xla/hlo/ir/hlo_input_output_alias_config.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/service/hlo.pb.h"
#include "xla/shape.h"

namespace xla {

// Prepares HLO module for PJRT CPU client by converting to destination passing
// style computation by flattening tuple parameters and appending destination
// parameters.
//
// No changes are made if HLO module is already flattened and all outputs are
// aliased.
absl::Status RewriteToDestinationPassingStyle(
    HloModule* hlo_module, const ProgramShape& program_shape,
    const HloInputOutputAliasConfig& alias_config);

}  // namespace xla

#endif  // XLA_CORE_HOST_OFFLOADING_HOST_OFFLOADING_TRANSFORMS_H_
