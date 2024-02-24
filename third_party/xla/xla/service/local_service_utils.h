/* Copyright 2023 The OpenXLA Authors.

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

#ifndef XLA_SERVICE_LOCAL_SERVICE_UTILS_H_
#define XLA_SERVICE_LOCAL_SERVICE_UTILS_H_

#include <memory>

#include "absl/types/span.h"
#include "xla/client/executable_build_options.h"
#include "xla/client/xla_computation.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/service/backend.h"
#include "xla/service/service.h"

namespace xla {
// Validates the computation argument layouts, and returns the corresponding
// HloModuleConfig.
StatusOr<std::unique_ptr<HloModuleConfig>> GetHloModuleConfig(
    const XlaComputation& computation,
    absl::Span<const Shape* const> argument_layouts,
    const ExecutableBuildOptions& build_options,
    ServiceOptions* options = nullptr, Backend* backend = nullptr);
}  // namespace xla

#endif  // XLA_SERVICE_LOCAL_SERVICE_UTILS_H_
