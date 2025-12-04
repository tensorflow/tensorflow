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

#ifndef XLA_HLO_SEPARATE_COMPILATION_HLO_MODULE_LINKING_H_
#define XLA_HLO_SEPARATE_COMPILATION_HLO_MODULE_LINKING_H_

#include <memory>

#include "absl/base/nullability.h"
#include "absl/status/statusor.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/separate_compilation/hlo_linking_manifest.h"

namespace xla::separate_compilation {

// Link all the callees together with the `root_computation` in the provided
// module. Returns the pointer to the linked version of `root_computation`
// in the `module`.
absl::StatusOr<HloComputation* absl_nonnull> LinkComputationInto(
    HloModule* module, const HloLinkingManifest& linking_manifest,
    const HloComputation* absl_nonnull root_computation);

// Create a new module by linking computations starting from `root_computation`.
absl::StatusOr<std::unique_ptr<HloModule>> LinkComputation(
    const HloLinkingManifest& linking_manifest,
    const HloComputation* absl_nonnull root_computation);

}  // namespace xla::separate_compilation

#endif  // XLA_HLO_SEPARATE_COMPILATION_HLO_MODULE_LINKING_H_
