
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

#ifndef XLA_CODEGEN_EMITTERS_COMPUTATION_FINGERPRINT_H_
#define XLA_CODEGEN_EMITTERS_COMPUTATION_FINGERPRINT_H_

#include <string>

#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "xla/codegen/emitters/kernel_arguments.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_instructions.h"
#include "xla/service/buffer_assignment.h"

namespace xla::emitters {

// Calculates the fingerprint of a (fused_computation, kernel_arguments,
// discriminator) tuple.
//
// If a given fusion is implemented using multiple kernels, then for each
// kernel we should provide a discriminator, such as "init" and "impl".
//
// If the same fingerprint is returned twice, then we can reuse the kernel
// generated for the first computation.
std::string GetComputationFingerprint(
    const HloComputation* fused_computation,
    absl::Span<const emitters::KernelArgument> kernel_arguments,
    absl::string_view discriminator = "");

}  // namespace xla::emitters

#endif  // XLA_CODEGEN_EMITTERS_COMPUTATION_FINGERPRINT_H_
