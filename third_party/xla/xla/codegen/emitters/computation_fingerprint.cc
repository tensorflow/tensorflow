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

#include "xla/codegen/emitters/computation_fingerprint.h"

#include <string>

#include "absl/strings/str_cat.h"
#include "absl/strings/str_join.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "xla/codegen/emitters/kernel_arguments.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_print_options.h"

namespace xla::emitters {

// Calculates a fingerprint of the kernel arguments, which can be used for
// checking reusability.
//
// For example 2 arguments that are aligned to 16 bytes, aliased and also
// written by the kernel will be represented as "0x16aw,0x16aw".
//
// Overlapping arguments are only marked aliased, if at least one of them is
// written and their buffers are not exactly the same.
// If 2 arguments' buffers are exactly the same, then they are not marked
// aliased, but have the same slice index, for example like this:
// "0x16,0x16,1x16w,1x16w". The example means that the 1st argument is the same
// as the 0th and the 3rd is the same as the 2nd.
std::string GetArgumentFingerprint(
    absl::Span<const emitters::KernelArgument> kernel_arguments) {
  return absl::StrJoin(kernel_arguments, ",",
                       [](std::string* s, const emitters::KernelArgument& arg) {
                         absl::StrAppend(s, arg.slice_index());
                         absl::StrAppend(s, "x");
                         absl::StrAppend(s, arg.alignment());
                         if (arg.aliased()) {
                           absl::StrAppend(s, "a");
                         }
                         if (arg.written()) {
                           absl::StrAppend(s, "w");
                         }
                       });
}

std::string GetComputationFingerprint(
    const HloComputation* fused_computation,
    absl::Span<const emitters::KernelArgument> kernel_arguments,
    absl::string_view discriminator) {
  // We have to print constants, because otherwise we would accidentally reuse
  // kernels which have different builtin constants.
  //
  // It is not a problem to recursively print sub-computations, because we don't
  // have them at this point.
  auto print_options = HloPrintOptions::Fingerprint()
                           .set_print_only_essential_constants(false)
                           .set_print_operand_shape(false);

  return absl::StrCat(discriminator, "(",
                      GetArgumentFingerprint(kernel_arguments), ")",
                      fused_computation->ToString(print_options));
}

}  // namespace xla::emitters
