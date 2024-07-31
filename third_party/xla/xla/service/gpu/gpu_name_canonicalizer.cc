/* Copyright 2024 The OpenXLA Authors.
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

#include "xla/service/gpu/gpu_name_canonicalizer.h"

#include <cstddef>
#include <string>

#include "absl/container/flat_hash_set.h"
#include "absl/log/check.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_module.h"

namespace xla::gpu {
namespace {

// This is the separator that we use to determine what is the meaningful prefix
// of the instruction/computation.
constexpr absl::string_view kSeparator = ".";

std::string GetNamePrefix(absl::string_view name) {
  std::string root = std::string(name);
  size_t separator_index = name.find(kSeparator);
  if (separator_index != std::string::npos && separator_index > 0 &&
      separator_index < root.size() - 1) {
    root = root.substr(0, separator_index);
  }
  return root;
}

void RenameComputationAndInstructions(HloComputation& computation) {
  computation.parent()->SetAndUniquifyComputationName(
      &computation, GetNamePrefix(computation.name()));
  for (HloInstruction* instr : computation.instructions()) {
    computation.parent()->SetAndUniquifyInstrName(instr,
                                                  GetNamePrefix(instr->name()));
  }
}

}  // namespace

absl::StatusOr<bool> GpuNameCanonicalizer::Run(
    HloModule* module,
    const absl::flat_hash_set<absl::string_view>& execution_threads) {
  for (HloComputation* computation :
       module->MakeComputationPostOrder(execution_threads)) {
    RenameComputationAndInstructions(*computation);
  }

  return false;  // semantically IR is not supposed to change
}

}  // namespace xla::gpu
