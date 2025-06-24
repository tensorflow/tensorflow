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

#ifndef XLA_SERVICE_GPU_TRANSFORMS_ADD_TRACKING_SUFFIX_TO_INSTRUCTION_NAMES_H_
#define XLA_SERVICE_GPU_TRANSFORMS_ADD_TRACKING_SUFFIX_TO_INSTRUCTION_NAMES_H_

#include "absl/container/flat_hash_set.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/pass/hlo_pass_interface.h"

namespace xla::gpu {

// Appends ".0" suffix to instruction names.
//
// Priority fusion pass duplicates instructions, and it's hard to match
// instructions before and after the run as they got renamed.
// To make debugging easier, we append ".0" suffix to instruction names
// and priority fusion pass will increment this suffix:
//
// Original: broadcast.123
// After this pass: broadcast.123.0
// After priority fusion: broadcast.123.1, broadcast.123.2, ...
//
// One can match instructions before and after by their original name.
class AddTrackingSuffixToInstructionNames : public HloModulePass {
 public:
  absl::string_view name() const override { return "rename-instructions"; }

  using HloPassInterface::Run;
  absl::StatusOr<bool> Run(
      HloModule* module,
      const absl::flat_hash_set<absl::string_view>& execution_threads) override;
};

}  // namespace xla::gpu

#endif  // XLA_SERVICE_GPU_TRANSFORMS_ADD_TRACKING_SUFFIX_TO_INSTRUCTION_NAMES_H_
