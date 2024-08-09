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

#ifndef XLA_SERVICE_CONVERT_MEMORY_PLACEMENT_TO_INTERNAL_ANNOTATIONS_H_
#define XLA_SERVICE_CONVERT_MEMORY_PLACEMENT_TO_INTERNAL_ANNOTATIONS_H_

#include <functional>
#include <optional>
#include <string>
#include <vector>

#include "absl/container/flat_hash_set.h"
#include "absl/strings/string_view.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/service/hlo_pass_interface.h"

namespace xla {

class ConvertMemoryPlacementToInternalAnnotations : public HloModulePass {
 public:
  explicit ConvertMemoryPlacementToInternalAnnotations()
      : ConvertMemoryPlacementToInternalAnnotations(
            [](const HloInstruction& host_call) -> bool { return false; }) {}
  explicit ConvertMemoryPlacementToInternalAnnotations(
      std::function<bool(const HloInstruction&)> is_host_compute_call)
      : is_host_compute_call_(is_host_compute_call) {}

  absl::string_view name() const override {
    return "convert-memory-placement-to-internal-annotations";
  }
  using HloPassInterface::Run;
  absl::StatusOr<bool> Run(
      HloModule* module,
      const absl::flat_hash_set<absl::string_view>& execution_threads) override;

 private:
  // Returns true if the input instruction is a host compute instruction.
  std::function<bool(const HloInstruction&)> is_host_compute_call_;
};

}  // namespace xla

#endif  // XLA_SERVICE_CONVERT_MEMORY_PLACEMENT_TO_INTERNAL_ANNOTATIONS_H_
