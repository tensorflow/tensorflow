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
#ifndef XLA_CODEGEN_EMITTERS_FUSION_WRAPPER_BASE_H_
#define XLA_CODEGEN_EMITTERS_FUSION_WRAPPER_BASE_H_

#include "absl/container/flat_hash_set.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/pass/hlo_pass_interface.h"

namespace xla {
namespace emitters {

// Wraps single operations in a fusion.
// The derived classes determine which operations to wrap and
// the type of the wrapper.
class FusionWrapperBase : public HloModulePass {
 public:
  virtual bool MustWrapInstruction(HloOpcode opcode) = 0;
  virtual HloInstruction::FusionKind ChooseFusionKind(
      const HloInstruction& producer, const HloInstruction& consumer) {
    return HloInstruction::FusionKind::kLoop;
  };

  using HloPassInterface::Run;
  absl::StatusOr<bool> Run(
      HloModule* module,
      const absl::flat_hash_set<absl::string_view>& execution_threads) override;
};

}  // namespace emitters
}  // namespace xla

#endif  // XLA_CODEGEN_EMITTERS_FUSION_WRAPPER_BASE_H_
