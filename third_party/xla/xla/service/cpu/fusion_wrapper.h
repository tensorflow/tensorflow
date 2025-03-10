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

#ifndef XLA_SERVICE_CPU_FUSION_WRAPPER_H_
#define XLA_SERVICE_CPU_FUSION_WRAPPER_H_

#include "absl/strings/string_view.h"
#include "xla/codegen/emitters/fusion_wrapper_base.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_opcode.h"

namespace xla {
namespace cpu {

// Wraps certain HLO ops with a fusion op, so that the fusion emitter can
// kick in.
class FusionWrapper : public emitters::FusionWrapperBase {
 public:
  explicit FusionWrapper() = default;
  ~FusionWrapper() override = default;

  absl::string_view name() const override { return "fusion-wrapper"; }

  bool MustWrapInstruction(HloOpcode opcode) override;
};

}  // namespace cpu
}  // namespace xla

#endif  // XLA_SERVICE_CPU_FUSION_WRAPPER_H_
