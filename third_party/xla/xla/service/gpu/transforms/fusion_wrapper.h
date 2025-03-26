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
#ifndef XLA_SERVICE_GPU_TRANSFORMS_FUSION_WRAPPER_H_
#define XLA_SERVICE_GPU_TRANSFORMS_FUSION_WRAPPER_H_

#include "absl/strings/string_view.h"
#include "xla/codegen/emitters/fusion_wrapper_base.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/stream_executor/device_description.h"

namespace xla {
namespace gpu {

// Wraps leftover unfused instruction that are in the entry computation that
// have no LHLO equivalent in fusions containing just that instruction.
class FusionWrapper : public emitters::FusionWrapperBase {
 public:
  explicit FusionWrapper(const se::DeviceDescription& device_description)
      : device_description_(device_description) {}

  absl::string_view name() const override { return "fusion-wrapper"; }

  bool MustWrapInstruction(HloOpcode opcode) override;
  HloInstruction::FusionKind ChooseFusionKind(
      const HloInstruction& producer, const HloInstruction& consumer) override;

 private:
  const se::DeviceDescription& device_description_;
};

}  // namespace gpu
}  // namespace xla

#endif  // XLA_SERVICE_GPU_TRANSFORMS_FUSION_WRAPPER_H_
