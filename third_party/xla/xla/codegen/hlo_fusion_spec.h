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

#ifndef XLA_CODEGEN_HLO_FUSION_SPEC_H_
#define XLA_CODEGEN_HLO_FUSION_SPEC_H_

#include <cstdint>
#include <memory>
#include <utility>

#include "absl/container/inlined_vector.h"
#include "xla/hlo/utils/hlo_traversal.h"

namespace xla {

// A simple wrapper around HloFusionAdaptor that also stores the fusion roots
// and heroes.
class HloFusionSpec {
 public:
  HloFusionSpec(std::unique_ptr<HloFusionAdaptor> fusion,
                absl::InlinedVector<HloInstructionAdaptor, 2> fusion_roots,
                absl::InlinedVector<HloInstructionAdaptor, 2> fusion_heroes)
      : fusion_(std::move(fusion)),
        fusion_roots_(std::move(fusion_roots)),
        fusion_heroes_(std::move(fusion_heroes)) {}

  HloFusionSpec(HloFusionSpec&&) = default;
  HloFusionSpec& operator=(HloFusionSpec&&) = default;

  const HloFusionAdaptor& fusion() const { return *fusion_; }

  const absl::InlinedVector<HloInstructionAdaptor, 2>& fusion_roots() const {
    return fusion_roots_;
  }
  HloInstructionAdaptor fusion_root(int64_t i) const {
    return fusion_roots_[i];
  }
  int64_t fusion_root_count() const { return fusion_roots_.size(); }

  const absl::InlinedVector<HloInstructionAdaptor, 2>& fusion_heroes() const {
    return fusion_heroes_;
  }
  HloInstructionAdaptor fusion_hero(int64_t i) const {
    return fusion_heroes_[i];
  }

 private:
  // Owning pointer to the fusion adaptor object.
  std::unique_ptr<HloFusionAdaptor> fusion_;

  // A list of all roots of the fusion. The instruction adaptors have `fusion_`
  // as their parent and should not outlive `fusion_`.
  absl::InlinedVector<HloInstructionAdaptor, 2> fusion_roots_;

  // A list of all heroes of the fusion. The instruction adaptors have `fusion_`
  // as their parent and should not outlive `fusion_`.
  absl::InlinedVector<HloInstructionAdaptor, 2> fusion_heroes_;
};

}  // namespace xla

#endif  // XLA_CODEGEN_HLO_FUSION_SPEC_H_
