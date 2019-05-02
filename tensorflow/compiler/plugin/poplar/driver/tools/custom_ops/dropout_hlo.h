/* Copyright 2019 Graphcore Ltd

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

#ifndef TENSORFLOW_COMPILER_PLUGIN_POPLAR_DRIVER_TOOLS_CUSTOM_OPS_DROPOUT_H
#define TENSORFLOW_COMPILER_PLUGIN_POPLAR_DRIVER_TOOLS_CUSTOM_OPS_DROPOUT_H

#include "tensorflow/compiler/plugin/poplar/driver/tools/custom_ops/hlo_poplar_instruction.h"

namespace xla {
namespace poplarplugin {

class HloDropoutInstruction : public HloPoplarInstruction {
 public:
  explicit HloDropoutInstruction(HloInstruction* operand, HloInstruction* seed,
                                 float rate, float scale, int32_t seed_modifier,
                                 bool should_use_user_seed);

  absl::flat_hash_set<int64> AllocatingIndices() const override;
  absl::flat_hash_map<int64, int64> LayoutDependencies() const override;
  uint64 NumberOfInplaceOperands() const override;

  bool IsPopOpsElementwise() const;

  // Probability of a given element being set to zero.
  float Rate() const { return rate; }

  // Scale to apply to all elements that aren't dropped out.
  float Scale() const { return scale; }

  // The seed modifier provided by the user.
  int32_t SeedModifier() const { return seed_modifier; }

  // Track whether or not we should use the user provided seed.
  bool IsUserSeed() const { return is_user_seed; }

 private:
  std::unique_ptr<HloInstruction> CloneWithNewOperandsImpl(
      const Shape& shape, absl::Span<HloInstruction* const>,
      HloCloneContext*) const override;

  float scale;
  float rate;
  int32_t seed_modifier;
  bool is_user_seed;
};

std::unique_ptr<HloInstruction> CreateDropout(HloInstruction* operand,
                                              HloInstruction* seed, float rate,
                                              float scale,
                                              uint32_t seed_modifier,
                                              bool should_use_user_seed);

}  // namespace poplarplugin
}  // namespace xla

#endif  // TENSORFLOW_COMPILER_PLUGIN_POPLAR_DRIVER_TOOLS_CUSTOM_OPS_DROPOUT_H
