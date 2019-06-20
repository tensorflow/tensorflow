/* Copyright 2019 The TensorFlow Authors. All Rights Reserved.

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

#ifndef TENSORFLOW_COMPILER_PLUGIN_POPLAR_DRIVER_TOOLS_CUSTOM_OPS_STATEFUL_GRADIENT_ACCUMULATE_H
#define TENSORFLOW_COMPILER_PLUGIN_POPLAR_DRIVER_TOOLS_CUSTOM_OPS_STATEFUL_GRADIENT_ACCUMULATE_H

#include "tensorflow/compiler/plugin/poplar/driver/tools/custom_ops/hlo_poplar_instruction.h"

namespace xla {
namespace poplarplugin {

class HloStatefulGradientAccumulate : public HloPoplarInstruction {
 public:
  explicit HloStatefulGradientAccumulate(HloInstruction* operand,
                                         int32 num_mini_batches);

  absl::flat_hash_set<int64> AllocatingIndices() const override;
  absl::flat_hash_map<int64, int64> LayoutDependencies() const override;
  uint64 NumberOfInplaceOperands() const override;

  bool IsPopOpsElementwise() const;

  // The number of mini batches which will be accumulated.
  int32 MiniBatchesToAccumulate() const { return num_mini_batches_; }

 private:
  std::unique_ptr<HloInstruction> CloneWithNewOperandsImpl(
      const Shape& shape, absl::Span<HloInstruction* const>,
      HloCloneContext*) const override;

  int32 num_mini_batches_;
};

std::unique_ptr<HloInstruction> CreateStatefulGradientAccumulation(
    HloInstruction* operand, int32 num_mini_batches);

}  // namespace poplarplugin
}  // namespace xla

#endif  // TENSORFLOW_COMPILER_PLUGIN_POPLAR_DRIVER_TOOLS_CUSTOM_OPS_STATEFUL_GRADIENT_ACCUMULATE_H
